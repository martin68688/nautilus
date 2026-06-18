"""
Run 20260514_190327 Top1 推理脚本
LogLoss: 0.01097 (索引正确, 无泄露)
模型: DeBERTa-v3-large + MultiHeadAttention + Mean池化 + extra_features(text_length, punctuation)
Checkpoint: best_model_c14daad52e724126a542d40ac87af5ea.pt

用法: python infer_190327.py
"""

import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 路径配置
# ============================================================
INFERENCE_DIR = "/workspace/nautilus/mlevolve/inference"
TEST_CSV = f"{INFERENCE_DIR}/test.csv"
CHECKPOINT = f"{INFERENCE_DIR}/checkpoints/best_model_190327.pt"
OUTPUT_CSV = f"{INFERENCE_DIR}/submissions/run_190327_top1_full_8392.csv"
MAX_LENGTH = 512
BATCH_SIZE = 16

# ============================================================
# 模型定义 (与solution.py完全一致)
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")


class SpookyAuthorClassifier(nn.Module):
    def __init__(self, num_authors=3, dropout_rate=0.3):
        super().__init__()
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-large",
            num_labels=num_authors,
            output_hidden_states=True,
            output_attentions=False,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
        )
        for param in self.backbone.deberta.parameters():
            param.requires_grad = False
        for layer in self.backbone.deberta.encoder.layer[-8:]:
            for param in layer.parameters():
                param.requires_grad = True
        hidden_size = self.backbone.config.hidden_size
        self.head = nn.Linear(hidden_size + 2, num_authors)  # +2 for extra features
        # Multi-head attention pooling: 8 heads, each head dimension = hidden_size/8
        self.num_heads = 8
        self.head_dim = hidden_size // self.num_heads
        assert hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"
        # Learnable query tensor
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size))
        # MultiheadAttention: key and value come from mean-pooled layers (4 layers)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=self.num_heads,
            batch_first=False,
        )
        # LayerNorm for residual connection
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids, attention_mask, extra_features=None):
        outputs = self.backbone.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Extract all hidden states from all layers
        all_hidden_states = outputs.hidden_states  # tuple of (batch, seq_len, hidden_size)
        # Take the last 4 layers (excluding the embedding layer which is the first element)
        last_4_layers = all_hidden_states[-4:]  # list of 4 tensors

        # Apply mean pooling (masked) to each of the last 4 layers
        masked_pooled = []
        for layer_hidden in last_4_layers:
            # Expand attention_mask to match hidden state dimensions
            mask_expanded = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            # Zero out padding tokens and sum
            masked_sum = (layer_hidden * mask_expanded).sum(dim=1)  # (batch, hidden_size)
            # Count non-padding tokens
            token_counts = mask_expanded.sum(dim=1) + 1e-10  # (batch, 1)
            # Compute mean
            mean_pooled = masked_sum / token_counts  # (batch, hidden_size)
            masked_pooled.append(mean_pooled)

        # Stack pooled representations: (batch, 4, hidden_size)
        stacked_pooled = torch.stack(masked_pooled, dim=1)  # (batch, 4, hidden_size)

        # Multi-head attention pooling
        # Permute to (seq_len=4, batch, hidden_size) for batch_first=False
        stacked_pooled_t = stacked_pooled.permute(1, 0, 2)  # (4, batch, hidden_size)
        # Expand query to match batch size: (1, batch, hidden_size)
        query_t = self.query.expand(-1, stacked_pooled_t.size(1), -1)  # (1, batch, hidden_size)
        # Apply multihead attention: query = learned query, key=value=mean-pooled layers
        attn_output, _ = self.multihead_attn(
            query=query_t,
            key=stacked_pooled_t,
            value=stacked_pooled_t,
        )  # attn_output: (1, batch, hidden_size)
        # Squeeze sequence dimension: (batch, hidden_size)
        attn_output = attn_output.squeeze(0)  # (batch, hidden_size)

        # Global mean pool of last layer
        last_layer_hidden = all_hidden_states[-1]  # (batch, seq_len, hidden_size)
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        masked_sum = (last_layer_hidden * mask_expanded).sum(dim=1)
        token_counts = mask_expanded.sum(dim=1) + 1e-10
        global_mean_pool = masked_sum / token_counts  # (batch, hidden_size)

        # Residual connection: add global mean pool to attention output
        pooled_output = self.layer_norm(attn_output + global_mean_pool)  # (batch, hidden_size)

        # Concatenate extra features if provided
        if extra_features is not None:
            pooled_output = torch.cat([pooled_output, extra_features], dim=1)  # (batch, hidden_size+2)

        logits = self.head(pooled_output)
        return logits


class SpookyDataset(Dataset):
    def __init__(self, texts, tokenizer=None, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        # Compute text length (characters) and punctuation count
        text_length = len(text)
        punctuation_count = sum(1 for ch in text if ch in '. , ! ? ; :')
        extra_features = torch.tensor([text_length / 1000.0, punctuation_count / 100.0], dtype=torch.float)
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors=None,
        )
        item = {
            "input_ids": torch.tensor(encodings["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(
                encodings["attention_mask"], dtype=torch.long
            ),
            "extra_features": extra_features,
        }
        return item


# ============================================================
# 推理
# ============================================================
test_df = pd.read_csv(TEST_CSV)
print(f"Test shape: {test_df.shape}")

model = SpookyAuthorClassifier(num_authors=3, dropout_rate=0.3)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.to(device)
model.eval()
print("Model loaded.")

test_texts = test_df["text"].values
test_ids = test_df["id"].values

test_dataset = SpookyDataset(test_texts, tokenizer, max_length=MAX_LENGTH)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
)

all_test_probs = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with autocast():
            logits = model(input_ids, attention_mask, extra_features=batch["extra_features"].to(device))
            probs = torch.softmax(logits, dim=1)
        all_test_probs.append(probs.cpu().numpy())

test_probs = np.concatenate(all_test_probs, axis=0)

# ============================================================
# 生成提交文件
# ============================================================
submission = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_probs[:, 0],
        "HPL": test_probs[:, 1],
        "MWS": test_probs[:, 2],
    }
)

epsilon = 1e-15
for col in ["EAP", "HPL", "MWS"]:
    submission[col] = np.clip(submission[col], epsilon, 1 - epsilon)

row_sums = submission[["EAP", "HPL", "MWS"]].sum(axis=1)
submission["EAP"] = submission["EAP"] / row_sums
submission["HPL"] = submission["HPL"] / row_sums
submission["MWS"] = submission["MWS"] / row_sums

submission.to_csv(OUTPUT_CSV, index=False)
print(f"Submission saved: {submission.shape}")
