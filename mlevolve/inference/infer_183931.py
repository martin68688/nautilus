"""
Run 20260514_183931 Top1 推理脚本
LogLoss: 0.04434 (INDEX_BUG - 验证集泄露)
模型: DeBERTa-v3-large + AttentionMLP + Mean + Residual
Checkpoint: best_model_774e75e999ed43b7bb50f493d1f0c326.pt

用法: python infer_183931.py
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
CHECKPOINT = f"{INFERENCE_DIR}/checkpoints/best_model_183931.pt"
OUTPUT_CSV = f"{INFERENCE_DIR}/submissions/run_183931_top1_full_8392.csv"
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
        self.attention_mlp = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )
        self.attention_mlp_residual = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, hidden_size),
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, num_authors)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]
        attn_logits = self.attention_mlp(hidden_states).squeeze(-1)  # [batch, seq_len]
        attn_logits = attn_logits.masked_fill(attention_mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_logits, dim=1)  # [batch, seq_len]
        attended = (hidden_states * attn_weights.unsqueeze(-1)).sum(dim=1)  # [batch, hidden]
        # Mean pooling of hidden_states
        mask = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
        mean_pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # [batch, hidden]
        # Residual: attended + mean_pooled, then pass through MLP
        combined = attended + mean_pooled
        residual_out = self.attention_mlp_residual(combined)
        pooled = combined + residual_out
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
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
            logits = model(input_ids, attention_mask)
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
