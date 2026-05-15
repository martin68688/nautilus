"""
Run 20260514_190327 Top3 推理脚本
LogLoss: 0.05066 (真实 log_loss, 有 INDEX_BUG 但不影响推理)
模型: DeBERTa-v3-large + AttentionPooling (last 4 layers, attention-weighted)
Checkpoint: best_model_023c6a4888a842da9af27b9e94e947cf.pt

用法: python infer_top3_05066.py
"""

import pandas as pd
import numpy as np
import torch
from torch import nn
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
CHECKPOINT = "/workspace/nautilus/mlevolve/runs/20260514_190327_spooky-author-identification/workspace/working/best_model_023c6a4888a842da9af27b9e94e947cf.pt"
OUTPUT_CSV = f"{INFERENCE_DIR}/submissions/run_190327_top3_logloss_0.05066_full_8392.csv"
MAX_LENGTH = 512
BATCH_SIZE = 16

# ============================================================
# 模型定义 (与solution.py完全一致)
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size, num_layers=4):
        super().__init__()
        self.num_layers = num_layers
        self.attention_scorer = nn.Linear(hidden_size, 1)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.attention_scorer.weight, std=0.02)
        nn.init.zeros_(self.attention_scorer.bias)

    def forward(self, hidden_states, attention_mask):
        batch_size, seq_len = attention_mask.shape
        hidden = hidden_states[-self.num_layers:]
        stacked = torch.stack(hidden, dim=0)
        avg_hidden = stacked.mean(dim=0)
        extended_mask = attention_mask.unsqueeze(-1).float()
        avg_hidden_masked = avg_hidden * extended_mask
        scores = self.attention_scorer(avg_hidden_masked).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        attention_weights = torch.softmax(scores, dim=-1)
        weighted = attention_weights.unsqueeze(-1) * avg_hidden
        pooled = weighted.sum(dim=1)
        return pooled


class SpookyAuthorClassifier(nn.Module):
    def __init__(self, num_labels=3, hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.2):
        super().__init__()
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-large",
            num_labels=num_labels,
            output_hidden_states=True,
            output_attentions=False,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
        )
        hidden_size = self.backbone.config.hidden_size
        self.attention_pool = AttentionPooling(hidden_size, num_layers=4)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self._init_classifier()

    def _init_classifier(self):
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states
        pooled = self.attention_pool(hidden_states, attention_mask)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return {'logits': logits}


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

model = SpookyAuthorClassifier(
    num_labels=3,
    hidden_dropout_prob=0.2,
    attention_probs_dropout_prob=0.2,
)
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
            outputs = model(input_ids, attention_mask)
            logits = outputs['logits']
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
