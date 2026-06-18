"""
Run 20260514_171209 Top1 推理脚本
LogLoss: 0.00864 (INDEX_BUG - 验证集泄露)
模型: DeBERTa-v3-large + StyleAwareClassifier (StylometricFeatureEncoder + SupConHead)
注意: 原solution测试推理时未传入stylometric_features, 模型fallback到zero embedding
Checkpoint: best_model_881ed9428f2844cdb751e7c13fe4945c.pt

用法: python infer_171209.py
"""

import pandas as pd
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 路径配置
# ============================================================
INFERENCE_DIR = "/workspace/nautilus/mlevolve/inference"
TEST_CSV = f"{INFERENCE_DIR}/test.csv"
CHECKPOINT = f"{INFERENCE_DIR}/checkpoints/best_model_171209.pt"
OUTPUT_CSV = f"{INFERENCE_DIR}/submissions/run_171209_top1_full_8392.csv"
MAX_LENGTH = 256
BATCH_SIZE = 16

# ============================================================
# 模型定义 (与solution.py完全一致)
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")


class StylometricFeatureEncoder(nn.Module):
    """Encodes 141 handcrafted stylometric features into a dense embedding."""
    def __init__(self, input_dim=141, hidden_dim=256, output_dim=128, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class SupConProjectionHead(nn.Module):
    """Projects concatenated representation for supervised contrastive loss."""
    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.proj(x)


class StyleAwareClassifier(nn.Module):
    def __init__(self, num_authors=3, dropout_rate=0.3, stylometric_dim=141):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            "microsoft/deberta-v3-large",
            output_hidden_states=False,
            output_attentions=False,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
        )
        # Unfreeze all backbone parameters for full fine-tuning
        for param in self.backbone.parameters():
            param.requires_grad = True

        hidden_size = self.backbone.config.hidden_size  # 1024 for deberta-v3-large

        # Stylometric feature encoder
        self.stylo_encoder = StylometricFeatureEncoder(
            input_dim=stylometric_dim,
            hidden_dim=256,
            output_dim=128,
            dropout_rate=dropout_rate
        )

        # Combined representation dimension
        combined_dim = hidden_size + 128

        # Projection head for supervised contrastive learning
        self.supcon_head = SupConProjectionHead(
            input_dim=combined_dim,
            hidden_dim=256,
            output_dim=128
        )

        # Classifier head taking combined CLS + stylometric embedding
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_authors),
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        stylometric_features=None,
        return_embeddings=False,
    ):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token (first token) for classification
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch, hidden_size]

        # Process stylometric features if provided
        if stylometric_features is not None:
            stylo_emb = self.stylo_encoder(stylometric_features)  # [batch, 128]
            combined = torch.cat([cls_output, stylo_emb], dim=1)  # [batch, combined_dim]
        else:
            # Fallback: zero padding for stylometric features
            batch_size = cls_output.shape[0]
            device = cls_output.device
            stylo_emb = torch.zeros(batch_size, 128, device=device)
            combined = torch.cat([cls_output, stylo_emb], dim=1)

        logits = self.classifier(combined)

        if return_embeddings:
            proj_emb = self.supcon_head(combined)
            return logits, proj_emb
        return logits


class SpookyDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=256):
        self.texts = texts
        self.labels = labels
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

model = StyleAwareClassifier(num_authors=3, dropout_rate=0.3).to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()
print("Model loaded.")

test_texts = test_df["text"].values
test_ids = test_df["id"].values

test_dataset = SpookyDataset(test_texts, None, tokenizer, max_length=MAX_LENGTH)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
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
