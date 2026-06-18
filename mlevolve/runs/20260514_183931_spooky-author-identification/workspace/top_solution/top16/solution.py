import pandas as pd
import numpy as np
import os
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# DATA LOADING
# ============================================================
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Train/Validation split (StratifiedKFold)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(train_df, train_df["author"]))

train_set = train_df.iloc[train_idx].reset_index(drop=True)
val_set = train_df.iloc[val_idx].reset_index(drop=True)

le = LabelEncoder()
train_set["author"] = le.fit_transform(train_set["author"])
val_set["author"] = le.transform(val_set["author"])

train_indices = train_idx
val_indices = val_idx

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# MODEL DESIGN - Ensemble (RoBERTa-large, ELECTRA-large, DeBERTa-v3-large)
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")

MODEL_NAMES = [
    "roberta-large",
    "google/electra-large-discriminator",
    "microsoft/deberta-v3-large",
]

# We'll train each model independently; define training config
NUM_AUTHORS = 3
NUM_EPOCHS = 15
PATIENCE = 5
BATCH_SIZE = 8  # reduced batch size for larger models; adjust to memory
MAX_LENGTH = 512
CONSTANT_LR = 1e-5
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
RANDOM_SEEDS = [42, 2024, 123]

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler_grad = GradScaler()

print(f"Model parameters for ensemble: {len(MODEL_NAMES)} models")
for mname in MODEL_NAMES:
    print(f"  - {mname}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Max epochs: {NUM_EPOCHS}")
print(f"Constant LR: {CONSTANT_LR}")


# ============================================================
# EDA AUGMENTATION
# ============================================================
import random

# Simple synonym dictionary for EDA (small subset for demo; in practice expand)
synonym_dict = {
    "good": ["great", "fine", "excellent", "superb"],
    "bad": ["poor", "terrible", "awful", "horrible"],
    "big": ["large", "huge", "enormous", "massive"],
    "small": ["tiny", "little", "miniature", "compact"],
    "beautiful": ["pretty", "lovely", "gorgeous", "stunning"],
    "ugly": ["hideous", "unsightly", "grotesque", "monstrous"],
    "fast": ["quick", "rapid", "swift", "speedy"],
    "slow": ["sluggish", "leisurely", "unhurried", "gradual"],
    "happy": ["joyful", "cheerful", "delighted", "glad"],
    "sad": ["unhappy", "melancholy", "sorrowful", "gloomy"],
    "strange": ["odd", "peculiar", "unusual", "weird"],
    "dark": ["dim", "shadowy", "murky", "gloomy"],
    "light": ["bright", "luminous", "radiant", "brilliant"],
    "old": ["ancient", "elderly", "aged", "venerable"],
    "young": ["youthful", "juvenile", "adolescent", "immature"],
    "strong": ["powerful", "robust", "sturdy", "mighty"],
    "weak": ["feeble", "fragile", "delicate", "frail"],
    "cold": ["chilly", "cool", "frigid", "icy"],
    "hot": ["warm", "scorching", "burning", "fiery"],
    "deep": ["profound", "abyssal", "bottomless", "unfathomable"],
    "wide": ["broad", "expansive", "vast", "extensive"],
    "narrow": ["slender", "thin", "tight", "cramped"],
    "hard": ["difficult", "tough", "arduous", "strenuous"],
    "soft": ["gentle", "mild", "tender", "smooth"],
    "rich": ["wealthy", "affluent", "prosperous", "opulent"],
    "poor": ["impoverished", "destitute", "needy", "indigent"],
    "brave": ["courageous", "fearless", "valiant", "dauntless"],
    "afraid": ["scared", "frightened", "terrified", "fearful"],
    "calm": ["serene", "peaceful", "tranquil", "composed"],
    "angry": ["furious", "irate", "enraged", "wrathful"],
    "clear": ["obvious", "evident", "apparent", "transparent"],
    "confusing": ["bewildering", "puzzling", "perplexing", "baffling"],
    "empty": ["vacant", "hollow", "void", "barren"],
    "full": ["complete", "packed", "crowded", "brimming"],
    "silent": ["quiet", "hushed", "still", "noiseless"],
    "loud": ["noisy", "boisterous", "raucous", "deafening"],
    "safe": ["secure", "protected", "shielded", "guarded"],
    "dangerous": ["perilous", "hazardous", "risky", "treacherous"],
    "sweet": ["sugary", "saccharine", "honeyed", "candied"],
    "bitter": ["sour", "acidic", "acerbic", "caustic"],
    "bright": ["shining", "gleaming", "glowing", "luminous"],
    "dull": ["boring", "mundane", "tedious", "monotonous"],
    "fancy": ["elegant", "ornate", "elaborate", "decorative"],
    "plain": ["simple", "unadorned", "basic", "modest"],
    "friendly": ["amiable", "cordial", "genial", "affable"],
    "hostile": ["unfriendly", "antagonistic", "belligerent", "aggressive"],
    "gentle": ["kind", "tender", "soft", "mild"],
    "rough": ["harsh", "coarse", "rugged", "uneven"],
    "smart": ["intelligent", "clever", "brilliant", "sharp"],
    "foolish": ["stupid", "silly", "unwise", "absurd"],
    "rare": ["uncommon", "scarce", "unusual", "unique"],
    "common": ["ordinary", "typical", "usual", "standard"],
    "ancient": ["old", "primitive", "antique", "archaic"],
    "modern": ["contemporary", "current", "new", "up-to-date"],
    "noble": ["dignified", "honorable", "majestic", "regal"],
    "vile": ["despicable", "wicked", "evil", "depraved"],
    "sorrow": ["grief", "anguish", "woe", "misery"],
    "joy": ["delight", "happiness", "bliss", "ecstasy"],
    "fear": ["dread", "terror", "panic", "alarm"],
    "love": ["adoration", "affection", "devotion", "fondness"],
    "hate": ["detest", "loathe", "abhor", "despise"],
    "life": ["existence", "living", "vitality", "animation"],
    "death": ["demise", "end", "passing", "decease"],
    "hope": ["aspiration", "desire", "wish", "expectation"],
    "despair": ["hopelessness", "dejection", "despondency", "melancholy"],
    "light": ["illumination", "brightness", "radiance", "glow"],
    "darkness": ["gloom", "shadow", "obscurity", "dimness"],
    "beauty": ["loveliness", "splendor", "grace", "elegance"],
    "ugliness": ["hideousness", "grotesqueness", "unsightliness", "deformity"],
    "strength": ["power", "might", "force", "vigor"],
    "weakness": ["frailty", "feebleness", "infirmity", "delicacy"],
    "wisdom": ["knowledge", "sagacity", "insight", "intelligence"],
    "folly": ["foolishness", "stupidity", "absurdity", "nonsense"],
    "courage": ["bravery", "valor", "audacity", "nerve"],
    "cowardice": ["timidity", "fearfulness", "pusillanimity", "spinelessness"],
    "truth": ["verity", "accuracy", "correctness", "honesty"],
    "falsehood": ["lie", "deception", "pretense", "fabrication"],
    "friend": ["companion", "ally", "comrade", "confidant"],
    "enemy": ["foe", "adversary", "antagonist", "opponent"],
    "home": ["dwelling", "residence", "abode", "domicile"],
    "journey": ["voyage", "expedition", "trip", "odyssey"],
    "dream": ["vision", "fantasy", "reverie", "daydream"],
    "nightmare": ["horror", "terror", "torment", "ordeal"],
    "shadow": ["shade", "silhouette", "outline", "umbra"],
    "ghost": ["spirit", "phantom", "apparition", "specter"],
    "whisper": ["murmur", "mutter", "breathe", "susurrate"],
    "shout": ["yell", "cry", "bellow", "roar"],
    "walk": ["stroll", "wander", "stride", "saunter"],
    "run": ["sprint", "dash", "hurry", "speed"],
    "look": ["gaze", "peer", "glance", "observe"],
    "see": ["perceive", "discern", "behold", "witness"],
    "think": ["ponder", "contemplate", "deliberate", "reflect"],
    "feel": ["sense", "experience", "perceive", "detect"],
    "know": ["understand", "comprehend", "realize", "recognize"],
    "believe": ["trust", "accept", "maintain", "hold"],
    "doubt": ["question", "distrust", "skepticism", "uncertainty"],
    "begin": ["start", "commence", "initiate", "undertake"],
    "end": ["finish", "conclude", "terminate", "cease"],
    "create": ["make", "produce", "fashion", "build"],
    "destroy": ["ruin", "demolish", "annihilate", "devastate"],
    "find": ["discover", "locate", "unearth", "detect"],
    "lose": ["misplace", "forfeit", "drop", "squander"],
    "give": ["grant", "bestow", "confer", "donate"],
    "take": ["seize", "grasp", "capture", "claim"],
    "speak": ["talk", "utter", "voice", "articulate"],
    "listen": ["hear", "attend", "hark", "eavesdrop"],
    "sleep": ["slumber", "doze", "nap", "rest"],
    "wake": ["awaken", "rouse", "stir", "arise"],
    "laugh": ["chuckle", "giggle", "snicker", "cackle"],
    "cry": ["weep", "sob", "wail", "lament"],
    "sing": ["chant", "warble", "croon", "carol"],
    "dance": ["prance", "sway", "spin", "whirl"],
    "fly": ["soar", "glide", "hover", "wing"],
    "fall": ["drop", "descend", "tumble", "plunge"],
    "rise": ["ascend", "climb", "soar", "mount"],
    "hide": ["conceal", "secrete", "camouflage", "mask"],
    "reveal": ["disclose", "uncover", "expose", "divulge"],
    "help": ["aid", "assist", "support", "succor"],
    "harm": ["hurt", "injure", "damage", "wound"],
    "save": ["rescue", "salvage", "preserve", "protect"],
    "spend": ["expend", "consume", "exhaust", "drain"],
    "gather": ["collect", "assemble", "accumulate", "amass"],
    "scatter": ["disperse", "spread", "distribute", "dissipate"],
    "write": ["pen", "compose", "scribe", "draft"],
    "read": ["peruse", "scan", "browse", "scrutinize"],
    "teach": ["instruct", "educate", "train", "coach"],
    "learn": ["study", "acquire", "master", "absorb"],
}


def synonym_replacement(text, prob=0.1):
    words = text.split()
    new_words = words.copy()
    for i, word in enumerate(words):
        if word.lower() in synonym_dict and random.random() < prob:
            new_words[i] = random.choice(synonym_dict[word.lower()])
    return " ".join(new_words)


def random_insertion(text, prob=0.05):
    words = text.split()
    if len(words) == 0:
        return text
    num_insert = max(1, int(len(words) * prob))
    new_words = words.copy()
    for _ in range(num_insert):
        ins_word = random.choice(words)
        pos = random.randint(0, len(new_words))
        if random.random() < 0.5:
            ins_word = random.choice(list(synonym_dict.keys()))
        new_words.insert(pos, ins_word)
    return " ".join(new_words)


def eda_augment(text, synonym_prob=0.1, insertion_prob=0.05):
    text = synonym_replacement(text, prob=synonym_prob)
    text = random_insertion(text, prob=insertion_prob)
    return text


# ============================================================
# DATASET AND DATALOADER
# ============================================================
class SpookyDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512, augment=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        if self.augment:
            text = eda_augment(text, synonym_prob=0.1, insertion_prob=0.05)
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
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# Get original texts for training
author_map = {"EAP": 0, "HPL": 1, "MWS": 2}
train_texts_orig = train_df["text"].values
train_labels_orig = np.array([author_map[a] for a in train_df["author"].values])
test_texts = test_df["text"].values
test_ids = test_df["id"].values

# Use original split indices (before reset_index)
train_texts_final = train_texts_orig[train_indices]
train_labels_final = train_labels_orig[train_indices]
val_texts_final = train_texts_orig[val_indices]
val_labels_final = train_labels_orig[val_indices]

# ============================================================
# TRAINING AND ENSEMBLE INFERENCE
# ============================================================
all_test_probs_list = []
all_val_probs_list = []
best_val_scores = []

for model_idx, model_name in enumerate(MODEL_NAMES):
    seed = RANDOM_SEEDS[model_idx]
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    print(f"\n{'='*60}")
    print(f"Training model {model_idx+1}/{len(MODEL_NAMES)}: {model_name}")
    print(f"Random seed: {seed}")
    print(f"{'='*60}")

    # Load tokenizer for this model
    if "deberta" in model_name:
        model_tokenizer = AutoTokenizer.from_pretrained(model_name)
        max_length = 512  # DeBERTa supports up to 512
    elif "electra" in model_name:
        model_tokenizer = AutoTokenizer.from_pretrained(model_name)
        max_length = 512
    else:  # roberta
        model_tokenizer = AutoTokenizer.from_pretrained(model_name)
        max_length = 512

    # Create datasets and dataloaders
    train_dataset = SpookyDataset(
        train_texts_final, train_labels_final, model_tokenizer, max_length, augment=True
    )
    val_dataset = SpookyDataset(
        val_texts_final, val_labels_final, model_tokenizer, max_length, augment=False
    )
    test_dataset = SpookyDataset(
        test_texts, None, model_tokenizer, max_length, augment=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_AUTHORS,
        hidden_dropout_prob=0.3,
        attention_probs_dropout_prob=0.3,
    )
    model.to(device)

    # Optimizer - constant LR, no scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=CONSTANT_LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )

    # Training loop
    best_val_score = float("inf")
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        num_train_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler_grad.scale(loss).backward()
            scaler_grad.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            scaler_grad.step(optimizer)
            scaler_grad.update()

            total_train_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = total_train_loss / num_train_batches

        # Validation
        model.eval()
        total_val_loss = 0
        num_val_batches = 0
        all_val_probs = []
        all_val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=1)

                total_val_loss += loss.item()
                num_val_batches += 1
                all_val_probs.append(probs.cpu().numpy())
                all_val_labels.append(labels.cpu().numpy())

        avg_val_loss = total_val_loss / num_val_batches
        val_probs = np.concatenate(all_val_probs, axis=0)
        val_true = np.concatenate(all_val_labels, axis=0)
        val_probs_clipped = np.clip(val_probs, 1e-15, 1 - 1e-15)
        val_probs_clipped = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)
        val_score = log_loss(val_true, val_probs_clipped)

        print(
            f"  Epoch {epoch+1:2d}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LogLoss: {val_score:.4f}"
        )

        if val_score < best_val_score:
            best_val_score = val_score
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
            # Save best model per model
            torch.save(model.state_dict(), f"./working/best_model_{model_idx}.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"  Early stopping triggered after {epoch+1} epochs")
                break

    print(f"  Best validation log-loss for {model_name}: {best_val_score:.4f}")
    best_val_scores.append(best_val_score)

    # Load best model for this member
    model.load_state_dict(best_model_state)
    model.eval()

    # Validation predictions from best model
    all_val_probs = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
            all_val_probs.append(probs.cpu().numpy())
    val_probs_model = np.concatenate(all_val_probs, axis=0)
    all_val_probs_list.append(val_probs_model)

    # Test predictions from best model
    all_test_probs = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
            all_test_probs.append(probs.cpu().numpy())
    test_probs_model = np.concatenate(all_test_probs, axis=0)
    all_test_probs_list.append(test_probs_model)

    # Clean memory
    del model
    torch.cuda.empty_cache()

# ============================================================
# ENSEMBLE AVERAGING
# ============================================================
# Arithmetic mean of val predictions
val_ensemble_probs = np.mean(all_val_probs_list, axis=0)
val_ensemble_clipped = np.clip(val_ensemble_probs, 1e-15, 1 - 1e-15)
val_ensemble_clipped = val_ensemble_clipped / val_ensemble_clipped.sum(axis=1, keepdims=True)
ensemble_val_score = log_loss(val_true, val_ensemble_clipped)

# Arithmetic mean of test predictions
test_ensemble_probs = np.mean(all_test_probs_list, axis=0)

print(f"\n{'='*60}")
print("ENSEMBLE RESULTS")
print(f"{'='*60}")
for idx, mname in enumerate(MODEL_NAMES):
    print(f"  Model {idx+1} ({mname}): Best Val LogLoss = {best_val_scores[idx]:.4f}")
print(f"  Final Ensemble Val LogLoss: {ensemble_val_score:.4f}")

# ============================================================
# GENERATE SUBMISSION
# ============================================================
submission = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_ensemble_probs[:, 0],
        "HPL": test_ensemble_probs[:, 1],
        "MWS": test_ensemble_probs[:, 2],
    }
)

epsilon = 1e-15
for col in ["EAP", "HPL", "MWS"]:
    submission[col] = np.clip(submission[col], epsilon, 1 - epsilon)

row_sums = submission[["EAP", "HPL", "MWS"]].sum(axis=1)
submission["EAP"] = submission["EAP"] / row_sums
submission["HPL"] = submission["HPL"] / row_sums
submission["MWS"] = submission["MWS"] / row_sums

submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")

print(f"Final Ensemble Validation Score: {ensemble_val_score}")