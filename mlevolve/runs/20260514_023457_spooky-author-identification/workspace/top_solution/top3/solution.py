import pandas as pd
import numpy as np
import re
import string
import pickle
import os
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import log_loss
from sentence_transformers import SentenceTransformer
import xgboost as xgb

print("Starting data processing and feature engineering...")

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_sub = pd.read_csv("./input/sample_submission.csv")

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# Encode target
le = LabelEncoder()
train_df["author_encoded"] = le.fit_transform(train_df["author"])
author_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(f"Author mapping: {author_mapping}")

# Combine all text for consistent feature engineering
all_text = pd.concat([train_df["text"], test_df["text"]], axis=0).reset_index(drop=True)


# --- Feature Engineering Functions ---
def basic_text_features(text_series):
    """Extract basic stylometric features"""
    features = pd.DataFrame(index=text_series.index)

    # Sentence length features
    features["char_count"] = text_series.str.len()
    features["word_count"] = text_series.str.split().str.len()
    features["avg_word_len"] = features["char_count"] / (features["word_count"] + 1)
    features["sentence_count"] = text_series.str.count(r"[.!?]+") + 1

    # Punctuation features
    exclamation_count = text_series.str.count("!")
    question_count = text_series.str.count(r"\?")
    period_count = text_series.str.count(r"\.")
    comma_count = text_series.str.count(",")
    colon_count = text_series.str.count(":")
    semicolon_count = text_series.str.count(";")
    dash_count = text_series.str.count("—") + text_series.str.count("-")
    quote_count = text_series.str.count('"') + text_series.str.count("'")

    features["exclamation_ratio"] = exclamation_count / (features["word_count"] + 1)
    features["question_ratio"] = question_count / (features["word_count"] + 1)
    features["period_ratio"] = period_count / (features["word_count"] + 1)
    features["comma_ratio"] = comma_count / (features["word_count"] + 1)
    features["colon_ratio"] = colon_count / (features["word_count"] + 1)
    features["semicolon_ratio"] = semicolon_count / (features["word_count"] + 1)
    features["dash_ratio"] = dash_count / (features["word_count"] + 1)
    features["quote_ratio"] = quote_count / (features["word_count"] + 1)

    # Capitalization features
    features["capital_ratio"] = text_series.str.findall(r"[A-Z]").str.len() / (
        features["char_count"] + 1
    )
    features["all_caps_words"] = text_series.str.findall(r"\b[A-Z]{2,}\b").str.len() / (
        features["word_count"] + 1
    )

    # Special character features
    features["special_char_ratio"] = text_series.str.findall(
        r"[^a-zA-Z0-9\s]"
    ).str.len() / (features["char_count"] + 1)
    features["digit_ratio"] = text_series.str.findall(r"\d").str.len() / (
        features["char_count"] + 1
    )

    # Readability proxies
    features["avg_syllables"] = text_series.str.lower().str.count(r"[aeiou]+") / (
        features["word_count"] + 1
    )
    features["complex_word_ratio"] = text_series.str.findall(
        r"\b\w{7,}\b"
    ).str.len() / (features["word_count"] + 1)
    features["short_word_ratio"] = text_series.str.findall(r"\b\w{1,3}\b").str.len() / (
        features["word_count"] + 1
    )

    # Lexical diversity
    def lexical_diversity(text):
        words = str(text).lower().split()
        if len(words) == 0:
            return 0.0
        return len(set(words)) / len(words)

    features["lexical_diversity"] = text_series.apply(lexical_diversity)

    return features


def pos_style_features(text_series):
    """Extract part-of-speech-like features using regex patterns"""
    features = pd.DataFrame(index=text_series.index)

    features["article_count"] = text_series.str.lower().str.findall(
        r"\b(a|an|the)\b"
    ).str.len() / (text_series.str.split().str.len() + 1)
    features["pronoun_count"] = text_series.str.lower().str.findall(
        r"\b(i|you|he|she|it|we|they|me|him|her|us|them|my|your|his|her|its|our|their|mine|yours|his|hers|ours|theirs)\b"
    ).str.len() / (text_series.str.split().str.len() + 1)
    features["preposition_count"] = text_series.str.lower().str.findall(
        r"\b(in|on|at|to|for|with|by|from|of|about|into|through|during|before|after|above|below|between|under|over)\b"
    ).str.len() / (text_series.str.split().str.len() + 1)
    features["conjunction_count"] = text_series.str.lower().str.findall(
        r"\b(and|but|or|nor|yet|so|for|because|although|while|since|unless|if|when|where)\b"
    ).str.len() / (text_series.str.split().str.len() + 1)
    features["adverb_count"] = text_series.str.lower().str.findall(
        r"\b(very|really|quite|almost|always|never|often|sometimes|rarely|just|only|also|too|however|therefore|thus|then|now|here|there)\b"
    ).str.len() / (text_series.str.split().str.len() + 1)
    features["adjective_count"] = text_series.str.lower().str.findall(
        r"\b(able|ible|al|ial|ful|less|ous|ious|ive|ative|ic|ical|ant|ent|ary|ory|ish|like|some|worthy|bound|en|ern|ese|esque|fold|ic|ical|ine|ish|ive|less|like|ly|most|ous|proof|some|ward|y|an|ian|ar|ary|ory|ate|ble|ent|ful|ic|ical|ine|ing|ive|less|like|ly|oid|ory|ose|ous|some|y)\w*\b"
    ).str.len() / (text_series.str.split().str.len() + 1)

    first_person = (
        text_series.str.lower()
        .str.findall(r"\b(i|me|my|mine|myself|we|us|our|ours|ourselves)\b")
        .str.len()
    )
    third_person = (
        text_series.str.lower()
        .str.findall(
            r"\b(he|she|it|they|him|her|them|his|her|its|their|theirs|himself|herself|itself|themselves)\b"
        )
        .str.len()
    )
    features["first_person_ratio"] = first_person / (first_person + third_person + 1)
    features["third_person_ratio"] = third_person / (first_person + third_person + 1)

    return features


def emotional_tone_features(text_series):
    """Simple sentiment and emotional indicators"""
    features = pd.DataFrame(index=text_series.index)

    fear_words = r"\b(fear|afraid|terror|horror|dread|fright|panic|alarm|scare|shock|nightmare|ghost|spook|creepy|eerie|ominous|menacing|sinister|gloomy|dark|shadow|phantom|specter|apparition|haunt|chill|cold|shiver|tremble|quake|shudder|cower|cringe|despair|hopeless|doom|ominous|portent|ill|boding|foreboding|weird|strange|mysterious|uncanny|supernatural|unnatural|eldritch|macabre|gruesome|grisly|gory|bloody|hideous|monstrous|horrid|awful|terrible|dreadful|hideous|ghastly|abominable|revolting|repulsive|disgusting|loathsome|odious|heinous)\b"
    joy_words = r"\b(happy|joy|delight|pleasure|bliss|ecstasy|rapture|elation|euphoria|content|glad|cheer|merry|jolly|festive|rejoice|celebrate|laughter|smile|grin|beam|radiant|glow|bright|sunny|warm|comfort|cool|refreshing|peace|calm|serene|tranquil|beautiful|wonderful|marvelous|splendid|magnificent|glorious|lovely|charming|enchanting|fascinating|enthralling|captivating|thrilling|exciting|exhilarating)\b"
    sadness_words = r"\b(sad|sorrow|grief|mourn|weep|cry|tear|sob|wail|lament|despair|hopeless|melancholy|gloom|misery|anguish|agony|pain|suffering|torment|hurt|wound|broken|heartache|lonely|alone|isolated|forsaken|abandoned|rejected|excluded|neglected|ignored|unloved|depressed|sullen|morose|dismal|dreary|bleak|somber|grave|solemn|pensive|wistful|longing|yearning|nostalgic|remorse|regret|guilt|shame)\b"
    anger_words = r"\b(angry|rage|fury|wrath|ire|indignation|outrage|resentment|bitter|hostile|furious|enraged|infuriated|exasperated|aggravated|annoyed|irritated|vexed|frustrated|fuming|seething|boiling|explosive|violent|fierce|savage|brutal|cruel|ruthless|merciless|vicious|malevolent|malicious|spiteful|vindictive|vengeful|wrathful|ireful|irate|incensed|livid|apoplectic)\b"

    features["fear_score"] = text_series.str.lower().str.findall(
        fear_words, flags=re.IGNORECASE
    ).str.len() / (text_series.str.split().str.len() + 1)
    features["joy_score"] = text_series.str.lower().str.findall(
        joy_words, flags=re.IGNORECASE
    ).str.len() / (text_series.str.split().str.len() + 1)
    features["sadness_score"] = text_series.str.lower().str.findall(
        sadness_words, flags=re.IGNORECASE
    ).str.len() / (text_series.str.split().str.len() + 1)
    features["anger_score"] = text_series.str.lower().str.findall(
        anger_words, flags=re.IGNORECASE
    ).str.len() / (text_series.str.split().str.len() + 1)

    return features


# Extract all handcrafted features
print("Extracting basic text features...")
basic_feats = basic_text_features(all_text)
print("Extracting POS-style features...")
pos_feats = pos_style_features(all_text)
print("Extracting emotional tone features...")
emo_feats = emotional_tone_features(all_text)

# Combine handcrafted features
handcrafted_features = pd.concat([basic_feats, pos_feats, emo_feats], axis=1)

# --- Sentence Transformer Embeddings ---
print("Generating sentence transformer embeddings...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
text_list = all_text.fillna("").tolist()
embeddings = model.encode(
    text_list,
    batch_size=64,
    show_progress_bar=False,
    device="cuda" if os.path.exists("/usr/local/cuda/bin/nvcc") else "cpu",
)
embedding_df = pd.DataFrame(
    embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])]
)

# Combine all features
all_features = pd.concat([handcrafted_features, embedding_df], axis=1)

# Split back into train and test
train_features = all_features.iloc[: len(train_df)].copy()
train_features["author"] = train_df["author"].values
train_features["author_encoded"] = train_df["author_encoded"].values
train_features["id"] = train_df["id"].values

test_features = all_features.iloc[len(train_df) :].copy()
test_features["id"] = test_df["id"].values

# --- Stratified Split ---
print("Creating stratified train/validation splits...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X = train_features.drop(["author", "author_encoded", "id"], axis=1).values
y = train_features["author_encoded"].values

# Scale features (fit on train, transform both)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(test_features.drop("id", axis=1).values)

# Save processed data
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Save as numpy arrays for subsequent steps
np.save("./working/X_train.npy", X_scaled)
np.save("./working/y_train.npy", y)
np.save("./working/X_test.npy", X_test_scaled)
np.save("./working/test_ids.npy", test_features["id"].values)

# Save feature names and metadata
metadata = {
    "feature_names": list(handcrafted_features.columns) + list(embedding_df.columns),
    "num_handcrafted": handcrafted_features.shape[1],
    "num_embedding": embedding_df.shape[1],
    "total_features": X_scaled.shape[1],
    "author_mapping": author_mapping,
    "classes": list(le.classes_),
}

with open("./working/metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

# Also save the raw text and indices for potential use
train_df.to_pickle("./working/train_raw.pkl")
test_df.to_pickle("./working/test_raw.pkl")

# Save the StratifiedKFold splits
fold_splits = []
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
    fold_splits.append({"train": train_idx, "val": val_idx})
    print(f"Fold {fold_idx}: Train {len(train_idx)}, Val {len(val_idx)}")

with open("./working/fold_splits.pkl", "wb") as f:
    pickle.dump(fold_splits, f)

print(f"Total features: {X_scaled.shape[1]}")
print(f"Handcrafted features: {handcrafted_features.shape[1]}")
print(f"Embedding features: {embedding_df.shape[1]}")
print(f"Train samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")

# Quick check of feature statistics
print(f"\nFeature range: [{X_scaled.min():.3f}, {X_scaled.max():.3f}]")
print(f"Feature mean: {X_scaled.mean():.6f}")
print(f"Feature std: {X_scaled.std():.6f}")

print("Data processing and feature engineering complete!")

# --- Model Design & Training/Evaluation ---
print("\nStarting training and evaluation...")

# Load preprocessed data
X_train = np.load("./working/X_train.npy")
y_train = np.load("./working/y_train.npy")
X_test = np.load("./working/X_test.npy")
test_ids = np.load("./working/test_ids.npy", allow_pickle=True)

with open("./working/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)
with open("./working/fold_splits.pkl", "rb") as f:
    fold_splits = pickle.load(f)

num_classes = len(metadata["classes"])
print(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
print(f"Test samples: {X_test.shape[0]}")

# 5-fold cross-validation
all_oof_preds = np.zeros((X_train.shape[0], num_classes))
all_test_preds = np.zeros((X_test.shape[0], num_classes))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\nFold {fold_idx + 1}/5")

    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        max_depth=6,
        learning_rate=0.1,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_lambda=1.0,
        reg_alpha=0.1,
        min_child_weight=5,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=50,
    )

    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    # Validation predictions
    val_preds = model.predict_proba(X_val)
    val_preds = np.clip(val_preds, 1e-15, 1 - 1e-15)
    val_preds = val_preds / val_preds.sum(axis=1, keepdims=True)
    all_oof_preds[val_idx] = val_preds

    # Test predictions
    test_preds = model.predict_proba(X_test)
    test_preds = np.clip(test_preds, 1e-15, 1 - 1e-15)
    test_preds = test_preds / test_preds.sum(axis=1, keepdims=True)
    all_test_preds += test_preds / 5

    # Fold validation score
    fold_score = log_loss(y_val, val_preds)
    print(f"Fold {fold_idx + 1} Log Loss: {fold_score:.6f}")

# Overall validation score
val_score = log_loss(y_train, all_oof_preds)
print(f"\nOverall Validation Log Loss: {val_score:.6f}")

# Generate submission
test_preds_final = np.clip(all_test_preds, 1e-15, 1 - 1e-15)
test_preds_final = test_preds_final / test_preds_final.sum(axis=1, keepdims=True)

submission = pd.DataFrame(
    {
        "id": test_ids,
        "EAP": test_preds_final[:, 0],
        "HPL": test_preds_final[:, 1],
        "MWS": test_preds_final[:, 2],
    }
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: ./submission/submission.csv")
print(f"Submission shape: {submission.shape}")

print(f"Final Validation Score: {val_score}")