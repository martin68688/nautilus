import os
os.sched_setaffinity(0, {5, 6, 7, 8, 9, 10, 11, 20, 21, 22})
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import log_loss
import xgboost as xgb
import lightgbm as lgb
import re
import string
from collections import Counter
import os
import warnings
import joblib

warnings.filterwarnings("ignore")

# Create directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============== 1. LOAD DATA ==============
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Classes: {train_df['author'].value_counts().to_dict()}")


# ============== 2. TEXT CLEANING ==============
def clean_text(text):
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


train_df["clean_text"] = train_df["text"].apply(clean_text)
test_df["clean_text"] = test_df["text"].apply(clean_text)

# ============== 3. FEATURE ENGINEERING ==============


# --- 3a. Basic Statistics Features ---
def count_syllables(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word and word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
        count += 1
    if count == 0:
        count = 1
    return count


def extract_basic_features(text):
    features = {}
    words = text.split()
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    features["char_count"] = len(text)
    features["word_count"] = len(words)
    features["sentence_count"] = len(sentences) if sentences else 1
    features["avg_word_length"] = np.mean([len(w) for w in words]) if words else 0
    features["avg_sentence_length"] = features["word_count"] / max(
        1, features["sentence_count"]
    )

    features["exclamation_count"] = text.count("!")
    features["question_count"] = text.count("?")
    features["comma_count"] = text.count(",")
    features["semicolon_count"] = text.count(";")
    features["colon_count"] = text.count(":")
    features["dash_count"] = text.count("-") + text.count("—") + text.count("–")
    features["quote_count"] = text.count('"') + text.count("'")
    features["period_count"] = text.count(".")
    features["punctuation_ratio"] = sum(
        1 for c in text if c in string.punctuation
    ) / max(1, len(text))

    unique_words = set(w.lower() for w in words)
    features["unique_word_ratio"] = len(unique_words) / max(1, len(words))

    caps_words = sum(1 for w in words if len(w) > 0 and w[0].isupper())
    features["caps_ratio"] = caps_words / max(1, len(words))
    features["all_caps_words"] = sum(1 for w in words if w.isupper() and len(w) > 1)

    features["ellipsis_count"] = text.count("...")

    features["syllable_count"] = sum(count_syllables(w) for w in words)
    features["flesch_score"] = (
        206.835
        - 1.015 * (features["word_count"] / max(1, features["sentence_count"]))
        - 84.6 * (features["syllable_count"] / max(1, features["word_count"]))
    )

    return features


print("Extracting basic features...")
basic_train = train_df["clean_text"].apply(extract_basic_features).apply(pd.Series)
basic_test = test_df["clean_text"].apply(extract_basic_features).apply(pd.Series)

# --- 3b. N-gram Features ---
print("Extracting n-gram features...")
char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=500,
    sublinear_tf=True,
    min_df=3,
    max_df=0.9,
)
char_ngrams_train = char_vectorizer.fit_transform(train_df["clean_text"])
char_ngrams_test = char_vectorizer.transform(test_df["clean_text"])

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=1000,
    sublinear_tf=True,
    min_df=3,
    max_df=0.8,
    stop_words="english",
)
word_ngrams_train = word_vectorizer.fit_transform(train_df["clean_text"])
word_ngrams_test = word_vectorizer.transform(test_df["clean_text"])

# --- 3c. Function Words ---
function_words = [
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "because",
    "as",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "each",
    "every",
    "both",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "just",
    "can",
    "will",
    "should",
    "now",
]


def extract_function_word_features(text, fw_list):
    words = text.lower().split()
    word_counts = Counter(words)
    total_words = len(words)
    features = {}
    for fw in fw_list:
        features[f"fw_{fw}"] = word_counts.get(fw, 0) / max(1, total_words)
    return features


fw_train = (
    train_df["clean_text"]
    .apply(lambda x: extract_function_word_features(x, function_words))
    .apply(pd.Series)
)
fw_test = (
    test_df["clean_text"]
    .apply(lambda x: extract_function_word_features(x, function_words))
    .apply(pd.Series)
)

# --- 3d. Syntactic Features ---
pos_patterns = {
    "determiners": r"\b(the|a|an|this|that|these|those|my|your|his|her|its|our|their|some|any|no|every|each|all|both|few|many|much)\b",
    "prepositions": r"\b(in|on|at|to|for|of|with|by|from|up|down|into|through|during|without|between|among|before|after|above|below|under|over|against|around|about)\b",
    "conjunctions": r"\b(and|but|or|nor|yet|so|for|because|although|while|if|when|where|why|how|that|which|who|whom|whose|what|whether)\b",
    "pronouns": r"\b(I|you|he|she|it|we|they|me|him|her|us|them|my|your|his|its|our|their|mine|yours|his|hers|its|ours|theirs|myself|yourself|himself|herself|itself|ourselves|yourselves|themselves)\b",
    "adverbs_ly": r"\b\w+ly\b",
    "past_tense_ed": r"\b\w+ed\b",
    "ing_forms": r"\b\w+ing\b",
    "superlatives": r"\b\w+est\b",
    "comparatives": r"\b(more|less|better|worse|greater|lesser|further)\b",
    "negations": r"\b(not|no|never|nothing|nor|neither|none|nobody|nowhere|can't|don't|won't|wouldn't|shouldn't|couldn't|isn't|aren't|wasn't|weren't|haven't|hasn't|hadn't)\b",
    "archaic_words": r"\b(thou|thee|thy|thine|ye|hath|doth|dost|art|wast|wert|shalt|wilt|canst|wouldst|shouldst|couldst|didst|hadst|mays|mayst|hither|thither|whence|thence|whither|wherefore|henceforth|therein|thereupon|herewith|perchance|methinks|prithee|forsooth)\b",
}


def extract_pos_features(text, patterns):
    features = {}
    text_lower = text.lower()
    for name, pattern in patterns.items():
        count = len(re.findall(pattern, text_lower))
        features[f"pos_{name}"] = count / max(1, len(text.split()))
    return features


pos_train = (
    train_df["clean_text"]
    .apply(lambda x: extract_pos_features(x, pos_patterns))
    .apply(pd.Series)
)
pos_test = (
    test_df["clean_text"]
    .apply(lambda x: extract_pos_features(x, pos_patterns))
    .apply(pd.Series)
)

# --- 3e. Author-specific Markers ---
author_markers = {
    "lovecraftian": [
        "cyclopean",
        "non-euclidean",
        "eldritch",
        "squamous",
        "gibbous",
        "nameless",
        "unspeakable",
        "indescribable",
        "cryptic",
        "antediluvian",
        "primordial",
        "archaic",
        "cthulhu",
        "yog-sothoth",
        "necronomicon",
        "blasphemous",
        "charnel",
        "daemonic",
        "formless",
        "gargantuan",
        "loathsome",
        "monstrous",
        "obscene",
        "peculiar",
        "prodigious",
        "repulsive",
        "singular",
        "terrifying",
        "unfathomable",
        "unnatural",
        "vile",
        "weird",
        "profound",
        "remote",
        "ancient",
        "forbidden",
        "hidden",
        "secret",
        "strange",
        "unusual",
    ],
    "poe_style": [
        "nevermore",
        "chamber",
        "raven",
        "pendulum",
        "amontillado",
        "tell-tale",
        "premature",
        "grotesque",
        "arabesque",
        "dungeon",
        "mansion",
        "sepulchre",
        "tomb",
        "coffin",
        "shroud",
        "pallid",
        "maniacal",
        "lament",
        "melancholy",
        "desolate",
        "dreary",
        "weary",
        "bleak",
        "sombre",
        "gloomy",
        "shadowy",
        "phantasm",
        "spectral",
        "ghastly",
        "horrible",
        "terrible",
        "dreadful",
        "appalling",
        "fearful",
        "hideous",
        "repulsive",
    ],
    "shelley_style": [
        "creature",
        "monster",
        "frankenstein",
        "geneva",
        "wretch",
        "demon",
        "fiend",
        "victor",
        "clerval",
        "elizabeth",
        "justine",
        "walton",
        "ardent",
        "catastrophe",
        "countenance",
        "despond",
        "desolate",
        "enterprise",
        "exalted",
        "excursion",
        "existence",
        "explore",
        "extraordinary",
        "fervent",
        "gloomy",
        "gratitude",
        "imagination",
        "immortal",
        "impatient",
        "impulse",
        "incredulous",
        "induce",
        "inexorable",
        "infinite",
        "influence",
        "inquiry",
        "insurmountable",
        "intellectual",
        "intense",
        "irrevocable",
        "labour",
        "magnificent",
        "melancholy",
        "misery",
        "misfortune",
        "modify",
        "mortal",
        "mysterious",
        "natural",
        "necessity",
        "neglect",
        "obscure",
        "obstacle",
        "oppressed",
        "overwhelming",
    ],
}


def extract_author_markers(text, markers_dict):
    features = {}
    text_lower = text.lower()
    for author, markers in markers_dict.items():
        count = sum(1 for m in markers if m in text_lower)
        features[f"{author}_density"] = count / max(1, len(text.split()))
        features[f"{author}_raw"] = count
    return features


markers_train = (
    train_df["clean_text"]
    .apply(lambda x: extract_author_markers(x, author_markers))
    .apply(pd.Series)
)
markers_test = (
    test_df["clean_text"]
    .apply(lambda x: extract_author_markers(x, author_markers))
    .apply(pd.Series)
)


# --- 3f. Structure Features ---
def extract_structure_features(text):
    features = {}
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) > 0:
        sent_lengths = [len(s.split()) for s in sentences]
        features["mean_sent_length"] = np.mean(sent_lengths)
        features["std_sent_length"] = np.std(sent_lengths)
        features["max_sent_length"] = max(sent_lengths)
        features["min_sent_length"] = min(sent_lengths)
        features["clause_density"] = sum(s.count(",") for s in sentences) / max(
            1, len(sentences)
        )
        features["has_that_clause"] = int("that" in text.lower())
        features["has_which_clause"] = int("which" in text.lower())
        features["has_whom"] = int("whom" in text.lower())

    words = text.lower().split()
    if len(words) > 0:
        features["quoted_speech"] = len(re.findall(r'"([^"]*)"', text)) + len(
            re.findall(r"'([^']*)'", text)
        )

    return features


structure_train = (
    train_df["clean_text"].apply(extract_structure_features).apply(pd.Series)
)
structure_test = (
    test_df["clean_text"].apply(extract_structure_features).apply(pd.Series)
)


# --- 3g. Readability Metrics ---
def extract_readability_features(text):
    features = {}
    words = text.split()
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(words) > 0 and len(sentences) > 0:
        char_count = len(re.findall(r"\w", text))
        avg_chars_per_word = char_count / len(words)
        avg_words_per_sentence = len(words) / len(sentences)
        features["ari_score"] = (
            (4.71 * avg_chars_per_word) + (0.5 * avg_words_per_sentence) - 21.43
        )

        L = (char_count / len(words)) * 100
        S = (len(sentences) / len(words)) * 100
        features["coleman_liau"] = (0.0588 * L) - (0.296 * S) - 15.8

        polysyllabic = sum(1 for w in words if count_syllables(w) >= 3)
        features["smog_index"] = (
            1.0430 * np.sqrt(polysyllabic * (30 / max(1, len(sentences)))) + 3.1291
        )

        complex_words = sum(1 for w in words if count_syllables(w) >= 3)
        features["gunning_fog"] = 0.4 * (
            avg_words_per_sentence + 100 * (complex_words / max(1, len(words)))
        )

    return features


readability_train = (
    train_df["clean_text"].apply(extract_readability_features).apply(pd.Series)
)
readability_test = (
    test_df["clean_text"].apply(extract_readability_features).apply(pd.Series)
)

# ============== 4. COMBINE ALL FEATURES ==============
print("Combining all features...")

non_sparse_train = pd.concat(
    [
        basic_train,
        fw_train,
        pos_train,
        markers_train,
        structure_train,
        readability_train,
    ],
    axis=1,
)
non_sparse_test = pd.concat(
    [basic_test, fw_test, pos_test, markers_test, structure_test, readability_test],
    axis=1,
)

non_sparse_train = non_sparse_train.fillna(0)
non_sparse_test = non_sparse_test.fillna(0)

# Scale features
scaler = StandardScaler()
non_sparse_train_scaled = pd.DataFrame(
    scaler.fit_transform(non_sparse_train),
    columns=non_sparse_train.columns,
    index=non_sparse_train.index,
)
non_sparse_test_scaled = pd.DataFrame(
    scaler.transform(non_sparse_test),
    columns=non_sparse_test.columns,
    index=non_sparse_test.index,
)

# Convert sparse matrices
char_train_dense = char_ngrams_train.toarray()
char_test_dense = char_ngrams_test.toarray()
word_train_dense = word_ngrams_train.toarray()
word_test_dense = word_ngrams_test.toarray()

char_feature_names = [f"char_ngram_{i}" for i in range(char_train_dense.shape[1])]
word_feature_names = [f"word_ngram_{i}" for i in range(word_train_dense.shape[1])]

char_train_df = pd.DataFrame(
    char_train_dense, columns=char_feature_names, index=train_df.index
)
char_test_df = pd.DataFrame(
    char_test_dense, columns=char_feature_names, index=test_df.index
)
word_train_df = pd.DataFrame(
    word_train_dense, columns=word_feature_names, index=train_df.index
)
word_test_df = pd.DataFrame(
    word_test_dense, columns=word_feature_names, index=test_df.index
)

X_train = pd.concat([non_sparse_train_scaled, char_train_df, word_train_df], axis=1)
X_test = pd.concat([non_sparse_test_scaled, char_test_df, word_test_df], axis=1)

print(f"Total features: {X_train.shape[1]}")

# ============== 5. FEATURE SELECTION ==============
print("Selecting top features...")
y_train = train_df["author"]
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_train)

selector = SelectKBest(score_func=mutual_info_classif, k=min(2000, X_train.shape[1]))
X_train_selected = selector.fit_transform(X_train, y_encoded)
X_test_selected = selector.transform(X_test)

selected_indices = selector.get_support(indices=True)
selected_features = X_train.columns[selected_indices].tolist()

X_train_final = pd.DataFrame(
    X_train_selected, columns=selected_features, index=train_df.index
)
X_test_final = pd.DataFrame(
    X_test_selected, columns=selected_features, index=test_df.index
)

print(f"Selected features: {X_train_final.shape[1]}")

# ============== 6. STRATIFIED K-FOLD CROSS VALIDATION ==============
print("\nStarting stratified 5-fold cross-validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_val_scores = []
fold_models = []
fold_test_probs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_final, y_encoded)):
    print(f"\n--- Fold {fold + 1}/5 ---")

    X_tr = X_train_final.iloc[train_idx]
    y_tr = y_encoded[train_idx]
    X_val = X_train_final.iloc[val_idx]
    y_val = y_encoded[val_idx]

    # XGBoost model
    xgb_model = xgb.XGBClassifier(
        n_estimators=2000,
        max_depth=7,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=2.0,
        reg_alpha=0.5,
        gamma=0.1,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])

    # LightGBM model
    lgb_model = lgb.LGBMClassifier(
        n_estimators=2000,
        max_depth=7,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_lambda=2.0,
        reg_alpha=0.5,
        min_split_gain=0.1,
        objective="multiclass",
        num_class=3,
        metric="multi_logloss",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        force_row_wise=True,
    )
    lgb_model.fit(
        X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50)]
    )

    # Get validation probabilities
    xgb_val_probs = xgb_model.predict_proba(X_val)
    lgb_val_probs = lgb_model.predict_proba(X_val)

    # Simple ensemble (equal weights for validation)
    val_probs = (xgb_val_probs + lgb_val_probs) / 2
    val_probs = np.clip(val_probs, 1e-15, 1 - 1e-15)
    val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)

    fold_val_score = log_loss(y_val, val_probs)
    fold_val_scores.append(fold_val_score)
    print(f"Fold {fold + 1} Validation Log Loss: {fold_val_score:.6f}")

    # Store models for ensemble
    fold_models.append((xgb_model, lgb_model))

    # Test predictions
    xgb_test_probs = xgb_model.predict_proba(X_test_final)
    lgb_test_probs = lgb_model.predict_proba(X_test_final)
    test_probs = (xgb_test_probs + lgb_test_probs) / 2
    fold_test_probs.append(test_probs)

print(f"\nCross-validation scores: {[f'{s:.6f}' for s in fold_val_scores]}")
mean_val_score = np.mean(fold_val_scores)
std_val_score = np.std(fold_val_scores)
print(f"Mean Validation Log Loss: {mean_val_score:.6f} (+/- {std_val_score:.6f})")

# ============== 7. OPTIMIZE ENSEMBLE WEIGHTS ==============
print("\nOptimizing ensemble weights...")

all_val_probs_xgb = []
all_val_probs_lgb = []
all_val_labels = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_final, y_encoded)):
    X_tr = X_train_final.iloc[train_idx]
    y_tr = y_encoded[train_idx]
    X_val = X_train_final.iloc[val_idx]
    y_val = y_encoded[val_idx]

    xgb_model, lgb_model = fold_models[fold]
    all_val_probs_xgb.append(xgb_model.predict_proba(X_val))
    all_val_probs_lgb.append(lgb_model.predict_proba(X_val))
    all_val_labels.append(y_val)

all_val_probs_xgb = np.concatenate(all_val_probs_xgb, axis=0)
all_val_probs_lgb = np.concatenate(all_val_probs_lgb, axis=0)
all_val_labels = np.concatenate(all_val_labels, axis=0)

best_score = float("inf")
best_weights = (0.5, 0.5)

for w1 in np.arange(0.0, 1.05, 0.05):
    for w2 in np.arange(0.0, 1.05 - w1, 0.05):
        w3 = 1.0 - w1 - w2
        if w3 < 0.0:
            continue

        ensemble_probs = w1 * all_val_probs_xgb + w2 * all_val_probs_lgb
        ensemble_probs = np.clip(ensemble_probs, 1e-15, 1 - 1e-15)
        ensemble_probs = ensemble_probs / ensemble_probs.sum(axis=1, keepdims=True)

        score = log_loss(all_val_labels, ensemble_probs)

        if score < best_score:
            best_score = score
            best_weights = (w1, w2)

print(
    f"Best ensemble weights: XGBoost={best_weights[0]:.3f}, LightGBM={best_weights[1]:.3f}"
)
print(f"Best ensemble validation score: {best_score:.6f}")

# ============== 8. FINAL TEST PREDICTIONS ==============
print("\nGenerating final test predictions...")

final_test_probs = best_weights[0] * fold_test_probs[0] + best_weights[1] * fold_test_probs[1]
for fold_idx in range(1, len(fold_test_probs)):
    final_test_probs += (
        best_weights[0] * fold_test_probs[fold_idx]
        + best_weights[1] * fold_test_probs[fold_idx]
    )
final_test_probs /= len(fold_test_probs)

final_test_probs = np.clip(final_test_probs, 1e-15, 1 - 1e-15)
final_test_probs = final_test_probs / final_test_probs.sum(axis=1, keepdims=True)

# ============== 9. CREATE SUBMISSION ==============
print("Creating submission file...")
submission = pd.DataFrame(
    {
        "id": test_df["id"],
        "EAP": final_test_probs[:, 0],
        "HPL": final_test_probs[:, 1],
        "MWS": final_test_probs[:, 2],
    }
)

submission.to_csv("./submission/submission_5610053e997d406aa96a206a73cf7051.csv", index=False)
print(f"Submission saved to ./submission/submission_5610053e997d406aa96a206a73cf7051.csv")
print(f"Submission shape: {submission.shape}")
print(f"Submission columns: {list(submission.columns)}")

# Save models and processed data
os.makedirs("./working", exist_ok=True)
for fold, (xgb_model, lgb_model) in enumerate(fold_models):
    xgb_model.save_model(f"./working/xgb_fold_{fold}.json")
    lgb_model.booster_.save_model(f"./working/lgb_fold_{fold}.txt")

joblib.dump(scaler, "./working/scaler.pkl")
joblib.dump(char_vectorizer, "./working/char_vectorizer.pkl")
joblib.dump(word_vectorizer, "./working/word_vectorizer.pkl")
joblib.dump(label_encoder, "./working/label_encoder.pkl")
joblib.dump(selector, "./working/feature_selector.pkl")

print(f"\nFinal Validation Score: {best_score}")