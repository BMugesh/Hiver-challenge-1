import os
import sys
import re
from typing import Dict, List, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stage5_retrieval import load_data, init_embedding_model
from src.stage6_grounded_reply import clean_twitter_noise

CLEAN_STOPWORDS = [
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "for", "from", "further", "had",
    "hadn", "has", "hasn", "have", "haven", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "if", "in", "into", "is", "isn", "it",
    "its", "itself", "let", "me", "more", "most", "mustn", "my", "myself", "of",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
    "over", "own", "same", "shan", "she", "should", "shouldn", "so", "some",
    "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "too", "under",
    "until", "up", "very", "was", "wasn", "we", "were", "weren", "while", "with",
    "won", "would", "wouldn", "you", "your", "yours", "yourself", "yourselves"
]

def normalize_query(text: str) -> str:
    cleaned = clean_twitter_noise(text).strip()
    lower = cleaned.lower()
    tokens = lower.split()
    context = []
    
    # Check for iOS 11 "I" bug / autocorrect patterns
    if re.search(r'\b(a\s*[\?\[\]]|letter\s+i|the\s+i\s+bug|the\s+i\s+glitch|typing\s+i|when\s+i\s+type\s+i|capital\s+i)\b|#?ios11bug', lower):
        context.append("keyboard typing autocorrect text replacement letter i glitch bug")

    if len(tokens) <= 8:
        if any(w in lower for w in ["wifi", "wi-fi", "bluetooth", "airdrop", "hotspot", "cellular"]):
            if any(w in lower for w in ["dead", "drop", "broken", "wont connect", "not working", "cant connect", "disconnect"]):
                context.append("connectivity wifi bluetooth network connection issue")
        if any(w in lower for w in ["battery", "drain", "draining", "charge", "charging", "died", "power"]):
            if any(w in lower for w in ["dead", "died", "fast", "quick", "wont charge", "dying"]):
                context.append("battery charging power drain percentage")
        if any(w in lower for w in ["type", "typing", "keyboard", "autocorrect", "predictive", "key"]):
            if any(w in lower for w in ["cant", "broken", "freeze", "lag", "glitch", "stuck"]):
                context.append("keyboard typing autocorrect predictive text issue")
        if any(w in lower for w in ["screen", "display", "touch", "unresponsive", "black screen", "flicker"]):
            context.append("display touch screen unresponsive touch issue")
        if any(w in lower for w in ["sound", "audio", "speaker", "volume", "mic", "microphone", "airpod"]):
            context.append("audio sound speaker volume microphone issue")
        if any(w in lower for w in ["billing", "charged", "refund", "subscription", "apple music", "itunes"]):
            context.append("app store purchases billing charge refund account")

    if context:
        return f"{cleaned} {' '.join(context)}"
    return cleaned

def compute_diagnostic_bonus(text: str, classes: List[str]) -> Dict[str, float]:
    lower = text.lower()
    bonuses = {cls: 0.0 for cls in classes}

    # Keyboard & Autocorrect (including iOS 11 'I' bug)
    if re.search(r'\b(keyboard|autocorrect|auto-correct|predictive|text\s+replacement|swipe|emoji|dictation|spacebar|letter\s+i|typing\s+(the\s+)?i|a\s*[\?]|a\s*\[\?\]|i\s*bug|i\s*glitch)\b|#?ios11bug', lower):
        bonuses["KEYBOARD_TYPING_AUTOCORRECT"] += 0.75

    # How-To vs General Device Inquiry
    is_howto = bool(re.search(r'\b(how\s+do\s+i|how\s+can\s+i|how\s+to|where\s+do\s+i|where\s+is\s+the|can\s+i\s+change|can\s+i\s+turn|how\s+would\s+i|how\s+does\s+one|how\s+to\s+enable|how\s+to\s+disable|how\s+to\s+set|how\s+to\s+turn)\b', lower))
    has_settings_kw = any(w in lower for w in ["setting", "settings", "dark mode", "night shift", "wallpaper", "ringtone", "notifications", "widget", "control center", "lock screen", "home screen", "font size", "display zoom", "passcode", "face id setup", "touch id setup", "dnd", "do not disturb", "airdrop setting"])
    has_breakage = any(w in lower for w in ["broken", "not working", "draining", "wont connect", "failed", "error", "crashing", "freeze", "stuck", "slow", "lag", "died", "dead"])

    if is_howto and not has_breakage:
        bonuses["HOW_TO_SETTINGS_CONFIGURATION"] += 0.55
        if has_settings_kw:
            bonuses["HOW_TO_SETTINGS_CONFIGURATION"] += 0.40

    # General Inquiry (device specs, features, compatibility, non-symptom)
    is_general = bool(re.search(r'\b(what\s+is|what\s+are|what\s+does|which\s+iphone|difference\s+between|does\s+the\s+iphone|is\s+it\s+worth|compatible\s+with|specifications|specs|warranty|applecare|trade\s*in|unboxing|apple\s+store\s+hours)\b', lower))
    if is_general and not has_breakage:
        bonuses["GENERAL_DEVICE_INQUIRY"] += 0.50

    # OS Update & System Performance
    has_update = any(w in lower for w in ["update", "updated", "updating", "ios 11", "ios 10", "ios 12", "new ios", "software update", "ios11", "11.0.3", "11.1"])
    has_perf = any(w in lower for w in ["slow", "lag", "laggy", "sluggish", "freeze", "freezing", "restart loop", "storage full", "other storage", "battery drain after", "draining after update", "since update", "after updating", "since upgrading", "after ios", "boot loop", "rebooting"])
    if has_update and has_perf:
        bonuses["OS_UPDATE_SYSTEM_PERFORMANCE"] += 0.70
    elif has_update and any(w in lower for w in ["cant update", "wont update", "failed to update", "stuck on verifying", "estimated time", "error downloading"]):
        bonuses["OS_UPDATE_SYSTEM_PERFORMANCE"] += 0.60

    # Connectivity
    if any(w in lower for w in ["wifi", "wi-fi", "bluetooth", "airdrop", "hotspot", "cellular", "lte", "no service", "carrier", "pairing", "disconnects", "carplay", "sim card"]):
        bonuses["CONNECTIVITY_WIFI_BLUETOOTH"] += 0.70

    # Battery & Power
    if any(w in lower for w in ["battery", "charger", "charging", "cable", "lightning", "drain", "draining", "power down", "shut off", "percentage", "battery health", "overheating"]) and not (has_update and has_perf):
        bonuses["BATTERY_CHARGING_POWER"] += 0.65

    # Account, Apple ID, iCloud
    if any(w in lower for w in ["apple id", "appleid", "icloud", "password", "passcode", "locked out", "two factor", "2fa", "verification code", "itunes account", "activation lock", "forgot password", "reset password"]):
        bonuses["ACCOUNT_APPLEID_ICLOUD"] += 0.70

    # App Store, Purchases, Billing
    if any(w in lower for w in ["app store", "purchase", "subscription", "billed", "charged", "refund", "receipt", "in-app", "billing address", "payment method", "itunes gift card", "duplicate charge", "unauthorized charge"]):
        bonuses["APP_STORE_PURCHASES_BILLING"] += 0.70

    # App Crash and Download
    if any(w in lower for w in ["app crash", "app crashing", "crashes on open", "cant download app", "app wont open", "wont download", "app frozen", "app store download", "cant install app", "apps crash", "keeps crashing", "app closes"]):
        bonuses["APP_CRASH_AND_DOWNLOAD"] += 0.75

    # Audio, Sound, Speaker
    if any(w in lower for w in ["audio", "sound", "speaker", "microphone", "mic", "airpods", "earbuds", "headphones", "no sound", "distorted sound", "static", "crackling", "earpiece"]):
        bonuses["AUDIO_SOUND_SPEAKER"] += 0.70

    # Display & Touch
    if any(w in lower for w in ["touch screen", "touchscreen", "screen unresponsive", "ghost touch", "black screen", "lines on screen", "cracked screen", "glitchy screen", "screen flicker", "touch id", "face id"]):
        bonuses["DISPLAY_TOUCH_SCREEN"] += 0.70

    return bonuses

train_df, val_df, test_df, _, _ = load_data()
model = init_embedding_model()

train_raw = train_df["customer_text"].fillna("").astype(str).tolist()
train_norm = [normalize_query(t) for t in train_raw]
train_labels = train_df["intent_id"].astype(str).tolist()

vec = TfidfVectorizer(max_features=12000, stop_words=CLEAN_STOPWORDS, ngram_range=(1, 3), sublinear_tf=True)
X_train = vec.fit_transform(train_norm)

classes, counts = np.unique(train_labels, return_counts=True)
total_samples = len(train_labels)
n_classes = len(classes)
smoothed_weights = {
    cls: float((total_samples / (n_classes * count)) ** 0.50)
    for cls, count in zip(classes, counts)
}

clf = LogisticRegression(max_iter=1000, random_state=42, C=2.0, class_weight=smoothed_weights)
clf.fit(X_train, train_labels)

# Prototypes
train_embs = model.encode(train_norm, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
df_emb = pd.DataFrame({"label": train_labels})
centroids = {}
for cls in clf.classes_:
    idxs = df_emb[df_emb["label"] == cls].index.values
    c_vec = np.mean(train_embs[idxs], axis=0)
    c_vec = c_vec / np.linalg.norm(c_vec)
    centroids[cls] = c_vec

def evaluate_set(df, name="Validation"):
    raw = df["customer_text"].fillna("").astype(str).tolist()
    norm = [normalize_query(t) for t in raw]
    true_labels = df["intent_id"].astype(str).tolist()

    X = vec.transform(norm)
    lr_probs = clf.predict_proba(X)
    embs = model.encode(norm, batch_size=128, show_progress_bar=False, normalize_embeddings=True)

    preds = []
    for i, norm_t in enumerate(norm):
        p_lr = lr_probs[i]
        emb = embs[i]
        diag = compute_diagnostic_bonus(norm_t, clf.classes_)
        fused = []
        for j, cls in enumerate(clf.classes_):
            sim = float(np.dot(emb, centroids[cls]))
            bonus = diag[cls]
            logit = np.log(max(p_lr[j], 1e-6)) + (0.40 * sim) + (0.35 * bonus)
            fused.append(logit)
        fused = np.array(fused)
        exp_f = np.exp(fused - np.max(fused))
        probs = exp_f / np.sum(exp_f)
        preds.append(clf.classes_[int(np.argmax(probs))])

    acc = accuracy_score(true_labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(true_labels, preds, average="macro", zero_division=0)
    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(true_labels, preds, labels=clf.classes_, average=None, zero_division=0)
    print(f"\n--- {name} Results (N={len(df)}) ---")
    print(f"Accuracy: {acc * 100:.2f}% ({int(acc * len(df))} / {len(df)})")
    print(f"Macro Precision: {p * 100:.2f}% | Macro Recall: {r * 100:.2f}% | Macro F1: {f1:.4f}")
    df_per = pd.DataFrame({"intent": clf.classes_, "precision": np.round(p_per, 4), "recall": np.round(r_per, 4), "f1": np.round(f1_per, 4), "support": sup_per}).sort_values(by="f1", ascending=False)
    print(df_per.to_string(index=False))

evaluate_set(test_df, "Test Split (Unseen)")
