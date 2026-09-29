import pandas as pd
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    classification_report
)

# ==========================================
# 1. LOAD DATASET
# ==========================================

print("Loading dataset...")

df = pd.read_csv("dataset/spam.csv", encoding="latin-1")
print("\nDataset loaded successfully!")
print("Original shape:", df.shape)

print("\nOriginal columns:")
print(df.columns.tolist())


# ==========================================
# 2. KEEP REQUIRED COLUMNS
# ==========================================

df = df[["v1", "v2"]]

df.columns = ["label", "message"]

print("\nRequired columns selected:")
print(df.columns.tolist())


# ==========================================
# 3. CHECK MISSING VALUES
# ==========================================

print("\nMissing values:")
print(df.isnull().sum())

df = df.dropna()


# ==========================================
# 4. REMOVE DUPLICATES
# ==========================================

print("\nDuplicate rows:", df.duplicated().sum())

df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)


# ==========================================
# 5. CLEAN TEXT
# ==========================================

def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Keep letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["clean_message"] = df["message"].apply(clean_text)


# ==========================================
# 6. CONVERT LABELS
# ==========================================

# ham = 0 = Not Spam
# spam = 1 = Spam

df["label_numeric"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

print("\nClass distribution:")
print(df["label"].value_counts())


# ==========================================
# 7. TRAIN / TEST SPLIT
# ==========================================

X = df["clean_message"]
y = df["label_numeric"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 8. TF-IDF FEATURE EXTRACTION
# ==========================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF feature count:", len(vectorizer.get_feature_names_out()))


# ==========================================
# 9. TRAIN LOGISTIC REGRESSION
# ==========================================

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_tfidf, y_train)


# ==========================================
# 10. GENERATE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test_tfidf)

y_probability = model.predict_proba(X_test_tfidf)[:, 1]


# ==========================================
# 11. MODEL EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

conf_matrix = confusion_matrix(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# ==========================================
# 12. DISPLAY RESULTS
# ==========================================

print("\n===================================")
print("MODEL EVALUATION")
print("===================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Not Spam", "Spam"],
        zero_division=0
    )
)


# ==========================================
# 13. CONFUSION MATRIX GRAPH
# ==========================================

plt.figure(figsize=(7, 5))

sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Spam", "Spam"],
    yticklabels=["Not Spam", "Spam"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix - Spam Detection")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.close()


# ==========================================
# 14. ROC CURVE
# ==========================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Spam Detection")

plt.legend()

plt.tight_layout()

plt.savefig(
    "roc_curve.png",
    dpi=300
)

plt.close()


# ==========================================
# 15. SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "spam_model.pkl"
)

joblib.dump(
    vectorizer,
    "tfidf_vectorizer.pkl"
)


# ==========================================
# 16. FINAL OUTPUT
# ==========================================

print("\n===================================")
print("FILES SAVED")
print("===================================")

print("spam_model.pkl")
print("tfidf_vectorizer.pkl")
print("confusion_matrix.png")
print("roc_curve.png")

print("\nModel training completed successfully!")
