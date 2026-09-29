# 📧 Email Spam Detection

## 📌 Project Overview

This project uses Machine Learning to classify an email/message as **Spam** or **Not Spam**.

The project uses **Logistic Regression** for classification and **TF-IDF** to convert text messages into numerical features.

## 🎯 Objective

The main objective of this project is to predict whether an email/message is:

- Spam
- Not Spam

## 📂 Dataset

The project uses the **SMS Spam Collection** dataset.

The dataset contains two important columns:

- `label` – identifies the message as spam or ham (not spam)
- `message` – contains the text message

Original dataset size: **5572 rows**

After removing duplicate rows: **5169 rows**

Class distribution:

- Not Spam (ham): 4516
- Spam: 653

## 🔧 Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset using Pandas.
2. Selected the required `label` and `message` columns.
3. Checked for missing values.
4. Removed duplicate rows.
5. Converted text to lowercase.
6. Removed URLs.
7. Removed special characters and numbers.
8. Removed extra spaces.
9. Converted the text into numerical features using **TF-IDF**.
10. Split the dataset into training and testing sets.

## 🤖 Model Building

The project uses:

**Machine Learning Algorithm:** Logistic Regression

**Text Feature Extraction:** TF-IDF Vectorization

The dataset was divided into:

- Training data: 4135 messages
- Testing data: 1034 messages

The TF-IDF vectorizer generated **5000 features**.

## 📊 Model Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC-AUC

### Results

| Metric | Score |
|---|---:|
| Accuracy | 94.97% |
| Precision | 98.77% |
| Recall | 61.07% |
| F1 Score | 75.47% |
| ROC-AUC | 99.39% |

### Confusion Matrix

```text
[[902   1]
 [ 51  80]]