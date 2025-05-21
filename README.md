# 🧠 Mental Health Text Classification
This project is a machine learning-based text classification system to detect and categorize various mental health conditions (like Anxiety, Depression, Suicidal thoughts, etc.) based on user-provided text input. It uses TF-IDF for feature extraction and a Random Forest classifier for prediction

## 📊 Dataset
- **Source**: [User-generated mental health statements](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health/data)
- **Size**: ~53,000 samples
- **Labels**:
  - Normal
  - Depression
  - Suicidal
  - Anxiety
  - Bipolar
  - Stress
  - Personality Disorder

## 🔧 Preprocessing Steps

1. Lowercasing
2. Removing punctuation and non-alphanumeric characters
3. Tokenization
4. Stemming (using `PorterStemmer`)
5. TF-IDF Vectorization

## 📈 Exploratory Data Analysis

- Bar and pie charts of label distribution
- Statement length distribution (with and without outliers)
- Word clouds for each mental health category
- Boxplots and bar charts for text length statistics

## 🧪 Model Training

Models tested:
- ✅ Random Forest Classifier (Best performance)
- Extra Trees Classifier
- Logistic Regression
- AdaBoost Classifier

### 🏆 Best Model: Random Forest

| Metric         | Test Accuracy | Train Accuracy | Precision (Test) |
|----------------|---------------|----------------|------------------|
| Random Forest  | **95.1%**     | 99.9%          | **95.0%**        |

## 💾 Model Saving

- `model.pkl` – Saved Random Forest model
- `vecctorizer.pkl` – Saved TF-IDF vectorizer

Saved using `joblib`.

## ✅ Key Insights
- Balanced training improves classifier fairness across all mental health categories.
- Longer statements tend to be linked to mental health struggles.
- TF-IDF vectorization provides strong feature extraction for mental health text classification.
