# Fake News Detection using Machine Learning

## Project Overview

This project builds a machine learning model to classify news articles as **Fake** or **Real** using Natural Language Processing (NLP) techniques and classical machine learning algorithms.

The project follows a complete machine learning workflow including:

- Data Exploration
- Data Preprocessing
- Feature Engineering
- Model Training
- Model Evaluation
- Prediction on New Articles

The project was developed as part of the AI/ML Online Assessment.

---

# Dataset

Dataset Used:

**Fake and Real News Dataset (Kaggle)**

The dataset contains two files:

- Fake.csv
- Real.csv

After merging both datasets:

| Description | Value |
|------------|-------|
| Total Articles | 44,898 |
| Fake Articles | 23,481 |
| Real Articles | 21,417 |

---

# Project Structure

```
fake-news-detection-ml/

│
├── artifacts/
│   ├── best_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── data/
│   ├── raw/
│   │   ├── Fake.csv
│   │   └── Real.csv
│   │
│   └── processed/
│       └── processed_news.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_Model_Training.ipynb
│   └── 05_Prediction.ipynb
│
├── reports/
│
├── predict.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Workflow

The project follows the workflow shown below.

```
Raw Dataset
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Data Preprocessing
      │
      ▼
TF-IDF Feature Engineering
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
Prediction
```

---

# Exploratory Data Analysis

The dataset was explored before model development.

The following analyses were performed:

- Dataset overview
- Missing value analysis
- Duplicate analysis
- Class distribution
- Text length analysis
- Subject analysis
- Date analysis
- Word frequency analysis
- N-gram analysis
- Correlation analysis
- Outlier analysis

### Important Findings

- Total articles: **44,898**
- No missing values in the original dataset
- 631 rows contained blank article text
- Duplicate titles and duplicate article bodies were found
- Dataset is reasonably balanced
- The **subject** column strongly leaked the class label and was removed before training

---

# Data Preprocessing

The following preprocessing steps were applied.

- Removed unnecessary columns
- Removed duplicate records
- Merged title and article text
- Converted text to lowercase
- Removed URLs
- Removed HTML tags
- Removed punctuation
- Removed numbers
- Removed stopwords
- Applied lemmatization

The cleaned dataset is saved as

```
data/processed/processed_news.csv
```

---

# Feature Engineering

The cleaned text was converted into numerical features using **TF-IDF Vectorization**.

### TF-IDF Parameters

| Parameter | Value |
|----------|-------|
| max_features | 20000 |
| ngram_range | (1,2) |
| min_df | 2 |
| max_df | 0.95 |

### Dataset Split

The dataset was divided using stratified sampling.

| Dataset | Percentage |
|---------|------------|
| Training | 70% |
| Validation | 10% |
| Testing | 20% |

Random seed used:

```
random_state = 42
```

---

# Machine Learning Models

Three classification models were trained.

- Multinomial Naive Bayes
- Logistic Regression
- Linear SVM

Each model was trained using the same TF-IDF features and evaluated on the validation dataset.

---

# Model Selection

The best model was selected based on **Validation F1 Score**.

| Model | Accuracy | Precision | Recall | F1 Score |
|--------|----------|-----------|--------|----------|
| Linear SVM | **0.9946** | **0.9948** | **0.9931** | **0.9940** |
| Logistic Regression | 0.9881 | 0.9919 | 0.9817 | 0.9869 |
| Multinomial Naive Bayes | 0.9568 | 0.9540 | 0.9502 | 0.9521 |

### Why Linear SVM?

Linear SVM achieved the highest validation F1-score among all trained models. Therefore, it was selected as the final model and evaluated on the unseen test dataset.

---

# Final Test Performance

The selected model achieved the following results on the test dataset.

| Metric | Score |
|---------|-------|
| Accuracy | **99.53%** |
| Precision | **99.74%** |
| Recall | **99.23%** |
| F1 Score | **99.48%** |

---

# Saved Artifacts

After training, the following files were saved.

```
artifacts/

best_model.pkl

tfidf_vectorizer.pkl
```

These files are used for prediction without retraining the model.

---

# Running the Project

## Step 1

Clone the repository.

```bash
git clone <repository-url>
```

---

## Step 2

Create a virtual environment.

```bash
python -m venv venv
```

Activate the environment.

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

---

## Step 3

Install all required libraries.

```bash
pip install -r requirements.txt
```

---

## Step 4

Run the notebooks in the following order.

```
01_EDA.ipynb

↓

02_Preprocessing.ipynb

↓

03_Feature_Engineering.ipynb

↓

04_Model_Training.ipynb

↓

05_Prediction.ipynb
```

---

## Step 5

Or directly run

```bash
python predict.py
```

Paste a news article when prompted.

The model will predict whether the article is **Fake** or **Real**.

---

# Reproducibility

The project is fully reproducible.

- Fixed random seed (`random_state = 42`)
- Train/Validation/Test split documented
- Data preprocessing steps documented
- TF-IDF parameters documented
- Python dependencies listed in `requirements.txt`
- Saved model and vectorizer included

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn
- Joblib
- Jupyter Notebook

---

# Possible Improvements

Some possible future improvements are:

- Transformer-based models (BERT, RoBERTa)
- Deep learning architectures
- Real-time fake news detection
- Streamlit or FastAPI deployment
- Cloud deployment

---

# Author

**Sarvesh Maurya**

MSc. Operations Research

Indian Institute of Technology Bombay

GitHub:
https://github.com/sarvesh0005/fake-news-detection-ml

LinkedIn:
www.linkedin.com/in/sarvesh-maurya005

