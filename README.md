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

## Dataset Setup

The dataset is not included in this repository as per the assignment instructions.

Download the Fake and Real News Dataset from the provided Kaggle link and place the files as shown below:

```
data/
└── raw/
    ├── Fake.csv
    └── Real.csv
```

After placing the dataset in the `data/raw/` folder, run the notebooks in the following order:

1. 01_EDA.ipynb
2. 02_Preprocessing.ipynb
3. 03_Feature_Engineering.ipynb
4. 04_Model_Training.ipynb
5. 05_Prediction.ipynb

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

Machine learning models cannot process raw text directly. Therefore, the cleaned news articles were converted into numerical feature vectors using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

TF-IDF was selected because it assigns higher importance to informative words while reducing the influence of very common words. It is computationally efficient, easy to interpret, and performs well with classical machine learning algorithms such as Logistic Regression and Linear SVM.

### TF-IDF Parameters

| Parameter | Value | Reason |
|----------|-------|--------|
| max_features | 20000 | Limits the vocabulary size to the most informative words, reducing memory usage and noise. |
| ngram_range | (1,2) | Uses both single words (unigrams) and two-word phrases (bigrams) to capture more context. |
| min_df | 2 | Removes words that appear only once, as they usually contribute little to model performance. |
| max_df | 0.95 | Removes extremely common words that appear in almost every document and carry little discriminative information. |

### Dataset Split

The dataset was divided using stratified sampling to maintain the class distribution across all subsets.

| Dataset | Percentage |
|---------|------------|
| Training | 70% |
| Validation | 10% |
| Testing | 20% |

The split was performed using **random_state = 42** to ensure reproducibility of the experimental results.
```

---

# Machine Learning Models

Three classification models were trained.

- Multinomial Naive Bayes
- Logistic Regression
- Linear SVM

Each model was trained using the same TF-IDF features and evaluated on the validation dataset.

## Why These Models?

### Multinomial Naive Bayes

Multinomial Naive Bayes is one of the most common baseline models for text classification. It is simple, computationally efficient, and works well with TF-IDF features. This model was used to establish a strong baseline for comparison.

### Logistic Regression

Logistic Regression is a linear classification algorithm that performs well on high-dimensional text data. It is easy to train, produces stable results, and is widely used for binary text classification problems.

### Linear SVM

Linear Support Vector Machine (Linear SVM) is considered one of the strongest classical machine learning algorithms for text classification. It performs well on sparse and high-dimensional feature spaces created by TF-IDF vectorization and often achieves better generalization than other linear models.

---

## Why Compare Multiple Models?

Instead of selecting a single algorithm directly, three different models were trained using the same preprocessing steps and TF-IDF features.

Comparing multiple models helps identify the algorithm that performs best on the given dataset. This makes the model selection process objective and data-driven rather than based on assumptions.
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

