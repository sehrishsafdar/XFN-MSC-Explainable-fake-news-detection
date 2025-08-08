# XFN-MSC-Explainable-fake-news-detection
This is the official repository "Towards Explainable Fake News Classification via Multilingual Stance Classification (XFN-MSC)"

# XFN-MSC — Cross-lingual Fake News & Misinformation Stance Classification

This repository contains code and experiments for **stance detection** in misinformation contexts, using **XLM-RoBERTa** on multiple multilingual datasets.  
The goal is to classify the stance of a post or comment (e.g., *Support*, *Deny*, *Query*, *Comment*) towards a claim.

---

## 📂 Datasets Used

The project uses three public benchmark datasets:

1. **[SemEval-2017 Task 8: RumourEval](https://figshare.com/articles/dataset/rumdetect2017/25406389)**
   - **Domain**: Twitter rumour threads
   - **Labels**: Support, Deny, Query, Comment
   - **Description**: This dataset is a well-known benchmark dataset created to aid in stance detection
(Task A) as well as veracity prediction or fake news identification (Task B). It com-
prises Twitter conversation threads that begin with a source tweet addressing a rumor
or assertion, followed by several responses. In Task A, each response in the discussion
is marked with one of four stance labels: support, deny, query, or comment, which
reflect the user’s response to the original claim. In Task B, the original tweet is tagged
with a truthfulness label: true, false, or unverified, based on factual claim verification..

2. **[RumorEval-2019](https://figshare.com/articles/dataset/RumourEval_2019_data/8845580)**
   - **Domain**: Twitter and Reddit posts
   - **Labels**: Same as RumourEval 2017 with additional expanded coverage
   - **Description**: This dataset [88] focused on both stance detection (Task A) and fake news identifi-
cation (Task B). It consists of Twitter discussions on actual events, with each reply
tweet labeled for its stance on the original post, and each original post marked with
a truthfulness label (true, false, or unverified). This allows for the combined modeling
and assessment of both stance and truthfulness tasks.

3. **[AMUSED Dataset](https://github.com/sehrishsafdar/AMUSED-Dataset.git)**
   - **Domain**: Multilingual misinformation discussions
   - **Labels**: Claim Stance (Support, Deny, Neutral)
   - **Description**: A Multilingual Urdu/English Stance and Fake News Detection Corpus [89] comprising
news, user comments, and metadata in English, Urdu, and Roman Urdu. It supports
to both Task A and Task B, providing annotations for stance (comment versus title)
and classification of fake news. It also encompasses user engagement elements like
likes, shares, user profiles, and sentiments, rendering it exceptionally appropriate for
multilingual and context-aware modeling..

---

## ⚙️ Requirements

Run the following to install the required dependencies:

```bash
pip install transformers torch scikit-learn pandas numpy matplotlib seaborn lime textstat spacy nltk empath-client
python -m spacy download en_core_web_sm

`````
`````
 ##📑 Notebook Workflow

The main notebook: XFN-MSC.ipynb

## Setup & Imports

Installs dependencies and imports required Python libraries.

## Data Loading

Loads datasets from provided paths (AMUSED, SemEval, RumorEval).

Consolidates into a unified DataFrame.

Encodes labels using LabelEncoder.

## Preprocessing

Tokenizes text using XLM-RoBERTa tokenizer.

Splits into train/test sets.

## Model Training

Fine-tunes XLMRobertaForSequenceClassification using Hugging Face Trainer.

Uses early stopping and evaluation metrics.

## Evaluation

Generates accuracy, confusion matrix, classification report.

Plots ROC curves and calculates AUC scores.

## Explainability

Uses LIME to explain individual predictions.

## 🚀 Running the Notebook
Open XFN-MSC.ipynb in Google Colab or Jupyter Notebook.

Install dependencies from the requirements section.

Make sure dataset files are available in the specified paths.

Run cells in order to:

Load data

Train the model

Evaluate results

Visualize LIME explanations




