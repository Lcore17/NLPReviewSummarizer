# nlp_mini_project_part_1_machine_learning.py
import pandas as pd
import joblib
import re
import spacy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE
from scipy.sparse import hstack
from collections import defaultdict

# Initialize spaCy
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def load_data(data_path):
    """Load dataset and identify its type"""
    df = pd.read_csv(data_path)
    
    if 'sentiment' in df.columns:
        dataset_type = 'imdb'
        target_col = 'sentiment'
        text_col = 'review'
        class_names = ['negative', 'positive']
    elif 'category' in df.columns:
        dataset_type = 'amazon'
        target_col = 'category'
        text_col = 'text'  # Will be updated to 'review' in app.py
        # Clean category labels
        df[target_col] = df[target_col].str.strip().str.lower()
        class_names = sorted(df[target_col].unique().tolist())
    else:
        raise ValueError("Dataset must contain 'sentiment' or 'category' column")
    
    print(f"Loaded {len(df)} entries from {dataset_type} dataset")
    return df, dataset_type, target_col, text_col, class_names

def perform_eda(df, target_col, text_col):
    """Enhanced Exploratory Data Analysis"""
    print("\n=== Basic Dataset Info ===")
    print(df.info())
    
    # Class distribution
    plt.figure(figsize=(10, 5))
    sns.countplot(x=target_col, data=df)
    plt.title("Class Distribution")
    plt.xticks(rotation=45)
    plt.show()
    
    # Text length analysis
    df['text_length'] = df[text_col].apply(len)
    df['word_count'] = df[text_col].apply(lambda x: len(x.split()))
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df['text_length'], bins=50)
    plt.title("Text Length Distribution")
    
    plt.subplot(1, 2, 2)
    sns.histplot(df['word_count'], bins=50)
    plt.title("Word Count Distribution")
    plt.tight_layout()
    plt.show()
    
    # Missing values
    print("\n=== Missing Values ===")
    print(df.isnull().sum())

def preprocess_text(text, dataset_type):
    """Advanced text cleaning and preprocessing"""
    # Remove special characters and lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    doc = nlp(text.lower())
    
    # Lemmatization with filtering
    lemmas = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop
        and token.is_alpha
        and len(token.lemma_) > 2
    ]
    
    # Dataset-specific processing
    if dataset_type == 'amazon':
        lemmas += [
            token.text.lower()
            for token in doc
            if token.pos_ in ['ADJ', 'ADV']
            and len(token.text) > 2
        ]
    
    return ' '.join(lemmas)

def generate_features(df, dataset_type):
    """Generate multiple feature sets"""
    # Basic NLP features
    df['char_count'] = df['cleaned_text'].apply(len)
    df['word_count'] = df['cleaned_text'].apply(lambda x: len(x.split()))
    df['avg_word_length'] = df['cleaned_text'].apply(
        lambda x: np.mean([len(word) for word in x.split()]))
    
    # Aspect-based features for Amazon
    if dataset_type == 'amazon':
        aspects = {
            'packaging': ['package', 'wrap', 'box'],
            'pricing': ['price', 'cost', 'value'],
            'quality': ['quality', 'material', 'durable'],
            'service': ['service', 'support', 'staff'],
            'shipping': ['shipping', 'delivery', 'tracking']
        }
        for aspect, keywords in aspects.items():
            df[f'count_{aspect}'] = df['cleaned_text'].apply(
                lambda x: sum(1 for word in x.split() if word in keywords))
    
    # Text vectorization
    tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(df['cleaned_text'])
    
    bow = CountVectorizer(max_features=2000)
    X_bow = bow.fit_transform(df['cleaned_text'])
    
    return df, X_tfidf, X_bow, tfidf, bow

def train_evaluate_model(X, y, model, dataset_type, class_names, feature_name):
    """Train and evaluate a single model"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Handle class imbalance
    if dataset_type == 'amazon':
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    text_report = classification_report(y_test, y_pred, target_names=class_names)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'text_report': text_report  # Add text report
    }
    
    # Display results in console
    print(f"\n{model.__class__.__name__} ({feature_name}) Performance:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(text_report)  
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d',
                cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{model.__class__.__name__} Confusion Matrix ({feature_name})")
    plt.show()
    
    return metrics

def main():
    # Configuration
    DATA_PATH = "data/imdb.csv"  # Change to your dataset path
    
    # Load and prepare data
    df, dataset_type, target_col, text_col, class_names = load_data(DATA_PATH)
    perform_eda(df, target_col, text_col)
    
    # Preprocess text
    df['cleaned_text'] = df[text_col].apply(
        lambda x: preprocess_text(x, dataset_type))
    
    # Generate features
    df, X_tfidf, X_bow, tfidf, bow = generate_features(df, dataset_type)
    
    # Prepare target
    y = df[target_col]
    if dataset_type == 'imdb':
        y = y.map({'negative': 0, 'positive': 1})
    
    # Initialize models
    models = [
        XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        LogisticRegression(max_iter=1000, class_weight='balanced'),
        RandomForestClassifier(class_weight='balanced', n_jobs=-1),
        MultinomialNB()
    ]
    
    # Feature sets to evaluate
    feature_sets = {
        'NLP Features': df.filter(regex='count_|char|word|avg'),
        'BoW': X_bow,
        'TF-IDF': X_tfidf,
        'Combined': hstack([df.filter(regex='count_|char|word|avg').values, X_tfidf])
    }
    
    # Train and evaluate models
    results = {}
    for feature_name, X in feature_sets.items():
        print(f"\n=== Training with {feature_name} ===")
        feature_results = {}
        for model in models:
            metrics = train_evaluate_model(
                X, y, model, dataset_type, class_names, feature_name)
            feature_results[model.__class__.__name__] = metrics
        results[feature_name] = feature_results
    
    # Comparative analysis
    comparison = []
    for feature_name, models in results.items():
        for model_name, metrics in models.items():
            comparison.append({
                'Feature Set': feature_name,
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'F1-Score': metrics['classification_report']['weighted avg']['f1-score']
            })
    
    comparison_df = pd.DataFrame(comparison)
    print("\n=== Final Comparison ===")
    print(comparison_df.to_markdown(index=False))
    
    # Save best model
    best_model_info = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
    print(f"\nBest Model: {best_model_info['Model']} with {best_model_info['Feature Set']} features")
    joblib.dump({
        'model': next(m for m in models if m.__class__.__name__ == best_model_info['Model']),
        'vectorizer': tfidf if 'TF-IDF' in best_model_info['Feature Set'] else bow,
        'feature_set': best_model_info['Feature Set']
    }, 'best_model.pkl')

if __name__ == "__main__":
    main()