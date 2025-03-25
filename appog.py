# app.py
import streamlit as st
import pandas as pd
import spacy
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from collections import Counter
from nrclex import NRCLex
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from transformers import pipeline
import torch
import re
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import joblib
from torch import nn
import gensim.downloader as api
import torch.serialization
import pickle

# NLP Initialization
nlp = spacy.load("en_core_web_sm")
nltk.download(['vader_lexicon', 'punkt', 'stopwords'])
sia = SentimentIntensityAnalyzer()
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
stop_words = set(stopwords.words('english'))
bert_sentiment = None
lstm_model = None
tokenizer = None

# Set up page config
st.set_page_config(
    page_title="NLP Review Summarizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("css/styles.css")

# Add LSTM model class definition
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, 
                 hidden_dim=256, output_dim=1, n_layers=2,
                 dropout=0.5, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.fc = nn.Linear(
            hidden_dim * 2 if bidirectional else hidden_dim, 
            output_dim
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])
            
        return self.fc(hidden)
    
def preprocess_for_lstm(text):
    """EXACT match of training preprocessing"""
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Process with spaCy
    doc = nlp(text.lower())
    tokens = [
        token.lemma_.lower() 
        for token in doc 
        if not token.is_stop 
        and token.is_alpha 
        and len(token.lemma_) > 1
    ]
    return ' '.join(tokens)

# Replace the duplicate/incorrect load_lstm_model with this corrected version
def load_lstm_model():
    try:
        if not os.path.exists('lstm_model.pth'):
            raise FileNotFoundError("LSTM model file not found")
        if not os.path.exists('tokenizer.pkl'):
            raise FileNotFoundError("Tokenizer file not found")

        tokenizer = joblib.load('tokenizer.pkl')
        checkpoint = torch.load('lstm_model.pth', map_location='cpu')

        lstm_model = SentimentLSTM(
            vocab_size=checkpoint['config']['vocab_size'],
            embedding_dim=checkpoint['config']['embedding_dim'],
            hidden_dim=checkpoint['config']['hidden_dim'],
            output_dim=checkpoint['config']['output_dim'],
            n_layers=checkpoint['config']['n_layers'],
            dropout=checkpoint['config']['dropout'],
            bidirectional=checkpoint['config']['bidirectional']
        )
        lstm_model.load_state_dict(checkpoint['model_state_dict'])
        lstm_model.eval()

        # Store in session state
        st.session_state.lstm_model = lstm_model
        st.session_state.lstm_tokenizer = tokenizer
        return True
    except Exception as e:
        st.error(f"LSTM Loading Failed: {str(e)}")
        st.session_state.lstm_model = None
        st.session_state.lstm_tokenizer = None
        return False

def preprocess_text(text):
    """NLP preprocessing pipeline"""
    doc = nlp(text)
    cleaned = [token.lemma_.lower() for token in doc 
              if not token.is_stop and not token.is_punct and token.is_alpha]
    return " ".join(cleaned)

def analyze_sentiment(text):
    """Enhanced sentiment analysis with sarcasm detection"""
    blob = TextBlob(text)
    vs = sia.polarity_scores(text)
    
    # Sarcasm detection heuristics
    sarcasm_triggers = {
        'exclamation_count': text.count('!') > 3,
        'contrast_words': len(set(['but', 'however', 'although']) & set(text.lower().split())) > 0,
        'hyperbole_words': len(re.findall(r'\b(extremely|utterly|absolutely|ridiculously)\b', text)) > 2,
        'negative_positive_ratio': len(re.findall(r'\b(not|never|no|none)\b', text)) / (len(text.split())+1) > 0.1
    }
    
    sarcasm_score = sum(sarcasm_triggers.values()) / len(sarcasm_triggers)
    adjusted_compound = vs['compound'] * (1 - sarcasm_score*0.7)
    
    return {
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity,
        'compound': adjusted_compound,
        'sentiment': 'positive' if adjusted_compound >= 0 else 'negative',
        'sarcasm_score': sarcasm_score
    }

def analyze_bert_sentiment(text):
    """Context-aware BERT analysis"""
    global bert_sentiment
    try:
        if bert_sentiment is None:
            bert_sentiment = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                device=0 if torch.cuda.is_available() else -1
            )
        
        result = bert_sentiment(text[:512])[0]
        
        # Detect contrastive context
        if 'but' in text.lower() or 'however' in text.lower():
            result['score'] *= 0.6  # Penalize positive scores
        
        # Detect sarcasm patterns
        if result['label'] == 'POSITIVE' and text.count('!') > 2:
            result['score'] = max(0, result['score'] - 0.3)
            
        return {
            'label': result['label'],
            'score': result['score']
        }
    except Exception as e:
        st.error(f"BERT analysis failed: {str(e)}")
        return {'label': 'ERROR', 'score': 0.0}\
        
# app.py - Update analyze_lstm_sentiment():
def analyze_lstm_sentiment(text):
    if 'lstm_model' not in st.session_state or 'lstm_tokenizer' not in st.session_state:
        return {'label': 'ERROR', 'score': 0.0, 'error': 'Model not initialized'}
    
    lstm_model = st.session_state.lstm_model
    tokenizer = st.session_state.lstm_tokenizer
    
    try:
        print("Processing text:", text)
        processed = preprocess_for_lstm(text)
        print("Processed text:", processed)
        if len(processed.split()) < 1:
            return {'label': 'NEUTRAL', 'score': 0.5, 'warning': 'No meaningful text after preprocessing'}
        
        max_len = tokenizer['max_len']
        words = processed.split()[:max_len]
        print("Words after splitting:", words)
        indexed = [tokenizer['vocab'].get(word, tokenizer['unk_idx']) for word in words]
        print("Indexed list:", indexed)
        
        padded = indexed + [tokenizer['pad_idx']] * (max_len - len(indexed))
        print("Padded list:", padded)
        
        tensor = torch.LongTensor(padded).unsqueeze(0)
        print("Tensor shape:", tensor.shape)
        
        with torch.no_grad():
            output = lstm_model(tensor)
            print("Model output:", output)
            probability = torch.sigmoid(output).item()
            print("Probability:", probability)
            
        label = 'POSITIVE' if probability > 0.5 else 'NEGATIVE'
        confidence = probability if label == 'POSITIVE' else 1 - probability
        
        return {
            'label': label,
            'score': confidence,
            'debug_info': {
                'original_length': len(text),
                'processed_length': len(processed.split()),
                'final_sequence_length': len(padded),
                'unk_count': indexed.count(tokenizer['unk_idx']),
                'pad_count': padded.count(tokenizer['pad_idx'])
            }
        }
        
    except Exception as e:
        print("Error occurred:", str(e))
        return {
            'label': 'ERROR',
            'score': 0.0,
            'error': str(e)
        }

def generate_verdict(review_text, basic_sentiment, bert_result, lstm_result=None):
    """Hybrid verdict logic combining multiple analysis results"""
    # Strong negative indicators
    strong_negative = {'awful', 'robbed', 'boring', 'gory', 'cringy', 'avoid', 'worst'}
    negative_keywords = ['never', 'doesn\'t matter', 'superfluous', 'destroy', 'waste']
    positive_keywords = ['good', 'best', 'enjoyable', 'excellent', 'recommend']
    
    # Keyword-based analysis
    keyword_balance = (
        sum(review_text.lower().count(k) for k in negative_keywords + list(strong_negative)) - 
        sum(review_text.lower().count(k) for k in positive_keywords)
    )

    # Immediate strong negative detection
    if any(word in review_text.lower() for word in strong_negative):
        return "Strong Negative (Critical Issues Detected)"

    # Base negativity score from basic and BERT analysis
    negativity_score = (
        (1 - basic_sentiment['compound']) * 0.4 +
        (1 - bert_result['score'] if bert_result and bert_result['label'] == 'POSITIVE' 
         else bert_result['score'] if bert_result else 0) * 0.4 +
        basic_sentiment['sarcasm_score'] * 0.2
    )

    # Incorporate LSTM results if available and valid
    if lstm_result and lstm_result.get('label') not in ['ERROR', None]:
        try:
            # Convert LSTM prediction to negativity contribution
            lstm_negativity = (1 - lstm_result['score']) if lstm_result['label'] == 'POSITIVE' \
                               else lstm_result['score']
            negativity_score += lstm_negativity * 0.2
            negativity_score /= 1.2  # Normalize back to 0-1 scale
        except KeyError:
            pass  # Ignore invalid LSTM results

    # Combine with keyword balance
    final_negativity = negativity_score + (keyword_balance * 0.05)

    # Determine final verdict
    if final_negativity > 0.7 or keyword_balance > 4:
        return "Negative (Strong Criticism)"
    elif final_negativity > 0.55 or keyword_balance > 2:
        return "Mostly Negative (Critical Observations)"
    elif final_negativity > 0.45:
        return "Mixed (Balanced Perspective)"
    elif final_negativity > 0.3:
        return "Mostly Positive (Generally Favorable)"
    else:
        return "Positive (Strong Recommendation)"

def extract_key_phrases(text):
    """Extract named entities and noun phrases"""
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    return entities, noun_phrases

def generate_word_cloud(text):
    """Create styled word cloud"""
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='#1b1b2f',
        colormap='viridis'
    ).generate(text)
    return wordcloud

def highlight_key_terms(text, terms):
    """Highlight important terms in text"""
    for term in terms:
        text = text.replace(term, f"<span class='highlight'>{term}</span>")
    return text

def generate_summary(text, ratio=0.2):
    """Custom extractive text summarization"""
    sentences = sent_tokenize(text)
    if len(sentences) < 3:
        return "Text too short for meaningful summary"
    
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    freq_dist = nltk.FreqDist(words)
    
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in freq_dist:
                sentence_scores[i] = sentence_scores.get(i, 0) + freq_dist[word]
    
    num_sentences = max(1, int(len(sentences) * ratio))
    top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    
    return ' '.join([sentences[i] for i in sorted(top_sentences)])

def topic_modeling(texts, n_topics=3, method='lda'):
    """Identify key themes in reviews"""
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)
    dtm = vectorizer.fit_transform(texts)
    
    if method == 'lda':
        model = LatentDirichletAllocation(n_components=n_topics)
    else:
        model = NMF(n_components=n_topics)
        
    model.fit(dtm)
    return {
        'vectorizer': vectorizer,
        'model': model,
        'topics': model.transform(dtm)
    }

def emotion_analysis(text):
    """Detect emotional dimensions using NRCLex"""
    emotion = NRCLex(text)
    return emotion.affect_frequencies

def aspect_sentiment_analysis(text, dataset_type='single'):
    """Analyze sentiment for specific aspects"""
    aspects = {
        'amazon': {
            'pricing': ['price', 'cost', 'value', 'expensive', 'cheap', 'affordable'],
            'quality': ['quality', 'material', 'durable', 'craftsmanship', 'build'],
            'packaging': ['packaging', 'wrap', 'box', 'container', 'damage'],
            'service': ['service', 'support', 'staff', 'help', 'response'],
            'shipping': ['shipping', 'delivery', 'arrived', 'package', 'tracking', 'carrier']
        },
        'imdb': {
            'acting': ['acting', 'performance', 'role', 'character'],
            'plot': ['story', 'plot', 'narrative', 'twist'],
            'production': ['cinematography', 'direction', 'music', 'effects']
        },
        'single': {
            'pricing': ['price', 'cost', 'value', 'expensive'],
            'quality': ['quality', 'material', 'durable', 'craftsmanship'],
            'packaging': ['packaging', 'wrap', 'box', 'damage'],
            'service': ['service', 'support', 'staff', 'help'],
            'shipping': ['shipping', 'delivery', 'package', 'tracking']
        }
    }
    
    doc = nlp(text)
    results = {}
    current_aspects = aspects[dataset_type]
    
    for aspect, keywords in current_aspects.items():
        sentences = [sent.text for sent in doc.sents 
                    if any(keyword in sent.text.lower() for keyword in keywords)]
        if sentences:
            aspect_sentiment = np.mean([analyze_sentiment(s)['compound'] for s in sentences])
            results[aspect] = aspect_sentiment
            
    return results

def analyze_dataset(df):
    """Perform comprehensive analysis on dataset"""
    df['cleaned_text'] = df['review'].apply(preprocess_text)
    
    # Detect dataset type
    dataset_type = 'amazon' if 'category' in df.columns else 'imdb'

    # Calculate aspect sentiment scores
    aspect_columns = ['pricing', 'quality', 'packaging', 'service', 'shipping']
    if dataset_type == 'amazon':
        # Calculate aspect sentiment scores
        aspect_scores = df['review'].apply(
            lambda x: aspect_sentiment_analysis(x, dataset_type)
        )
        aspect_df = pd.DataFrame(list(aspect_scores)).fillna(0)
        df = pd.concat([df, aspect_df], axis=1)
        
        # Calculate average sentiment scores for aspects
        aspect_metrics = {
            col: (df[col].mean() + 1) / 2  # Convert from [-1,1] to [0,1]
            for col in aspect_columns if col in df.columns
        }
    else:
        aspect_metrics = None
        if 'sentiment' not in df.columns:
            df['sentiment'] = df['review'].apply(
                lambda x: analyze_sentiment(x)['sentiment']
            )

    # Extract entities
    all_entities = []
    for doc in nlp.pipe(df['review'], batch_size=50):
        all_entities.extend([(ent.text, ent.label_) for ent in doc.ents])
    
    # TF-IDF Analysis
    tfidf = TfidfVectorizer(max_features=50)
    tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
    important_terms = tfidf.get_feature_names_out()
    
    return {
        'processed_df': df,
        'entities': all_entities,
        'important_terms': important_terms,
        'tfidf_matrix': tfidf_matrix,
        'dataset_type': dataset_type,
        'aspect_metrics': aspect_metrics
    }

def initialize_ml_models(selected_models):
    """Initialize selected ML models"""
    models = {
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(),
        'Naive Bayes': MultinomialNB()
    }
    return {name: models[name] for name in selected_models}

def train_ml_models(df, selected_models):
    """Train ML models on labeled data"""
    try:
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(df['review'])
        y = df['sentiment'].map({'positive': 1, 'negative': 0})
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        models = initialize_ml_models(selected_models)
        for name, model in models.items():
            model.fit(X_train, y_train)
        return models, vectorizer
    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
        return None, None

def ml_predict(text, models, vectorizer):
    """Make predictions using trained ML models"""
    results = {}
    text_vec = vectorizer.transform([text])
    for name, model in models.items():
        try:
            pred = model.predict(text_vec)[0]
            results[name] = 'Positive' if pred == 1 else 'Negative'
        except Exception as e:
            results[name] = f'Error: {str(e)}'
    return results

def main():
    st.title("📊 NLP Review Summarizer")
    
    with st.sidebar:
        st.header("Data Input")
        uploaded_file = st.file_uploader("Upload Reviews Dataset", type=["csv"])
        
        st.markdown("---")
        st.markdown("### ML/DL Algorithms")
        ml_options = ['XGBoost', 'Logistic Regression', 'Random Forest', 'Naive Bayes']
        selected_ml = st.multiselect("Select Algorithms", ml_options, default=['XGBoost'])
        
        st.markdown("---")
        st.markdown("### Advanced Features")
        enable_topics = st.checkbox("Enable Topic Modeling")
        enable_emotion = st.checkbox("Enable Emotion Detection")
        enable_aspect = st.checkbox("Enable Aspect Analysis")
        enable_similarity = st.checkbox("Enable Semantic Search")
        
        st.markdown("---")
        st.markdown("### Advanced Models")
        enable_bert = st.checkbox("Enable BERT Analysis")
        enable_lstm = st.checkbox("Enable LSTM Analysis")
        if enable_bert:
            bert_task = st.selectbox("BERT Task Type", ["sentiment-analysis", "text-classification"])
        if enable_lstm:
            if st.button("Initialize LSTM Model"):
                with st.spinner("Loading LSTM model..."):
                    if load_lstm_model():
                        st.success("LSTM model loaded!")
                    else:
                        st.error("Failed to load LSTM model")
            # Check if model is already loaded
            if 'lstm_model' not in st.session_state:
                st.warning("Click 'Initialize LSTM Model' to load the model")
        
        st.markdown("---")
        st.markdown("NLP Mini-Project A")

    # Initialize ML components
    ml_models, ml_vectorizer = None, None
    
    if uploaded_file:
        # Dataset analysis mode
        df = pd.read_csv(uploaded_file)
        if 'review' not in df.columns:
            st.error("Dataset must contain 'review' column")
            return
        
        # Train ML models if labels available
        if 'sentiment' in df.columns and selected_ml:
            with st.spinner("Training ML models..."):
                ml_models, ml_vectorizer = train_ml_models(df, selected_ml)
        
        analysis_results = analyze_dataset(df)
        processed_df = analysis_results['processed_df']
        dataset_type = analysis_results['dataset_type']

        st.header("📈 Dataset Overview")
        if dataset_type == 'imdb':
            cols = st.columns(4)
            cols[0].metric("Total Reviews", len(df))
            cols[1].metric("Positive", f"{processed_df['sentiment'].value_counts(normalize=True).get('positive', 0):.1%}")
            cols[2].metric("Negative", f"{processed_df['sentiment'].value_counts(normalize=True).get('negative', 0):.1%}")
            if enable_bert:
                with cols[3], st.spinner("BERT Analysis"):
                    try:
                        sample_reviews = df['review'].sample(100).tolist()
                        # Process reviews in batches
                        bert_results = []
                        for review in sample_reviews:
                            result = analyze_bert_sentiment(review)
                            if result['label'] != 'ERROR':
                                bert_results.append(result)
                        
                        if bert_results:
                            bert_pos = sum(1 for res in bert_results if res['label'] == 'POSITIVE')/len(bert_results)
                            st.metric("BERT Positive", f"{bert_pos:.1%}")
                        else:
                            st.warning("No valid BERT results")
                    except Exception as e:
                        st.error(f"BERT processing failed: {str(e)}")
        else:
            cols = st.columns(6)
            cols[0].metric("Total Reviews", len(df))
            aspects = ['Shipping', 'Pricing', 'Quality', 'Packaging', 'Service']
            for i, aspect in enumerate(aspects):
                col_name = aspect.lower()
                percentage = analysis_results['aspect_metrics'].get(col_name, 0)
                cols[i+1].metric(aspect, f"{percentage:.1%}")
                with cols[i+1]:
                    st.markdown(f'<div class="metric-card"><h3>{aspect}</h3><h1>{percentage:.1%}</h1></div>', 
                               unsafe_allow_html=True)
        
        # Sentiment analysis
        st.markdown('<div class="section-header">Aspect Analysis</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if dataset_type == 'imdb':
                fig = px.pie(processed_df, names='sentiment', 
                            title="Sentiment Distribution", hole=0.4)
            else:
                aspect_means = processed_df[['pricing', 'quality', 'packaging', 'service', 'shipping']].mean().reset_index()
                aspect_means.columns = ['Aspect', 'Average Score']
                fig = px.bar(aspect_means, x='Aspect', y='Average Score',
                            title="Average Aspect Scores", color='Aspect',
                            color_discrete_sequence=px.colors.sequential.Viridis)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if dataset_type == 'imdb':
                fig = px.histogram(processed_df, x='sentiment', color='sentiment',
                                  color_discrete_map={'positive':'#4ecdc4', 'negative':'#ff6b6b'})
            else:
                fig = px.box(processed_df.melt(value_vars=['pricing', 'quality', 'packaging', 'service', 'shipping']),
                            x='variable', y='value', color='variable',
                            labels={'variable': 'Aspect', 'value': 'Score'},
                            title="Aspect Score Distribution",
                            color_discrete_sequence=px.colors.sequential.Viridis)
            st.plotly_chart(fig, use_container_width=True)
        
        # Advanced Features Section
        st.markdown('<div class="section-header">Advanced Analytics</div>', unsafe_allow_html=True)
        
        # Topic Modeling
        if enable_topics:
            st.markdown('#### Topic Modeling')
            col1, col2 = st.columns([1, 3])
            with col1:
                n_topics = st.slider("Number of Topics", 2, 5, 3)
                method = st.selectbox("Algorithm", ['lda', 'nmf'])
                
            with col2:
                topic_results = topic_modeling(processed_df['cleaned_text'], n_topics, method)
                topics = []
                for idx, topic in enumerate(topic_results['model'].components_):
                    top_terms = [topic_results['vectorizer'].get_feature_names_out()[i] 
                                for i in topic.argsort()[-5:]]
                    topics.append(f"Topic {idx+1}: " + ", ".join(top_terms))
                
                st.markdown("**Discovered Topics:**")
                st.write("\n".join(topics))
        
        # Semantic Search
        if enable_similarity:
            st.markdown('#### Semantic Search')
            query = st.text_input("Find similar reviews:")
            if query:
                with st.spinner("Searching similar reviews..."):
                    embeddings = sentence_model.encode(processed_df['review'].tolist() + [query])
                    sim_scores = cosine_similarity([embeddings[-1]], embeddings[:-1])[0]
                    top_indices = sim_scores.argsort()[-3:][::-1]
                    similar_reviews = processed_df.iloc[top_indices]
                    
                    for idx, row in similar_reviews.iterrows():
                        st.markdown(f"**Similarity Score: {sim_scores[idx]:.2f}**")
                        st.write(row['review'])
                        st.markdown("---")
        
        # Word cloud
        st.markdown('<div class="section-header">Text Insights</div>', unsafe_allow_html=True)
        all_text = ' '.join(processed_df['cleaned_text'])
        wordcloud = generate_word_cloud(all_text)
        st.markdown('<div class="wordcloud-container">', unsafe_allow_html=True)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Entity analysis
        st.markdown('<div class="section-header">Named Entity Recognition</div>', unsafe_allow_html=True)
        entity_df = pd.DataFrame(analysis_results['entities'], columns=['Entity', 'Type'])
        top_entities = entity_df.groupby(['Type', 'Entity']).size()\
                        .reset_index(name='Count')\
                        .sort_values('Count', ascending=False).head(20)
        fig = px.sunburst(top_entities, path=['Type', 'Entity'], values='Count',
                         title="Entity Type Distribution", color='Count',
                         color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        # Single review analysis
        st.header("🔍 Single Review Analysis")
        review = st.text_area(
            "Enter your review for analysis:",
            height=200,
            placeholder="Paste your customer review here..."
        )
        
        if st.button("Analyze Text"):
            if len(review) < 50:
                st.error("Please enter a longer review (minimum 50 characters)")
                return
                
            with st.spinner("Performing deep analysis..."):
                cleaned_text = preprocess_text(review)
                sentiment = analyze_sentiment(review)
                entities, noun_phrases = extract_key_phrases(review)
                bert_result = analyze_bert_sentiment(review) if enable_bert else None
                lstm_result = analyze_lstm_sentiment(review) if enable_lstm else None
                final_verdict = generate_verdict(review, sentiment, bert_result, lstm_result)

                # ML Predictions
                ml_results = None
                if ml_models and ml_vectorizer:
                    ml_results = ml_predict(review, ml_models, ml_vectorizer)
                
                # Term Frequency Analysis
                vectorizer = TfidfVectorizer(max_features=10)
                tfidf = vectorizer.fit_transform([cleaned_text])
                important_terms = vectorizer.get_feature_names_out()
                highlighted_review = highlight_key_terms(review, important_terms)
                
                # Main content layout
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="plot-container">{highlighted_review}</div>', 
                               unsafe_allow_html=True)
                    
                    tabs = st.tabs(["📊 Sentiment", "🔑 Entities", "📈 Statistics", "🎭 Emotions"])
                    with tabs[0]:
                        if len(review.split()) > 100:
                            with st.expander("Text Summary"):
                                summary = generate_summary(review)
                                st.write(summary)
                        
                        cols = st.columns(4)
                        cols[0].metric("Polarity", f"{sentiment['polarity']:.2f}")
                        cols[1].metric("Subjectivity", f"{sentiment['subjectivity']:.2f}")
                        cols[2].metric("Compound", f"{sentiment['compound']:.2f}")
                        cols[3].metric("Verdict", final_verdict)

                        if ml_results:
                            with st.expander("ML Model Predictions"):
                                for model_name, prediction in ml_results.items():
                                    st.write(f"{model_name}: {prediction}")
                        
                        if enable_bert and bert_result:
                            st.markdown("#### BERT Analysis")
                            b_cols = st.columns(2)
                            b_cols[0].metric("BERT Sentiment", bert_result['label'].title())
                            b_cols[1].metric("Confidence Score", f"{bert_result['score']:.2%}")

                        if enable_lstm and lstm_result:
                            st.markdown("#### LSTM Analysis")
                            l_cols = st.columns(2)
                            l_cols[0].metric("LSTM Sentiment", lstm_result['label'].title())
                            l_cols[1].metric("Confidence Score", f"{lstm_result['score']:.2%}")
                            
                        # Visual comparison
                        comparison_data = []
                        comparison_data.append({'Model': 'TextBlob', 'Score': sentiment['polarity']})
                        comparison_data.append({'Model': 'VADER', 'Score': sentiment['compound']})

                        # Add BERT results if enabled and valid
                        if enable_bert and bert_result and bert_result['label'] != 'ERROR':
                            bert_score = bert_result['score'] if bert_result['label'] == 'POSITIVE' else -bert_result['score']
                            comparison_data.append({
                                'Model': f"BERT ({bert_result['label']})",
                                'Score': bert_score
                            })

                        # Add LSTM results if enabled and valid
                        if enable_lstm and lstm_result and lstm_result.get('error') is None:
                            lstm_score = lstm_result['score'] if lstm_result['label'] == 'POSITIVE' else -lstm_result['score']
                            comparison_data.append({
                                'Model': 'LSTM',
                                'Score': lstm_score
                        })

                        # Display comparison chart if we have any results
                        if comparison_data:
                            comparison_df = pd.DataFrame(comparison_data)
                            fig = px.bar(
                                comparison_df,
                                x='Model',
                                y='Score',
                                color='Model',
                                title="Model Comparison",
                                color_discrete_sequence=px.colors.qualitative.Pastel,
                                labels={'Score': 'Sentiment Score (-1 to 1)'}
                            )
                            st.plotly_chart(fig, use_container_width=True, key='model_comparison')
                        else:
                            st.warning("No valid model results available for comparison")
                        
                        fig = px.bar(
                            x=["Polarity", "Subjectivity", "Compound"],
                            y=[sentiment['polarity'], sentiment['subjectivity'], sentiment['compound']],
                            labels={'x': 'Metric', 'y': 'Value'},
                            color=["Polarity", "Subjectivity", "Compound"],
                            color_continuous_scale='viridis'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        if enable_aspect:
                            st.markdown("#### Aspect Sentiment Analysis")
                            aspect_scores = aspect_sentiment_analysis(review, 'single')
                            if aspect_scores:
                                fig = px.bar(
                                    x=list(aspect_scores.keys()),
                                    y=list(aspect_scores.values()),
                                    labels={'x': 'Aspect', 'y': 'Sentiment Score'},
                                    color=list(aspect_scores.values()),
                                    color_continuous_scale='viridis'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No specific aspects detected in review")
                    
                    with tabs[1]:
                        if entities:
                            df_entities = pd.DataFrame(entities, columns=["Entity", "Type"])
                            st.dataframe(df_entities.style.background_gradient(cmap='viridis'), 
                                        use_container_width=True)
                        else:
                            st.info("No named entities detected")
                        
                        st.markdown("**Top Noun Phrases:**")
                        st.write(noun_phrases[:10])
                    
                    with tabs[2]:
                        doc = nlp(review)
                        stats = {
                            "Word Count": len(doc),
                            "Unique Words": len(set(token.text for token in doc)),
                            "Avg Sentence Length": np.mean([len(sent) for sent in doc.sents]),
                        }
                        for key, value in stats.items():
                            st.markdown(f"**{key}:** `{value}`")
                        
                        pos_counts = Counter([token.pos_ for token in doc])
                        fig = px.pie(
                            names=list(pos_counts.keys()),
                            values=list(pos_counts.values()),
                            title="POS Distribution",
                            hole=0.4
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tabs[3]:
                        if enable_emotion:
                            emotion_scores = emotion_analysis(review)
                            fig = px.bar(
                                x=list(emotion_scores.keys()),
                                y=list(emotion_scores.values()),
                                title="Emotion Distribution",
                                color=list(emotion_scores.values()),
                                color_continuous_scale='viridis'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Enable Emotion Detection in sidebar to view this analysis")
                
                with col2:
                    st.markdown('<div class="section-header">Visualizations</div>', unsafe_allow_html=True)
                    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                    wordcloud = generate_word_cloud(cleaned_text)
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis("off")
                    st.pyplot(plt)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                    fig = px.bar(
                        x=important_terms,
                        y=tfidf.toarray()[0],
                        labels={'x': 'Term', 'y': 'TF-IDF Score'},
                        color=important_terms,
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    if entities:
                        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                        df_entities = pd.DataFrame(entities, columns=["Entity", "Type"])
                        fig = px.sunburst(df_entities, path=['Type', 'Entity'])
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()