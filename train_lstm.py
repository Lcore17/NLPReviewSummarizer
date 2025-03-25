import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import gensim.downloader as api
from collections import Counter
from sklearn.model_selection import train_test_split
import spacy
import joblib
from tqdm import tqdm
import re
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Initialize spaCy
nlp = spacy.load("en_core_web_sm")

# Configuration
SEED = 42
DATA_DIR = "data"
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# In train_lstm.py preprocessing function
def preprocess_text(text):
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
tqdm.pandas()

def load_data():
    """Load and combine datasets with validation"""
    amazon = pd.read_csv(os.path.join(DATA_DIR, "amazon.csv"), encoding='utf-8')
    imdb = pd.read_csv(os.path.join(DATA_DIR, "imdb.csv"), encoding='utf-8')

    imdb['sentiment'] = imdb['sentiment'].map({'positive': 1, 'negative': 0})
    
    amazon['sentiment'] = amazon['category'].map({
        'Shipping': 0, 'Pricing': 1, 'Packaging': 0, 'Service': 0, 'Quality': 1
    })

    combined = pd.concat([
        imdb[['review', 'sentiment']],
        amazon[['review', 'sentiment']]
    ], ignore_index=True)

    combined['sentiment'] = pd.to_numeric(combined['sentiment'], errors='coerce')
    combined = combined.dropna(subset=['sentiment'])
    combined['sentiment'] = combined['sentiment'].astype(int)
    
    combined = combined.dropna(subset=['review'])
    combined = combined[combined['review'].str.strip() != '']
    
    print("Final sentiment distribution:\n", combined['sentiment'].value_counts())
    return combined

combined = load_data()
combined['cleaned'] = combined['review'].progress_apply(preprocess_text)

# Vocabulary with special tokens
VOCAB_SIZE = 20000
MAX_LEN = 256 
SPECIAL_TOKENS = {'<PAD>': 0, '<UNK>': 1}

word_counts = Counter()
for text in tqdm(combined['cleaned']):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    word_counts.update(tokens)

vocab = {word: idx+len(SPECIAL_TOKENS) 
        for idx, (word, _) in enumerate(word_counts.most_common(VOCAB_SIZE))}
vocab.update(SPECIAL_TOKENS)

joblib.dump(vocab, 'tokenizer.pkl')

# Change the sequence processing code to:
MAX_LEN = 128  # This should only be defined once at the top
sequences = []
for text in tqdm(combined['cleaned'], desc="Processing sequences"):
    words = text.split()[:MAX_LEN]  # Explicit truncation
    seq = [vocab.get(word, 1) for word in words]
    seq += [0] * (MAX_LEN - len(seq))  # Pad to exact MAX_LEN
    sequences.append(seq)
    
X = torch.tensor(sequences)
y = torch.tensor(combined['sentiment'].values, dtype=torch.float32)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

class ReviewDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

BATCH_SIZE = 32
train_loader = DataLoader(ReviewDataset(X_train, y_train), 
                        batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(ReviewDataset(X_val, y_val), batch_size=BATCH_SIZE)

# Initialize embeddings
glove = api.load("glove-wiki-gigaword-300")
EMBEDDING_DIM = 300

embedding_matrix = np.zeros((len(vocab), EMBEDDING_DIM))
for word, idx in tqdm(vocab.items(), desc="Initializing embeddings"):
    if word in glove:
        embedding_matrix[idx] = glove[word]
    elif word == '<PAD>':
        embedding_matrix[idx] = np.zeros(EMBEDDING_DIM)
    else:
        embedding_matrix[idx] = np.random.normal(scale=0.6, size=(EMBEDDING_DIM,))

class SentimentLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=256, output_dim=1, 
                n_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix), padding_idx=0)
        self.lstm = nn.LSTM(
            EMBEDDING_DIM, hidden_dim, n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden)

neg_count = (combined['sentiment'] == 0).sum()
pos_count = (combined['sentiment'] == 1).sum()
pos_weight = torch.tensor([neg_count/pos_count], dtype=torch.float).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

model = SentimentLSTM(embedding_matrix).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    model.train()
    train_loss = 0
    for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            outputs = model(batch_X).squeeze()
            val_loss += criterion(outputs, batch_y).item()
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == batch_y).sum().item()
    
    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {train_loss/len(train_loader):.4f}")
    print(f"Val Loss: {val_loss/len(val_loader):.4f}")
    print(f"Val Acc: {correct/len(y_val):.4f}\n")

# Modify the model saving code (LAST FEW LINES)
torch.save({
    'config': {
        'vocab_size': len(vocab),  
        'embedding_dim': 300,
        'hidden_dim': 256,
        'output_dim': 1,
        'n_layers': 2,
        'dropout': 0.5,
        'bidirectional': True,
        'vocab_size': len(vocab)  # Add vocab size
    },
    'model_state_dict': model.state_dict(),
    'embedding_info': {  # Add critical embedding details
        'matrix_shape': embedding_matrix.shape,
        'padding_idx': 0
    }
}, 'lstm_model.pth')

# Save tokenizer properly
joblib.dump({
    'vocab': vocab,
    'unk_idx': 1,
    'pad_idx': 0,
    'max_len': MAX_LEN  # Add max_len parameter
}, 'tokenizer.pkl')

print("Training complete!")