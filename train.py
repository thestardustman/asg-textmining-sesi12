import os
import re
import json
import urllib.request
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from model_def import SentimentLSTM

# Parameters
DATA_URL = "https://raw.githubusercontent.com/rzyunanda/Text-Mining-Session-12/main/data.csv"
DATA_PATH = "data.csv"
MODEL_PATH = "model.pth"
VOCAB_PATH = "vocab.json"
METRICS_PATH = "metrics.json"

EMBEDDING_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.3
BATCH_SIZE = 64
EPOCHS = 15  # Good balance of speed and convergence
LEARNING_RATE = 1e-3
MAX_LEN = 100
MIN_FREQ = 2

def download_data():
    if not os.path.exists(DATA_PATH):
        print(f"Downloading dataset from {DATA_URL}...")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        try:
            import requests
            response = requests.get(DATA_URL, headers=headers, timeout=30)
            response.raise_for_status()
            with open(DATA_PATH, 'wb') as f:
                f.write(response.content)
            print("Download completed successfully.")
        except Exception as e:
            print(f"Error downloading with requests: {e}")
            print("Attempting download using urllib.request fallback...")
            try:
                import urllib.request
                req = urllib.request.Request(DATA_URL, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as response, open(DATA_PATH, 'wb') as out_file:
                    out_file.write(response.read())
                print("Download completed via urllib fallback.")
            except Exception as e2:
                print(f"Fallback download failed: {e2}")
                raise e2
    else:
        print("Dataset already exists locally.")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Keep only letters and spaces (remove punctuation and numbers)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def build_vocab(texts, min_freq=2):
    print("Building vocabulary...")
    word_counts = {}
    for text in texts:
        tokens = text.split()
        for token in tokens:
            word_counts[token] = word_counts.get(token, 0) + 1
            
    # Filter by min frequency
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True):
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
            
    print(f"Vocabulary size: {len(vocab)} (filtered words with frequency < {min_freq})")
    return vocab

def text_to_sequence(text, vocab, max_len):
    tokens = text.split()
    sequence = []
    for token in tokens:
        sequence.append(vocab.get(token, 1))  # 1 is index for <UNK>
        
    # Pad or truncate
    if len(sequence) < max_len:
        sequence = sequence + [0] * (max_len - len(sequence))  # 0 is index for <PAD>
    else:
        sequence = sequence[:max_len]
    return sequence

class SentimentDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def train_model():
    # 1. Prepare data
    download_data()
    df = pd.read_csv(DATA_PATH)
    
    # Preprocess texts
    print("Preprocessing texts...")
    df['text_clean'] = df['text'].apply(clean_text)
    
    # Split dataset (80% train, 20% validation)
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df['label']
    )
    
    print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")
    
    # Build vocabulary from training set only
    vocab = build_vocab(train_df['text_clean'], min_freq=MIN_FREQ)
    
    # Save vocabulary
    with open(VOCAB_PATH, 'w', encoding='utf-8') as f:
        json.dump({
            "word_to_idx": vocab,
            "max_len": MAX_LEN
        }, f, indent=4, ensure_ascii=False)
    print(f"Vocabulary saved to {VOCAB_PATH}")
    
    # Convert texts to sequences
    train_seqs = [text_to_sequence(t, vocab, MAX_LEN) for t in train_df['text_clean']]
    val_seqs = [text_to_sequence(t, vocab, MAX_LEN) for t in val_df['text_clean']]
    
    train_labels = train_df['label'].values
    val_labels = val_df['label'].values
    
    # Create datasets & loaders
    train_dataset = SentimentDataset(train_seqs, train_labels)
    val_dataset = SentimentDataset(val_seqs, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Setup model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SentimentLSTM(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        bidirectional=BIDIRECTIONAL,
        dropout=DROPOUT
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # 3. Training Loop
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    best_val_acc = 0.0
    
    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0
        
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(seqs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * seqs.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == labels).sum().item()
            total_train += seqs.size(0)
            
        epoch_train_loss = train_loss / total_train
        epoch_train_acc = train_correct / total_train
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        
        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs, labels = seqs.to(device), labels.to(device)
                logits = model(seqs)
                loss = criterion(logits, labels)
                
                val_loss += loss.item() * seqs.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == labels).sum().item()
                total_val += seqs.size(0)
                
        epoch_val_loss = val_loss / total_val
        epoch_val_acc = val_correct / total_val
        
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS:02d} | "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%")
              
        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  --> Saved new best model with Val Acc: {best_val_acc*100:.2f}%")
            
    print(f"\nTraining completed. Best Validation Accuracy: {best_val_acc*100:.2f}%")
    
    # 4. Final Evaluation of Best Model
    print("Evaluating best model...")
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for seqs, labels in val_loader:
            seqs = seqs.to(device)
            logits = model(seqs)
            preds = (torch.sigmoid(logits) >= 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    final_acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    cm = confusion_matrix(all_labels, all_preds)
    
    # Format and save metrics
    metrics = {
        "history": history,
        "eval": {
            "accuracy": float(final_acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix": cm.tolist()
        }
    }
    
    with open(METRICS_PATH, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {METRICS_PATH}")

if __name__ == "__main__":
    train_model()
