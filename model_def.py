import torch
import torch.nn as nn

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_layers=2, bidirectional=True, dropout=0.3):
        super(SentimentLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def forward(self, text, seq_lengths=None):
        # text shape: [batch_size, seq_len]
        embedded = self.embedding(text)  # shape: [batch_size, seq_len, embedding_dim]
        
        # Pass through LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out shape: [batch_size, seq_len, lstm_output_dim]
        
        # We perform Global Average Pooling over the sequence length dimension (dim=1)
        # to get a single representation for the whole sentence.
        # This is robust to variable padding.
        pooled = torch.mean(lstm_out, dim=1)  # shape: [batch_size, lstm_output_dim]
        
        # Pass to fully connected classifier
        logits = self.fc(pooled)  # shape: [batch_size, 1]
        return logits.squeeze(1)  # shape: [batch_size]
