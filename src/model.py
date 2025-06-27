
import torch
import torch.nn as nn

# --- Transformer Model Definition ---
class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, sequence_length):
        super().__init__()
        self.position_embeddings = nn.Embedding(sequence_length, embed_dim)
    
    def forward(self, inputs):
        positions = torch.arange(inputs.size(1), device=inputs.device)
        return inputs + self.position_embeddings(positions)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, embed_dim))
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, inputs):
        attn_output, _ = self.att(inputs, inputs, inputs)
        attn_output = self.dropout(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)

class VentilatorTransformer(nn.Module):
    def __init__(self, num_features, sequence_length, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.feature_embedding = nn.Linear(num_features, embed_dim)
        self.pos_embedding = PositionalEmbedding(embed_dim, sequence_length)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(4)]
        )
        self.final_layer = nn.Linear(embed_dim, 1)

    def forward(self, inputs):
        x = self.feature_embedding(inputs)
        x = self.pos_embedding(x)
        for block in self.transformer_blocks:
            x = block(x)
        return self.final_layer(x)

# --- Optimized LSTM Model Definition ---
class VentilatorLSTM(nn.Module):
    def __init__(self, input_dim, lstm_dim, dense_dim, n_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, lstm_dim, num_layers=n_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(lstm_dim * 2, dense_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_dim, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.head(lstm_out)

