import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout_rate=0.1):
        super(SelfAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout_rate)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, embedding_dim)
        attn_output, _ = self.multihead_attn(x, x, x)
        attn_output = self.dropout(attn_output)
        x = self.layer_norm(x + attn_output)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, embedding_dim)
        return x

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=30, num_heads=2, rnn_hidden_size=64, num_ffn_layers=3, ffn_hidden_size=128, output_size=None):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Linear(vocab_size, embedding_dim)
        
        # Self-attention layers
        self.self_attn1 = SelfAttentionLayer(embedding_dim, num_heads)
        self.self_attn2 = SelfAttentionLayer(embedding_dim, num_heads)
        
        # RNN layer
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=rnn_hidden_size, batch_first=True)
        
        # Feedforward layers
        self.ffn_layers = nn.Sequential(
            nn.Linear(rnn_hidden_size, ffn_hidden_size),
            nn.ReLU(),
            nn.Linear(ffn_hidden_size, ffn_hidden_size),
            nn.ReLU(),
            nn.Linear(ffn_hidden_size, ffn_hidden_size),
            nn.ReLU()
        )
        
        # Output layer
        self.output_layer = nn.Linear(ffn_hidden_size, output_size if output_size else vocab_size)
        
    def forward(self, x):
        # Input x shape: (batch_size, seq_len, vocab_size)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # Self-attention layers
        x = self.self_attn1(embedded)
        x = self.self_attn2(x)
        
        # RNN layer
        rnn_out, _ = self.rnn(x)
        
        # Feedforward layers
        rnn_out = rnn_out.mean(dim=1)  # Average pooling over the sequence length
        rnn_out = self.ffn_layers(rnn_out)
        
        # Output layer
        output = self.output_layer(rnn_out)
        
        return output

# Exemple d'utilisation du modèle
if __name__ == "__main__":
    # Définir la taille du vocabulaire et autres hyperparamètres
    vocab_size = 10000  # Taille fictive du vocabulaire, remplacez par la taille réelle
    embedding_dim = 30
    num_heads = 2
    rnn_hidden_size = 64
    num_ffn_layers = 3
    ffn_hidden_size = 128
    output_size = None  # Ici, None signifie que la sortie est la taille du vocabulaire
    
    # Créer une instance du modèle
    model = LanguageModel(vocab_size, embedding_dim, num_heads, rnn_hidden_size, num_ffn_layers, ffn_hidden_size, output_size)
    
    # Exemple de tensor d'entrée
    batch_size = 32
    seq_len = 20
    input_tensor = torch.randn(batch_size, seq_len, vocab_size)  # Tensor d'exemple avec des valeurs aléatoires
    
    # Passe en avant
    output = model(input_tensor)
    print("Output shape:", output.shape)  # Affiche la forme de la sortie
