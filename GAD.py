import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os

class SelfAttentionLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout_rate=0.1):
        super(SelfAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout_rate)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, embedding_dim)
        attn_output, _ = self.multihead_attn(x, x, x)
        attn_output = self.dropout(attn_output)
        x = self.layer_norm(x + attn_output)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, embedding_dim)
        return x

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=30, num_heads=2, rnn_hidden_size=64, num_ffn_layers=3, ffn_hidden_size=128):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Self-attention layers
        self.self_attn1 = SelfAttentionLayer(embedding_dim, num_heads)
        self.self_attn2 = SelfAttentionLayer(embedding_dim, num_heads)
        
        # RNN layer
        self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=rnn_hidden_size, batch_first=True)
        
        # Feedforward layers
        ffn_layers = []
        for _ in range(num_ffn_layers):
            ffn_layers.append(nn.Linear(rnn_hidden_size, ffn_hidden_size))
            ffn_layers.append(nn.ReLU())
            rnn_hidden_size = ffn_hidden_size
        self.ffn_layers = nn.Sequential(*ffn_layers)
        
        # Output layer
        self.output_layer = nn.Linear(ffn_hidden_size, vocab_size)
        
    def forward(self, x, hidden=None):
        # Input x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # Self-attention layers
        x = self.self_attn1(embedded)
        x = self.self_attn2(x)
        
        # RNN layer
        rnn_out, hidden = self.rnn(x, hidden)
        
        # Feedforward layers
        rnn_out = rnn_out[:, -1, :]  # Take the last time step output
        rnn_out = self.ffn_layers(rnn_out)
        
        # Output layer
        output = self.output_layer(rnn_out)
        
        return output, hidden

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def build_vocab(text, max_vocab_size=10000):
    words = text.split()
    word_freq = {}
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
    sorted_vocab = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    vocab = {word: idx for idx, (word, _) in enumerate(sorted_vocab[:max_vocab_size])}
    vocab["<UNK>"] = len(vocab)
    return vocab

def text_to_tensor(text, vocab):
    words = text.split()
    tensor = torch.tensor([vocab.get(word, vocab["<UNK>"]) for word in words], dtype=torch.long)
    return tensor

def batchify(data, batch_size, seq_len):
    num_batches = data.size(0) // (batch_size * seq_len)
    data = data[:num_batches * batch_size * seq_len]
    data = data.view(batch_size, -1)
    for i in range(0, data.size(1), seq_len):
        yield data[:, i:i + seq_len]

def generate_text(model, start_text, vocab, idx_to_word, max_len=100, temperature=1.0):
    model.eval()  # Mettre le modèle en mode évaluation (désactive le dropout)
    
    input_text = torch.tensor([[vocab.get(start_text, vocab["<UNK>"])]], dtype=torch.long)  # Convertir en tensor avec une dimension de lot de 1
    hidden = None
    text_generated = [start_text]

    with torch.no_grad():
        for _ in range(max_len):
            output, hidden = model(input_text, hidden)
            logits = output.squeeze(0) / temperature
            probabilities = F.softmax(logits, dim=-1)
            
            # Échantillonner un token en fonction des probabilités
            predicted_id = torch.multinomial(probabilities, num_samples=1)
            
            text_generated.append(idx_to_word[predicted_id.item()])
            
            # Préparer le prochain input pour le modèle
            input_text = predicted_id.unsqueeze(0)
        
    return ' '.join(text_generated)

if __name__ == "__main__":
    # Charger les données textuelles
    file_path = "data.txt"  # Remplacez par le chemin de votre fichier texte
    text = load_data(file_path)
    
    # Construire le vocabulaire
    vocab = build_vocab(text)
    idx_to_word = {idx: word for word, idx in vocab.items()}
    vocab_size = len(vocab)
    
    # Convertir le texte en tensor
    data = text_to_tensor(text, vocab)
    
    # Définir les hyperparamètres
    embedding_dim = 30
    num_heads = 2
    rnn_hidden_size = 64
    num_ffn_layers = 3
    ffn_hidden_size = 128
    batch_size = 32
    seq_len = 20
    num_epochs = 20
    learning_rate = 0.001
    
    # Créer une instance du modèle
    model = LanguageModel(vocab_size, embedding_dim, num_heads, rnn_hidden_size, num_ffn_layers, ffn_hidden_size)
    
    # Définir l'optimiseur et la fonction de perte
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Entraîner le modèle
    model.train()
    for epoch in range(num_epochs):
        batches = batchify(data, batch_size, seq_len)
        total_loss = 0
        for batch in batches:
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets.contiguous().view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(data)}')

    # Générer du texte
    start_text = "il"  # Mot de départ pour la génération de texte, remplacez par un mot approprié
    generated_text = generate_text(model, start_text, vocab, idx_to_word)
    
    print(generated_text)
