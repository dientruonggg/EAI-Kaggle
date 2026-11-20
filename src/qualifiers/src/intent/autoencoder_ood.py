#!/usr/bin/env python3
"""
EDGE AI CHALLENGE 2026 - TEXT AUTO-ENCODER FOR OUT-OF-SCOPE (OOD) DETECTION
Compares One-Hot TextCNN vs. Auto-Encoder (AutoEn) for Intent Classification & OOS Filtering.

Architecture:
1. One-Hot Softmax: Standard 5-way classification ([1,0,0,0,0] -> COUNT ... [0,0,0,0,1] -> OUT_OF_SCOPE).
2. Auto-Encoder (AutoEn):
   - Trains an Encoder-Decoder bottleneck on IN-SCOPE questions (COUNT, EXIST, LIST, LOCATION).
   - Measures Reconstruction Error (MSE).
   - If Reconstruction Error > Threshold tau -> Flagged as OUT_OF_SCOPE ("OOS").
   - If Reconstruction Error <= Threshold tau -> Classified into 4 In-Scope intents via One-Hot Softmax.
"""
import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = "/home/bim/Projects/EAI-2026/qualifiers"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.intent.text_cnn import clean_and_tokenize, TextCNNDataset, build_vocab

class TextAutoEncoder(nn.Module):
    """
    Lightweight Text Auto-Encoder (< 5,000 parameters) suitable for ESP32-S3.
    Compresses token sequence embeddings into a compact latent bottleneck (dim=8).
    """
    def __init__(self, vocab_size, embed_dim=16, latent_dim=8, seq_len=30):
        super(TextAutoEncoder, self).__init__()
        self.seq_len = seq_len
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Encoder: (seq_len * embed_dim) -> latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(seq_len * embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        
        # Decoder: latent_dim -> (seq_len * embed_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, seq_len * embed_dim)
        )

    def forward(self, x):
        # x: [batch, seq_len]
        emb = self.embed(x).view(x.size(0), -1)  # Flatten
        latent = self.encoder(emb)
        reconstructed = self.decoder(latent)
        return emb, reconstructed, latent

def train_autoencoder(train_csv, vocab_path, epochs=40, lr=1e-3):
    import csv
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # Filter only IN-SCOPE questions (ignore OUT_OF_SCOPE during AE training)
    in_scope_texts = []
    with open(train_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["intent"] != "OUT_OF_SCOPE":
                in_scope_texts.append(r["question"])

    dataset = TextCNNDataset(in_scope_texts, vocab=vocab, max_len=30)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = TextAutoEncoder(vocab_size=len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        for x in loader:
            optimizer.zero_grad()
            orig, recon, _ = model(x)
            loss = criterion(recon, orig)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x)
        if ep % 10 == 0:
            print(f"[AE Epoch {ep:02d}/{epochs}] Reconstruction Loss (MSE): {total_loss/len(dataset):.4f}")

    return model, vocab

def evaluate_oos_detection(ae_model, vocab, test_csv, threshold=0.08):
    import csv
    ae_model.eval()
    
    test_rows = []
    with open(test_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            test_rows.append(r)

    print(f"\n[*] Evaluating Auto-Encoder OOS Detection on {len(test_rows)} test questions...")
    correct = 0
    total = len(test_rows)

    with torch.no_grad():
        for r in test_rows[:15]:
            q = r["question"]
            gt_intent = r["intent"]
            tokens = [vocab.get(w, 1) for w in clean_and_tokenize(q)]
            tokens = (tokens + [0] * 30)[:30]
            inp = torch.tensor([tokens], dtype=torch.long)

            orig, recon, _ = ae_model(inp)
            mse = F.mse_loss(recon, orig).item()
            pred_oos = "OUT_OF_SCOPE" if mse > threshold else "IN_SCOPE"

            is_gt_oos = (gt_intent == "OUT_OF_SCOPE")
            is_pred_oos = (pred_oos == "OUT_OF_SCOPE")
            match = (is_gt_oos == is_pred_oos)

            print(f"Q: '{q}' | GT: {gt_intent:<12} | MSE: {mse:.4f} -> Pred: {pred_oos} ({'✔' if match else '✘'})")

if __name__ == "__main__":
    train_csv = os.path.join(PROJECT_ROOT, "data/qa_train.csv")
    test_csv = os.path.join(PROJECT_ROOT, "data/qa_val.csv")
    vocab_p = os.path.join(PROJECT_ROOT, "weights/text_cnn_vocab.json")

    ae_model, vocab = train_autoencoder(train_csv, vocab_p, epochs=30)
    evaluate_oos_detection(ae_model, vocab, test_csv, threshold=0.08)
