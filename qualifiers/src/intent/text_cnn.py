import os
import re
import json
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

INTENT_MAP = {
    0: "COUNT",
    1: "EXIST",
    2: "LIST",
    3: "LOCATION",
    4: "OUT_OF_SCOPE"
}
LABEL_MAP = {v: k for k, v in INTENT_MAP.items()}

import unicodedata

def remove_accents(text):
    text = unicodedata.normalize('NFD', text)
    text = re.sub(r'[\u0300-\u036f]', '', text)
    text = text.replace('đ', 'd').replace('Đ', 'd')
    return unicodedata.normalize('NFC', text)

def clean_and_tokenize(text):
    text = remove_accents(text)
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return [w for w in text.split() if w]

class TextCNNDataset(Dataset):
    def __init__(self, texts, labels=None, vocab=None, max_len=30):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        words = clean_and_tokenize(self.texts[idx])
        tokens = [self.vocab.get(w, self.vocab.get("<UNK>", 1)) for w in words]
        
        # Pad or truncate
        if len(tokens) < self.max_len:
            tokens = tokens + [self.vocab.get("<PAD>", 0)] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]
            
        x = torch.tensor(tokens, dtype=torch.long)
        if self.labels is not None:
            y = torch.tensor(self.labels[idx], dtype=torch.long)
            return x, y
        return x

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, num_filters=16, filter_sizes=(2, 3, 4), num_classes=5, dropout=0.2):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len]
        emb = self.embedding(x).transpose(1, 2)  # [batch_size, embed_dim, seq_len]
        feature_maps = []
        for conv in self.convs:
            act = F.relu(conv(emb))
            pooled = F.max_pool1d(act, kernel_size=act.size(2)).squeeze(2)
            feature_maps.append(pooled)
        out = torch.cat(feature_maps, dim=1)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits

def build_vocab(texts):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for t in texts:
        for w in clean_and_tokenize(t):
            if w not in vocab:
                vocab[w] = len(vocab)
    return vocab

def train_model(train_csv, model_save_path, vocab_save_path, epochs=80, lr=0.005):
    texts, labels = [], []
    with open(train_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["question"])
            labels.append(LABEL_MAP[row["intent"]])

    vocab = build_vocab(texts)
    os.makedirs(os.path.dirname(vocab_save_path), exist_ok=True)
    with open(vocab_save_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    dataset = TextCNNDataset(texts, labels, vocab=vocab)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = TextCNN(vocab_size=len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)

        acc = correct / total if total > 0 else 0.0
        if epoch % 20 == 0 or epoch == epochs:
            print(f"[Epoch {epoch:03d}/{epochs}] Loss: {total_loss/max(total, 1):.4f} | Accuracy: {acc*100:.2f}%")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"[✔] TextCNN trained and saved to: {model_save_path}")
    return model, vocab

def infer_test(test_csv, model_path, vocab_path, output_csv=None):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    model = TextCNN(vocab_size=len(vocab))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    texts = []
    q_ids = []
    with open(test_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q_ids.append(row["question_id"])
            texts.append(row["question"])

    dataset = TextCNNDataset(texts, vocab=vocab)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    predictions = []
    with torch.no_grad():
        for x in loader:
            logits = model(x)
            preds = logits.argmax(dim=1).tolist()
            predictions.extend([INTENT_MAP[p] for p in preds])

    results = []
    for qid, pred in zip(q_ids, predictions):
        results.append({
            "task_type": "INTENT",
            "sample_id": qid,
            "prediction": pred
        })

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["task_type", "sample_id", "prediction"])
            writer.writeheader()
            writer.writerows(results)
        print(f"[✔] Intent predictions saved to: {output_csv}")

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train/Infer TextCNN for Qualifiers")
    parser.add_argument("--train_csv", default="/home/bim/Projects/EAI-2026/qualifiers/data/intent_train.csv")
    parser.add_argument("--test_csv", default="/home/bim/Projects/EAI-2026/qualifiers/data/intent_val.csv")
    parser.add_argument("--model_save", default="/home/bim/Projects/EAI-2026/qualifiers/weights/text_cnn.pth")
    parser.add_argument("--vocab_save", default="/home/bim/Projects/EAI-2026/qualifiers/weights/text_cnn_vocab.json")
    parser.add_argument("--output_csv", default="/home/bim/Projects/EAI-2026/qualifiers/submissions/intent_preds.csv")
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()

    train_model(args.train_csv, args.model_save, args.vocab_save, epochs=args.epochs)
    if os.path.exists(args.test_csv):
        preds = infer_test(args.test_csv, args.model_save, args.vocab_save, args.output_csv)
        print(f"[✔] Inferred {len(preds)} test questions.")
