# GoEmotions Multi-Label Classification with BERT (GPU-Optimized)

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt





# Hyperparameters
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 4
LR = 2e-5
NUM_LABELS = 28

class GoEmotionsDataset(torch.utils.data.Dataset):
    def __init__(self, split="train"):
        data = load_dataset("go_emotions")['train']
        if split == "train":
            self.samples = data.select(range(0, int(0.9 * len(data))))
        else:
            self.samples = data.select(range(int(0.9 * len(data)), len(data)))
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        encoding = self.tokenizer(
            item["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        labels = torch.zeros(NUM_LABELS)
        for l in item["labels"]:
            labels[l] = 1.0
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels
        }

class BertMultiLabelClassifier(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        out = self.dropout(pooled)
        return self.classifier(out)

def compute_metrics(preds, labels, threshold=0.5):
    preds_bin = (preds >= threshold).astype(int)
    return {
        "micro_f1": f1_score(labels, preds_bin, average="micro"),
        "macro_f1": f1_score(labels, preds_bin, average="macro")
    }

# Load Datasets 
train_dataset = GoEmotionsDataset("train")
test_dataset = GoEmotionsDataset("test")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Initialize 
model = BertMultiLabelClassifier(NUM_LABELS).to(device)
optimizer = AdamW(model.parameters(), lr=LR)
loss_fn = nn.BCEWithLogitsLoss()

# Training Metrics Tracker
f1_history = {
    "epoch": [],
    "micro_f1": [],
    "macro_f1": [],
    "train_loss": []
}



# Training 
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    for batch in tqdm(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Training Loss: {avg_loss:.4f}")


    
    # Evaluate
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.sigmoid(outputs).cpu().numpy()
            labels = labels.cpu().numpy()
            preds_all.append(preds)
            labels_all.append(labels)

    preds_np = np.vstack(preds_all)
    labels_np = np.vstack(labels_all)
    metrics = compute_metrics(preds_np, labels_np)
    print(f"Micro-F1: {metrics['micro_f1']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}")

    # Log
    f1_history["epoch"].append(epoch + 1)
    f1_history["micro_f1"].append(metrics['micro_f1'])
    f1_history["macro_f1"].append(metrics['macro_f1'])
    f1_history["train_loss"].append(avg_loss)

    # Save checkpoint coz I'm scared :(
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'f1_history': f1_history
    }, f"checkpoint_epoch_{epoch+1}.pt")

# Save final model coz I'm scared always :((
torch.save(model.state_dict(), "goemotions_bert_model.pt")
print("Model saved to goemotions_bert_model.pt")




# Plot F1 :)
df = pd.DataFrame(f1_history)
df.to_csv("f1_scores.csv", index=False)
print("F1 scores saved to f1_scores.csv")

plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["micro_f1"], label="Micro-F1", marker='o', color='green')
plt.plot(df["epoch"], df["macro_f1"], label="Macro-F1", marker='x', color='orange')
plt.title("F1 Scores over Epochs")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("f1_plot.png")
plt.show()
