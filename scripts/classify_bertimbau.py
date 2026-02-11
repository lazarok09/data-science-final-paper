# -*- encoding: utf-8 -*-
"""
BERTimbau classificador para empatia, assertividade e tom.

Base de treino 80/20, treino em 80% por teste em 20%, 3 épocas por conta do menor dataset

Métricas já presentes neste script (v1):
  - Acurácia; F1 (weighted + macro); classes preditas.
  - Precisão e recall (weighted); Cohen's kappa; relatório de classificação;
    matriz de confusão.
  - Cohen's kappa (Kappa de Cohen): concordância além do acaso; útil para
    classes desbalanceadas ou ordinais (ex.: níveis de empatia).
  - F1 macro: média do F1 por classe; valoriza todas as classes por igual,
    bom quando há desbalanceamento.
  - Matriz de confusão: mostra quais classes o modelo confunde entre si.
  - Relatório de classificação: precisão/recall por classe.

  Opcionais para o TCC (ainda NÃO presentes neste script):
  - ROC-AUC (one-vs-rest): área sob a curva ROC, uma curva por classe.
    Discriminabilidade por classe; interpretação de probabilidades. Vale a pena
    em dataset maior porque as curvas ROC ficam estáveis e permitem comparar
    melhor o modelo entre classes e entre experimentos.
  - Coeficiente de correlação de Matthews (MCC): métrica única que equilibra
    VP, VN, FP e FN; boa para binário ou visão por alvo. Vale a pena em dataset
    maior porque é menos sensível a desbalanceamento e dá um número único
    fácil de reportar e comparar entre rodadas.
"""

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
)
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)

warnings.filterwarnings("ignore")

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH = SCRIPT_DIR / "baseline_dataset.csv"
BERTIMBAU_MODEL = "neuralmind/bert-base-portuguese-cased"

# Training
RANDOM_STATE = 42
TEST_SIZE = 0.2
EPOCHS = 3
BATCH_SIZE = 8
MAX_LENGTH = 128
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1


def load_and_clean_data(path: Path) -> pd.DataFrame:
    """Load CSV and drop rows with missing labels."""
    df = pd.read_csv(path, encoding="utf-8")
    df = df.rename(columns={c: c.strip() for c in df.columns})
    df["frase"] = df["frase"].astype(str).str.strip()
    # Drop empty rows and rows with missing target values
    df = df.dropna(subset=["empathy", "assertiveness", "tone"])
    df = df[df["frase"].str.len() > 0]
    return df.reset_index(drop=True)


class MultiLabelDataset(Dataset):
    """Dataset that returns (input_ids, attention_mask, empathy_id, assert_id, tone_id)."""

    def __init__(self, texts, empathy_ids, assert_ids, tone_ids, tokenizer, max_length):
        self.texts = list(texts)
        self.empathy_ids = torch.tensor(empathy_ids, dtype=torch.long)
        self.assert_ids = torch.tensor(assert_ids, dtype=torch.long)
        self.tone_ids = torch.tensor(tone_ids, dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "empathy": self.empathy_ids[idx],
            "assertiveness": self.assert_ids[idx],
            "tone": self.tone_ids[idx],
        }


class BERTimbauMultiHead(torch.nn.Module):
    """BERTimbau with three classification heads (empathy, assertiveness, tone)."""

    def __init__(self, model_name, n_empathy, n_assert, n_tone, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.dropout = torch.nn.Dropout(dropout)
        self.empathy_head = torch.nn.Linear(hidden, n_empathy)
        self.assert_head = torch.nn.Linear(hidden, n_assert)
        self.tone_head = torch.nn.Linear(hidden, n_tone)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        cls = self.dropout(cls)
        return self.empathy_head(cls), self.assert_head(cls), self.tone_head(cls)


def train_epoch(model, loader, device, optimizer, scheduler, criterion):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        e = batch["empathy"].to(device)
        a = batch["assertiveness"].to(device)
        t = batch["tone"].to(device)
        logits_e, logits_a, logits_t = model(input_ids, attention_mask)
        loss = criterion(logits_e, e) + criterion(logits_a, a) + criterion(logits_t, t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, label_encoders):
    model.eval()
    all_empathy_true, all_empathy_pred = [], []
    all_assert_true, all_assert_pred = [], []
    all_tone_true, all_tone_pred = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits_e, logits_a, logits_t = model(input_ids, attention_mask)
        all_empathy_true.append(batch["empathy"])
        all_empathy_pred.append(logits_e.argmax(dim=1).cpu())
        all_assert_true.append(batch["assertiveness"])
        all_assert_pred.append(logits_a.argmax(dim=1).cpu())
        all_tone_true.append(batch["tone"])
        all_tone_pred.append(logits_t.argmax(dim=1).cpu())
    y_emp_true = torch.cat(all_empathy_true).numpy()
    y_emp_pred = torch.cat(all_empathy_pred).numpy()
    y_assert_true = torch.cat(all_assert_true).numpy()
    y_assert_pred = torch.cat(all_assert_pred).numpy()
    y_tone_true = torch.cat(all_tone_true).numpy()
    y_tone_pred = torch.cat(all_tone_pred).numpy()
    return (
        (y_emp_true, y_emp_pred),
        (y_assert_true, y_assert_pred),
        (y_tone_true, y_tone_pred),
        label_encoders,
    )


def compute_metrics(y_true, y_pred, name, label_encoder, average="weighted"):
    labels = label_encoder.classes_.tolist()
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    pred_classes = label_encoder.inverse_transform(y_pred)
    return {
        "target": name,
        "accuracy": float(acc),
        "f1_weighted": float(f1),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(prec),
        "recall_weighted": float(rec),
        "cohen_kappa": float(kappa),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "predicted_classes": pred_classes.tolist(),
    }


def main():
    print("Loading data...")
    df = load_and_clean_data(DATASET_PATH)
    print(f"Rows after cleanup: {len(df)}")

    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    le_emp = LabelEncoder()
    le_assert = LabelEncoder()
    le_tone = LabelEncoder()
    df["empathy_id"] = le_emp.fit_transform(df["empathy"])
    df["assertiveness_id"] = le_assert.fit_transform(df["assertiveness"])
    df["tone_id"] = le_tone.fit_transform(df["tone"])
    label_encoders = {"empathy": le_emp, "assertiveness": le_assert, "tone": le_tone}

    # Stratified 80/20 by tone (to keep class balance in test)
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["tone"], random_state=RANDOM_STATE
    )

    print("Loading BERTimbau tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(BERTIMBAU_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTimbauMultiHead(
        BERTIMBAU_MODEL,
        n_empathy=len(le_emp.classes_),
        n_assert=len(le_assert.classes_),
        n_tone=len(le_tone.classes_),
    ).to(device)

    train_ds = MultiLabelDataset(
        train_df["frase"].tolist(),
        train_df["empathy_id"].tolist(),
        train_df["assertiveness_id"].tolist(),
        train_df["tone_id"].tolist(),
        tokenizer,
        MAX_LENGTH,
    )
    test_ds = MultiLabelDataset(
        test_df["frase"].tolist(),
        test_df["empathy_id"].tolist(),
        test_df["assertiveness_id"].tolist(),
        test_df["tone_id"].tolist(),
        tokenizer,
        MAX_LENGTH,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(WARMUP_RATIO * total_steps), num_training_steps=total_steps
    )
    criterion = torch.nn.CrossEntropyLoss()

    print(f"Training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        loss = train_epoch(model, train_loader, device, optimizer, scheduler, criterion)
        print(f"  Epoch {epoch + 1}/{EPOCHS} — train loss: {loss:.4f}")

    print("Evaluating on test set (20%)...")
    (emp_true, emp_pred), (assert_true, assert_pred), (tone_true, tone_pred), _ = evaluate(
        model, test_loader, device, label_encoders
    )

    results = {}
    for (y_true, y_pred), name, le in [
        ((emp_true, emp_pred), "empathy", le_emp),
        ((assert_true, assert_pred), "assertiveness", le_assert),
        ((tone_true, tone_pred), "tone", le_tone),
    ]:
        results[name] = compute_metrics(y_true, y_pred, name, le)

    # Console summary
    print("\n" + "=" * 60)
    print("METRICS (test set, 20% holdout)")
    print("=" * 60)
    for name, m in results.items():
        print(f"\n--- {name.upper()} ---")
        print(f"  Accuracy:       {m['accuracy']:.4f}")
        print(f"  F1 (weighted):  {m['f1_weighted']:.4f}")
        print(f"  F1 (macro):     {m['f1_macro']:.4f}")
        print(f"  Precision:      {m['precision_weighted']:.4f}")
        print(f"  Recall:         {m['recall_weighted']:.4f}")
        print(f"  Cohen's kappa:  {m['cohen_kappa']:.4f}")
        print(f"  Predicted classes: {m['predicted_classes']}")
        print("  Classification report:")
        print(m["classification_report"])
        print("  Confusion matrix:")
        print(np.array(m["confusion_matrix"]))

    # Save metrics JSON (without long report for readability; include report in file)
    out_path = SCRIPT_DIR / "classification_metrics.json"
    save_results = {
        k: {
            "accuracy": v["accuracy"],
            "f1_weighted": v["f1_weighted"],
            "f1_macro": v["f1_macro"],
            "precision_weighted": v["precision_weighted"],
            "recall_weighted": v["recall_weighted"],
            "cohen_kappa": v["cohen_kappa"],
            "predicted_classes": v["predicted_classes"],
            "classification_report": v["classification_report"],
            "confusion_matrix": v["confusion_matrix"],
        }
        for k, v in results.items()
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_results, f, ensure_ascii=False, indent=2)
    print(f"\nMetrics saved to {out_path}")

    return results


if __name__ == "__main__":
    main()
