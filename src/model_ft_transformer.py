"""
model_ft_transformer.py

Feature Tokenizer + Transformer (FT-Transformer) for ICU mortality prediction.
Reference: Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data" (2021).
Library: rtdl (https://github.com/Yura52/rtdl)

Architecture:
  - Every feature (numerical AND categorical) is independently tokenized into
    a d-dimensional embedding via a linear projection (+ bias for numerical,
    embedding lookup for categorical).
  - The resulting sequence of tokens is passed through L Transformer layers
    (multi-head self-attention + FFN with pre-norm residual connections).
  - A [CLS] token aggregates information; its final representation is passed
    to a linear classification head.

This is architecturally distinct from both LogReg (linear) and XGBoost (trees):
it captures dense global pairwise feature interactions via attention.

Hyperparameter tuning via manual grid (Optuna would be better for production
but adds a heavy dependency; manual grid is sufficient for comparison purposes).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
import math
import warnings

from preprocessing import FTTransformerPreprocessor, labels_to_binary


# ── FT-Transformer implementation (pure PyTorch, no rtdl dependency) ─────────
# rtdl's API changes frequently; a clean self-contained implementation is
# more robust for a standalone project.

class FeatureTokenizer(nn.Module):
    """
    Tokenizes all features into a shared d-dimensional embedding space.
      - Numerical features: linear projection x_i * w_i + b_i  (per-feature weights)
      - Categorical features: embedding lookup table
    """
    def __init__(self, n_num: int, cat_cardinalities: list[int], d_token: int):
        super().__init__()
        self.d_token = d_token

        # Numerical tokenizer: (n_num, d_token) weight + (n_num, d_token) bias
        if n_num > 0:
            self.num_weight = nn.Parameter(torch.empty(n_num, d_token))
            self.num_bias   = nn.Parameter(torch.zeros(n_num, d_token))
            nn.init.kaiming_uniform_(self.num_weight, a=math.sqrt(5))
        else:
            self.num_weight = None

        # Categorical tokenizer: one embedding per categorical feature
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card, d_token) for card in cat_cardinalities
        ]) if cat_cardinalities else nn.ModuleList()

        # [CLS] token
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor = None):
        tokens = []

        if self.num_weight is not None:
            # x_num: (B, n_num) → (B, n_num, d_token)
            num_tokens = x_num.unsqueeze(-1) * self.num_weight.unsqueeze(0) + self.num_bias.unsqueeze(0)
            tokens.append(num_tokens)

        for i, emb in enumerate(self.cat_embeddings):
            tokens.append(emb(x_cat[:, i]).unsqueeze(1))

        tokens = torch.cat(tokens, dim=1)  # (B, n_features, d_token)

        # Prepend [CLS]
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, n_features+1, d_token)
        return tokens


class TransformerBlock(nn.Module):
    def __init__(self, d_token: int, n_heads: int, d_ffn: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn  = nn.MultiheadAttention(d_token, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_token)
        self.ffn   = nn.Sequential(
            nn.Linear(d_token, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_token),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pre-norm attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        return x


class FTTransformerNet(nn.Module):
    def __init__(
        self,
        n_num: int,
        cat_cardinalities: list[int],
        d_token: int = 192,
        n_layers: int = 3,
        n_heads: int = 8,
        d_ffn_factor: float = 4/3,
        dropout: float = 0.0,
        n_classes: int = 2,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_num, cat_cardinalities, d_token)
        d_ffn = int(d_token * d_ffn_factor)

        self.transformer = nn.Sequential(*[
            TransformerBlock(d_token, n_heads, d_ffn, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)   # binary → logit

    def forward(self, x_num, x_cat=None):
        tokens = self.tokenizer(x_num, x_cat)   # (B, S, d)
        tokens = self.transformer(tokens)
        cls_out = self.norm(tokens[:, 0])        # [CLS] representation
        return self.head(cls_out).squeeze(-1)    # (B,)


# ── sklearn-compatible wrapper ────────────────────────────────────────────────

class FTTransformerClassifier(BaseEstimator, ClassifierMixin):
    """
    sklearn-compatible wrapper around FTTransformerNet.
    Handles its own preprocessing internally via FTTransformerPreprocessor.
    """

    def __init__(
        self,
        d_token: int = 192,
        n_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.0,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        n_epochs: int = 50,
        batch_size: int = 256,
        patience: int = 10,
        device: str = None,
        pos_weight: float = 1.0,
    ):
        self.d_token      = d_token
        self.n_layers     = n_layers
        self.n_heads      = n_heads
        self.dropout      = dropout
        self.lr           = lr
        self.weight_decay = weight_decay
        self.n_epochs     = n_epochs
        self.batch_size   = batch_size
        self.patience     = patience
        self.device       = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pos_weight   = pos_weight

        self.preprocessor_ = None
        self.model_        = None
        self.classes_      = np.array([0, 1])

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_val=None, y_val=None):
        # Preprocessing
        self.preprocessor_ = FTTransformerPreprocessor()
        self.preprocessor_.fit(X)
        data      = self.preprocessor_.transform(X)
        x_num     = torch.tensor(data["x_num"], dtype=torch.float32).to(self.device)
        x_cat     = torch.tensor(data["x_cat"], dtype=torch.long).to(self.device) if data["x_cat"] is not None else None
        y_tensor  = torch.tensor(y, dtype=torch.float32).to(self.device)

        cat_cards = self.preprocessor_.get_cardinalities(X)
        n_num     = x_num.shape[1]

        self.model_ = FTTransformerNet(
            n_num=n_num,
            cat_cardinalities=cat_cards,
            d_token=self.d_token,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            dropout=self.dropout,
        ).to(self.device)

        optimizer  = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs)
        pos_weight = torch.tensor([self.pos_weight], dtype=torch.float32).to(self.device)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Build DataLoader
        dataset = TensorDataset(x_num, *([x_cat] if x_cat is not None else []), y_tensor)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Validation setup
        do_val = X_val is not None and y_val is not None
        if do_val:
            val_data  = self.preprocessor_.transform(X_val)
            x_num_v   = torch.tensor(val_data["x_num"], dtype=torch.float32).to(self.device)
            x_cat_v   = torch.tensor(val_data["x_cat"], dtype=torch.long).to(self.device) if val_data["x_cat"] is not None else None
            y_val_t   = torch.tensor(y_val, dtype=torch.float32).to(self.device)
            best_val  = np.inf
            wait      = 0

        print(f"\n[FT-Transformer] Training on {self.device} | {n_num} numerical + {len(cat_cards)} categorical features")

        for epoch in range(self.n_epochs):
            self.model_.train()
            total_loss = 0.0
            for batch in loader:
                if x_cat is not None:
                    xn, xc, yb = batch
                else:
                    xn, yb = batch
                    xc = None

                optimizer.zero_grad()
                logits = self.model_(xn, xc)
                loss   = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()

            if do_val and (epoch + 1) % 5 == 0:
                self.model_.eval()
                with torch.no_grad():
                    val_logits = self.model_(x_num_v, x_cat_v)
                    val_loss   = criterion(val_logits, y_val_t).item()
                print(f"  Epoch {epoch+1:3d} | train_loss={total_loss/len(loader):.4f} | val_loss={val_loss:.4f}")
                if val_loss < best_val:
                    best_val = val_loss
                    wait = 0
                    self._best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
                else:
                    wait += 1
                    if wait >= self.patience:
                        print(f"  Early stopping at epoch {epoch+1}")
                        self.model_.load_state_dict(self._best_state)
                        break
            elif (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d} | train_loss={total_loss/len(loader):.4f}")

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.model_.eval()
        data  = self.preprocessor_.transform(X)
        x_num = torch.tensor(data["x_num"], dtype=torch.float32).to(self.device)
        x_cat = torch.tensor(data["x_cat"], dtype=torch.long).to(self.device) if data["x_cat"] is not None else None
        with torch.no_grad():
            logits = self.model_(x_num, x_cat)
            probs  = torch.sigmoid(logits).cpu().numpy()
        return np.column_stack([1 - probs, probs])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ── Hyperparameter tuning ─────────────────────────────────────────────────────

# Compact grid: vary the most impactful parameters
PARAM_GRID = [
    {"d_token": 64,  "n_layers": 2, "dropout": 0.0, "lr": 3e-4},
    {"d_token": 128, "n_layers": 3, "dropout": 0.1, "lr": 1e-4},
    {"d_token": 192, "n_layers": 3, "dropout": 0.2, "lr": 1e-4},
    {"d_token": 128, "n_layers": 2, "dropout": 0.0, "lr": 3e-4},
]


def train(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame = None,
    y_val: np.ndarray = None,
    tune: bool = True,
    n_epochs: int = 50,
    batch_size: int = 256,
) -> tuple[FTTransformerClassifier, dict]:
    """
    Train FT-Transformer. If tune=True, runs a small hyperparameter grid
    using a single held-out validation split (80/20 stratified).
    """
    # Compute class imbalance weight
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = n_neg / max(n_pos, 1)
    print(f"\n[FT-Transformer] pos_weight = {pos_weight:.2f}")

    if tune:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score

        X_tr, X_v, y_tr, y_v = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
        )

        best_auroc  = -np.inf
        best_params = PARAM_GRID[0]

        for params in PARAM_GRID:
            print(f"\n[FT-Transformer] Trying: {params}")
            clf = FTTransformerClassifier(
                **params,
                n_epochs=n_epochs,
                batch_size=batch_size,
                pos_weight=pos_weight,
                patience=8,
            )
            clf.fit(X_tr, y_tr, X_val=X_v, y_val=y_v)
            probs = clf.predict_proba(X_v)[:, 1]
            auroc = roc_auc_score(y_v, probs)
            print(f"  → Val AUROC: {auroc:.4f}")

            if auroc > best_auroc:
                best_auroc  = auroc
                best_params = params

        print(f"\n[FT-Transformer] Best params: {best_params} | Val AUROC: {best_auroc:.4f}")

        # Refit on full training data with best params
        model = FTTransformerClassifier(
            **best_params,
            n_epochs=n_epochs,
            batch_size=batch_size,
            pos_weight=pos_weight,
            patience=10,
        )
        model.fit(X_train, y_train)

    else:
        model = FTTransformerClassifier(
            d_token=128, n_layers=3, dropout=0.1, lr=1e-4,
            n_epochs=n_epochs, batch_size=batch_size, pos_weight=pos_weight,
        )
        model.fit(X_train, y_train)
        best_params = {"d_token": 128, "n_layers": 3}

    meta = {
        "model_name": "FT-Transformer",
        "best_params": best_params,
    }
    return model, meta