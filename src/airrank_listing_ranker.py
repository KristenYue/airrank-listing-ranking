import os
import re
import json
import zipfile
import random
import logging
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ===================== 1.Fix random seed =====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


# ===================== 2. configuration =====================
class Config:
    # Use repository-relative defaults so the project works on other computers.
    # Both paths can be overridden without editing the source code:
    #   AIRRANK_DATASET_ZIP=/path/to/datasets.zip
    #   AIRRANK_OUTPUT_DIR=/path/to/output
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ZIP_PATH = os.getenv(
        "AIRRANK_DATASET_ZIP",
        os.path.join(PROJECT_ROOT, "data", "datasets.zip")
    )
    INNER_FILE = "step1_review_dataset.jsonl"

    OUTPUT_DIR = os.getenv(
        "AIRRANK_OUTPUT_DIR",
        os.path.join(PROJECT_ROOT, "outputs")
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Classify by listing_id to prevent the leakage of the same property information into training and testing
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15

    # Text parameters
    MAX_VOCAB_SIZE = 20000
    MIN_FREQ = 2
    MAX_LEN = 180

    # model parameter：CNN + BiLSTM
    EMBED_DIM = 128
    HIDDEN_DIM = 128
    NUM_FILTERS = 64
    KERNEL_SIZES = [3, 4, 5]
    DROPOUT = 0.35

    # Training Parameters
    BATCH_SIZE = 32
    EPOCHS = 12
    LR = 1e-3
    PATIENCE = 3

    # Ranking evaluation
    TOP_K = 10

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===================== 3. log =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(Config.OUTPUT_DIR, "training.log"), encoding="utf-8"),
        logging.StreamHandler()
    ]
)


# ===================== 4. text processing =====================
def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    return text.split()


# ===================== 5. reading data =====================
def load_review_data(zip_path, inner_file):
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(
            "Dataset archive not found: "
            f"{zip_path}\n"
            "Place datasets.zip in the repository data directory or set "
            "the AIRRANK_DATASET_ZIP environment variable."
        )

    rows = []
    with zipfile.ZipFile(zip_path, "r") as z:
        if inner_file not in z.namelist():
            raise FileNotFoundError(
                f"{inner_file} was not found inside {zip_path}."
            )
        with z.open(inner_file) as f:
            for line in f:
                rows.append(json.loads(line.decode("utf-8")))

    df = pd.DataFrame(rows)

    required_cols = ["listing_id", "review_id", "review_text", "rating", "date"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"lack field: {col}")

    df["review_text"] = df["review_text"].fillna("").apply(clean_text)
    df = df[df["review_text"].str.len() > 0].reset_index(drop=True)

    # Ensure that the rating is float
    df["rating"] = df["rating"].astype(float)

    return df


# ===================== 6. Classified by listing_id =====================
def split_by_listing(df, test_size=0.15, val_size=0.15, random_state=42):
    listing_ids = df["listing_id"].unique().tolist()

    train_val_ids, test_ids = train_test_split(
        listing_ids,
        test_size=test_size,
        random_state=random_state
    )

    val_ratio_in_train_val = val_size / (1 - test_size)

    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_ratio_in_train_val,
        random_state=random_state
    )

    train_df = df[df["listing_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["listing_id"].isin(val_ids)].reset_index(drop=True)
    test_df = df[df["listing_id"].isin(test_ids)].reset_index(drop=True)

    return train_df, val_df, test_df


# ===================== 7.Build a vocabulary list =====================
def build_vocab(texts, max_vocab_size=20000, min_freq=2):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {"<pad>": 0, "<unk>": 1}

    for word, freq in counter.most_common():
        if freq < min_freq:
            continue
        if len(vocab) >= max_vocab_size:
            break
        vocab[word] = len(vocab)

    return vocab


def encode_text(text, vocab, max_len):
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokenize(text)]
    ids = ids[:max_len]
    mask = [1] * len(ids)

    if len(ids) < max_len:
        pad_len = max_len - len(ids)
        ids += [vocab["<pad>"]] * pad_len
        mask += [0] * pad_len

    return np.array(ids, dtype=np.int64), np.array(mask, dtype=np.int64)


# ===================== 8. Dataset =====================
class ReviewRegressionDataset(Dataset):
    def __init__(self, df, vocab, max_len):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        input_ids, attention_mask = encode_text(row["review_text"], self.vocab, self.max_len)

        # rating: 1~5 -> Normalize to 0 to 1
        target_norm = (float(row["rating"]) - 1.0) / 4.0

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "target_norm": torch.tensor(target_norm, dtype=torch.float32),
            "true_rating": torch.tensor(float(row["rating"]), dtype=torch.float32),
            "listing_id": str(row["listing_id"]),
            "review_id": str(row["review_id"]),
            "review_text": row["review_text"]
        }


# ===================== 9. Model：Model: TextCNN + BiLSTM regression =====================
class TextCNNBiLSTMRegressor(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 num_filters=64, kernel_sizes=(3, 4, 5), dropout=0.35):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=hidden_dim * 2,
                out_channels=num_filters,
                kernel_size=k
            ) for k in kernel_sizes
        ])

        feature_dim = num_filters * len(kernel_sizes) + (hidden_dim * 2) * 2

        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None):
        emb = self.embedding(input_ids)                 # [B, L, E]
        lstm_out, _ = self.lstm(emb)                    # [B, L, 2H]

        x = lstm_out.transpose(1, 2)                    # [B, 2H, L]
        conv_features = []
        for conv in self.convs:
            c = torch.relu(conv(x))                     # [B, F, L-k+1]
            p = torch.max(c, dim=2).values              # [B, F]
            conv_features.append(p)

        cnn_feat = torch.cat(conv_features, dim=1)      # [B, F*len(kernels)]

        avg_pool = torch.mean(lstm_out, dim=1)          # [B, 2H]
        max_pool = torch.max(lstm_out, dim=1).values    # [B, 2H]

        feat = torch.cat([cnn_feat, avg_pool, max_pool], dim=1)
        feat = self.dropout(feat)

        raw_out = self.regressor(feat).squeeze(1)       # [B]
        pred_norm = torch.sigmoid(raw_out)              # [0,1]
        pred_rating = 1.0 + 4.0 * pred_norm             # [1,5]

        return pred_norm, pred_rating


# ===================== 10. DataLoader =====================
def make_loaders(train_df, val_df, test_df, full_df, vocab):
    train_dataset = ReviewRegressionDataset(train_df, vocab, Config.MAX_LEN)
    val_dataset = ReviewRegressionDataset(val_df, vocab, Config.MAX_LEN)
    test_dataset = ReviewRegressionDataset(test_df, vocab, Config.MAX_LEN)
    full_dataset = ReviewRegressionDataset(full_df, vocab, Config.MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    full_loader = DataLoader(full_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, full_loader


# ===================== 11. indicator function =====================
def regression_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return mae, rmse, r2


# ===================== 12. Training and Evaluation =====================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    y_true, y_pred = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_norm = batch["target_norm"].to(device)
        true_rating = batch["true_rating"].to(device)

        optimizer.zero_grad()
        pred_norm, pred_rating = model(input_ids, attention_mask)

        loss = criterion(pred_norm, target_norm)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        y_true.extend(true_rating.cpu().numpy())
        y_pred.extend(pred_rating.detach().cpu().numpy())

    avg_loss = total_loss / len(loader)
    mae, rmse, r2 = regression_metrics(y_true, y_pred)

    return avg_loss, mae, rmse, r2


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_norm = batch["target_norm"].to(device)
        true_rating = batch["true_rating"].to(device)

        pred_norm, pred_rating = model(input_ids, attention_mask)
        loss = criterion(pred_norm, target_norm)

        total_loss += loss.item()

        y_true.extend(true_rating.cpu().numpy())
        y_pred.extend(pred_rating.cpu().numpy())

    avg_loss = total_loss / len(loader)
    mae, rmse, r2 = regression_metrics(y_true, y_pred)

    return avg_loss, mae, rmse, r2, y_true, y_pred


@torch.no_grad()
def predict_reviews(model, loader, device):
    model.eval()

    all_rows = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        pred_norm, pred_rating = model(input_ids, attention_mask)

        pred_norm = pred_norm.cpu().numpy()
        pred_rating = pred_rating.cpu().numpy()

        for i in range(len(pred_rating)):
            row = {
                "listing_id": batch["listing_id"][i],
                "review_id": batch["review_id"][i],
                "review_text": batch["review_text"][i],
                "true_rating": float(batch["true_rating"][i].item()),
                "pred_rating": float(pred_rating[i]),
                "review_score": float(pred_norm[i])  # 0~1
            }
            all_rows.append(row)

    pred_df = pd.DataFrame(all_rows)
    pred_df["abs_error"] = np.abs(pred_df["true_rating"] - pred_df["pred_rating"])
    return pred_df


# ===================== 13. Property Rating and ranking (Main Objective) =====================
def aggregate_listing_scores(review_pred_df):
    grp = review_pred_df.groupby("listing_id")

    listing_df = grp.agg(
        review_count=("review_id", "count"),
        avg_rating=("true_rating", "mean"),
        mean_pred_rating=("pred_rating", "mean"),
        std_pred_rating=("pred_rating", "std"),
        mean_review_score=("review_score", "mean")
    ).reset_index()

    listing_df["std_pred_rating"] = listing_df["std_pred_rating"].fillna(0.0)

    # Normalized term
    listing_df["avg_rating_norm"] = (listing_df["avg_rating"] - 1.0) / 4.0
    listing_df["mean_pred_norm"] = (listing_df["mean_pred_rating"] - 1.0) / 4.0

    # Stability: The smaller the fluctuation, the better
    listing_df["stability"] = 1.0 - np.clip(listing_df["std_pred_rating"] / 2.0, 0, 1)

    # The confidence level of comment count: The more comments there are, the more reliable it is
    listing_df["confidence"] = np.clip(
        np.log1p(listing_df["review_count"]) / np.log(50),
        0,
        1
    )

    # Sort the principal fraction
    listing_df["listing_score"] = (
        0.55 * listing_df["mean_pred_norm"] +
        0.30 * listing_df["avg_rating_norm"] +
        0.15 * listing_df["stability"]
    )

    # Final ranking score
    listing_df["final_score"] = listing_df["listing_score"] * (0.7 + 0.3 * listing_df["confidence"])

    listing_df = listing_df.sort_values(by="final_score", ascending=False).reset_index(drop=True)
    return listing_df


# ===================== 14. Ranking evaluation =====================
def evaluate_listing_ranking(listing_df, top_k=10):
    """
    Main task evaluation: listing ranking
    baseline: Sorted by avg_rating
    """
    model_ranked = listing_df.sort_values("final_score", ascending=False).reset_index(drop=True)

    baseline_ranked = listing_df.sort_values(
        ["avg_rating", "review_count"],
        ascending=[False, False]
    ).reset_index(drop=True)

    model_topk_avg_rating = model_ranked.head(top_k)["avg_rating"].mean()
    baseline_topk_avg_rating = baseline_ranked.head(top_k)["avg_rating"].mean()

    model_topk_review_count = model_ranked.head(top_k)["review_count"].mean()
    baseline_topk_review_count = baseline_ranked.head(top_k)["review_count"].mean()

    model_topk_stability = model_ranked.head(top_k)["stability"].mean()
    baseline_topk_stability = baseline_ranked.head(top_k)["stability"].mean()

    model_topk_final_score = model_ranked.head(top_k)["final_score"].mean()
    baseline_topk_final_score = baseline_ranked.head(top_k)["final_score"].mean()

    metrics = {
        "top_k": top_k,
        "model_topk_avg_rating": float(model_topk_avg_rating),
        "baseline_topk_avg_rating": float(baseline_topk_avg_rating),
        "model_topk_review_count": float(model_topk_review_count),
        "baseline_topk_review_count": float(baseline_topk_review_count),
        "model_topk_stability": float(model_topk_stability),
        "baseline_topk_stability": float(baseline_topk_stability),
        "model_topk_final_score": float(model_topk_final_score),
        "baseline_topk_final_score": float(baseline_topk_final_score),
    }

    return metrics, model_ranked, baseline_ranked


def save_ranking_metrics(metrics, output_dir):
    ranking_df = pd.DataFrame([
        ["Top-K", metrics["top_k"]],
        ["Model Top-K Avg Rating", metrics["model_topk_avg_rating"]],
        ["Baseline Top-K Avg Rating", metrics["baseline_topk_avg_rating"]],
        ["Model Top-K Avg Review Count", metrics["model_topk_review_count"]],
        ["Baseline Top-K Avg Review Count", metrics["baseline_topk_review_count"]],
        ["Model Top-K Stability", metrics["model_topk_stability"]],
        ["Baseline Top-K Stability", metrics["baseline_topk_stability"]],
        ["Model Top-K Final Score", metrics["model_topk_final_score"]],
        ["Baseline Top-K Final Score", metrics["baseline_topk_final_score"]],
    ], columns=["Metric", "Value"])

    ranking_df.to_csv(
        os.path.join(output_dir, "ranking_metrics.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    return ranking_df


# ===================== 15. Export function =====================
def save_jsonl(df, path):
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


def save_training_config(df, vocab_size, train_df, val_df, test_df, output_dir):
    config_rows = [
        ["Task", "AirRank: Intelligent Airbnb Listing Ranking Based on Review Text"],
        ["Primary objective", "Rank Airbnb listings"],
        ["Intermediate task", "Predict review-level satisfaction score from review text"],
        ["Techniques used", "CNN + BiLSTM"],
        ["Input file", Config.INNER_FILE],
        ["Total reviews", len(df)],
        ["Total listings", df["listing_id"].nunique()],
        ["Train reviews", len(train_df)],
        ["Validation reviews", len(val_df)],
        ["Test reviews", len(test_df)],
        ["Train listings", train_df["listing_id"].nunique()],
        ["Validation listings", val_df["listing_id"].nunique()],
        ["Test listings", test_df["listing_id"].nunique()],
        ["Vocabulary size", vocab_size],
        ["Max sequence length", Config.MAX_LEN],
        ["Embedding dimension", Config.EMBED_DIM],
        ["LSTM type", "BiLSTM"],
        ["Hidden dimension", Config.HIDDEN_DIM],
        ["CNN filters", Config.NUM_FILTERS],
        ["CNN kernel sizes", str(Config.KERNEL_SIZES)],
        ["Dropout", Config.DROPOUT],
        ["Batch size", Config.BATCH_SIZE],
        ["Optimizer", "Adam"],
        ["Learning rate", Config.LR],
        ["Loss function", "SmoothL1Loss on normalized review score"],
        ["Max epochs", Config.EPOCHS],
        ["Early stopping patience", Config.PATIENCE],
        ["Device", Config.DEVICE],
        ["Ranking baseline", "Sort listings by avg_rating"],
        ["Final ranking score", "Weighted combination of predicted review quality, avg rating, stability, and confidence"]
    ]

    config_df = pd.DataFrame(config_rows, columns=["Item", "Setting"])
    config_df.to_csv(
        os.path.join(output_dir, "training_configuration.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    return config_df


def save_summary_metrics(test_mae, test_rmse, test_r2, output_dir):
    summary_df = pd.DataFrame({
        "Metric": [
            "Review-level Test MAE",
            "Review-level Test RMSE",
            "Review-level Test R2"
        ],
        "Value": [test_mae, test_rmse, test_r2]
    })
    summary_df.to_csv(
        os.path.join(output_dir, "summary_metrics.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    return summary_df


# ===================== 16. visualizatio =====================
def plot_history(history, save_path):
    plt.figure(figsize=(9, 5))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.plot(history["train_mae"], label="train_mae")
    plt.plot(history["val_mae"], label="val_mae")
    plt.legend()
    plt.title("Training History (Review Scoring Module)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_true_vs_pred(y_true, y_pred, save_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.4)
    plt.xlabel("True Rating")
    plt.ylabel("Predicted Rating")
    plt.title("True vs Predicted Review Ratings")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_top_listings(listing_df, save_path, top_k=10):
    top_df = listing_df.head(top_k).copy()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=top_df, x="listing_id", y="final_score")
    plt.xticks(rotation=75)
    plt.title(f"Top {top_k} Ranked Listings (Model Ranking)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_model_vs_baseline(model_ranked_df, baseline_ranked_df, save_path, top_k=10):
    compare_df = pd.DataFrame({
        "listing_id": list(model_ranked_df.head(top_k)["listing_id"].astype(str)) +
                      list(baseline_ranked_df.head(top_k)["listing_id"].astype(str)),
        "score": list(model_ranked_df.head(top_k)["final_score"]) +
                 list(baseline_ranked_df.head(top_k)["final_score"]),
        "method": ["Model"] * top_k + ["Baseline"] * top_k
    })

    plt.figure(figsize=(12, 5))
    sns.barplot(data=compare_df, x="listing_id", y="score", hue="method")
    plt.xticks(rotation=75)
    plt.title(f"Top {top_k} Listings: Model vs Baseline")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


# ===================== 17. main process =====================
if __name__ == "__main__":
    try:
        logging.info(f"use equipment: {Config.DEVICE}")
        logging.info("The main objective of the project: Ranking Airbnb listings（listing ranking）")
        logging.info("Intermediate module: Use CNN + BiLSTM to model the satisfaction/rating of the comment text")

        # 1) Load the original comment data
        df = load_review_data(Config.ZIP_PATH, Config.INNER_FILE)
        logging.info(f"Total number of comments: {len(df)}")
        logging.info(f"Total number of available properties: {df['listing_id'].nunique()}")
        logging.info("Score distribution:\n" + str(df["rating"].value_counts().sort_index()))

        # 2) Classified by listing_id
        train_df, val_df, test_df = split_by_listing(
            df,
            test_size=Config.TEST_SIZE,
            val_size=Config.VAL_SIZE,
            random_state=42
        )

        logging.info(
            f"Train reviews={len(train_df)}, Val reviews={len(val_df)}, Test reviews={len(test_df)}"
        )
        logging.info(
            f"Train listings={train_df['listing_id'].nunique()}, "
            f"Val listings={val_df['listing_id'].nunique()}, "
            f"Test listings={test_df['listing_id'].nunique()}"
        )

        # 3) Build a vocabulary list (only use the training set)
        vocab = build_vocab(
            texts=train_df["review_text"].tolist(),
            max_vocab_size=Config.MAX_VOCAB_SIZE,
            min_freq=Config.MIN_FREQ
        )
        logging.info(f"Word list size {len(vocab)}")

        # Save the training configuration
        save_training_config(df, len(vocab), train_df, val_df, test_df, Config.OUTPUT_DIR)

        # 4) DataLoader
        train_loader, val_loader, test_loader, full_loader = make_loaders(
            train_df, val_df, test_df, df, vocab
        )

        # 5) build model
        model = TextCNNBiLSTMRegressor(
            vocab_size=len(vocab),
            embed_dim=Config.EMBED_DIM,
            hidden_dim=Config.HIDDEN_DIM,
            num_filters=Config.NUM_FILTERS,
            kernel_sizes=Config.KERNEL_SIZES,
            dropout=Config.DROPOUT
        ).to(Config.DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
        criterion = nn.SmoothL1Loss()

        # 6) train
        best_val_mae = float("inf")
        patience_counter = 0
        best_model_path = os.path.join(Config.OUTPUT_DIR, "best_cnn_bilstm_listing_ranker.pt")

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_mae": [],
            "val_mae": [],
            "train_rmse": [],
            "val_rmse": []
        }

        for epoch in range(1, Config.EPOCHS + 1):
            train_loss, train_mae, train_rmse, train_r2 = train_one_epoch(
                model, train_loader, optimizer, criterion, Config.DEVICE
            )

            val_loss, val_mae, val_rmse, val_r2, _, _ = evaluate(
                model, val_loader, criterion, Config.DEVICE
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_mae"].append(train_mae)
            history["val_mae"].append(val_mae)
            history["train_rmse"].append(train_rmse)
            history["val_rmse"].append(val_rmse)

            logging.info(
                f"Epoch [{epoch}/{Config.EPOCHS}] | "
                f"train_loss={train_loss:.4f}, train_mae={train_mae:.4f}, train_rmse={train_rmse:.4f}, train_r2={train_r2:.4f} | "
                f"val_loss={val_loss:.4f}, val_mae={val_mae:.4f}, val_rmse={val_rmse:.4f}, val_r2={val_r2:.4f}"
            )

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                logging.info("✅ The MAE of the validation set has decreased, and the best model has been saved（review scoring module）")
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    logging.info("⏹️ trigger Early Stopping")
                    break

        # Save the training history
        history_df = pd.DataFrame(history)
        history_df.to_csv(
            os.path.join(Config.OUTPUT_DIR, "training_history.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        # 7) evaluation set test（review-level）
        model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))

        test_loss, test_mae, test_rmse, test_r2, y_true, y_pred = evaluate(
            model, test_loader, criterion, Config.DEVICE
        )

        save_summary_metrics(test_mae, test_rmse, test_r2, Config.OUTPUT_DIR)

        print("\n" + "=" * 60)
        print("Review-level Evaluation")
        print(f"✅ Test MAE  : {test_mae:.4f}")
        print(f"✅ Test RMSE : {test_rmse:.4f}")
        print(f"✅ Test R2   : {test_r2:.4f}")

        # 8) Export the prediction and error samples of the test set
        test_pred_df = predict_reviews(model, test_loader, Config.DEVICE)
        test_pred_df.to_csv(
            os.path.join(Config.OUTPUT_DIR, "test_review_predictions.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        largest_errors_df = test_pred_df.sort_values("abs_error", ascending=False).head(50)
        largest_errors_df.to_csv(
            os.path.join(Config.OUTPUT_DIR, "largest_errors.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        # 9) Rate all the comments with the best model
        full_pred_df = predict_reviews(model, full_loader, Config.DEVICE)

        # export step2_review_predictions.jsonl
        step2_df = full_pred_df.copy()
        save_jsonl(step2_df, os.path.join(Config.OUTPUT_DIR, "step2_review_predictions.jsonl"))

        # 10) Generate the property level score (main objective)
        listing_score_df = aggregate_listing_scores(full_pred_df)

        # export step3_listing_scores.jsonl
        save_jsonl(listing_score_df, os.path.join(Config.OUTPUT_DIR, "step3_listing_scores.jsonl"))

        # export step4_top5_listings.json
        top5 = listing_score_df.head(5).to_dict(orient="records")
        with open(os.path.join(Config.OUTPUT_DIR, "step4_top5_listings.json"), "w", encoding="utf-8") as f:
            json.dump(top5, f, ensure_ascii=False, indent=2)

        # Save the csv file separately for easy viewing
        listing_score_df.to_csv(
            os.path.join(Config.OUTPUT_DIR, "listing_scores.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        # 10.1) Ranking evaluation: Comparison with the baseline
        ranking_metrics, model_ranked_df, baseline_ranked_df = evaluate_listing_ranking(
            listing_score_df,
            top_k=Config.TOP_K
        )

        save_ranking_metrics(ranking_metrics, Config.OUTPUT_DIR)

        model_ranked_df.head(Config.TOP_K).to_csv(
            os.path.join(Config.OUTPUT_DIR, "model_topk_listings.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        baseline_ranked_df.head(Config.TOP_K).to_csv(
            os.path.join(Config.OUTPUT_DIR, "baseline_topk_listings.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        print("\n" + "=" * 60)
        print("Listing-level Ranking Evaluation")
        print(f"Model Top-{Config.TOP_K} Avg Rating        : {ranking_metrics['model_topk_avg_rating']:.4f}")
        print(f"Baseline Top-{Config.TOP_K} Avg Rating     : {ranking_metrics['baseline_topk_avg_rating']:.4f}")
        print(f"Model Top-{Config.TOP_K} Avg Review Count  : {ranking_metrics['model_topk_review_count']:.4f}")
        print(f"Baseline Top-{Config.TOP_K} Avg Review Count: {ranking_metrics['baseline_topk_review_count']:.4f}")
        print(f"Model Top-{Config.TOP_K} Stability         : {ranking_metrics['model_topk_stability']:.4f}")
        print(f"Baseline Top-{Config.TOP_K} Stability      : {ranking_metrics['baseline_topk_stability']:.4f}")

        # 11) visualization
        plot_history(history, os.path.join(Config.OUTPUT_DIR, "training_curve.png"))
        plot_true_vs_pred(y_true, y_pred, os.path.join(Config.OUTPUT_DIR, "true_vs_pred.png"))
        plot_top_listings(listing_score_df, os.path.join(Config.OUTPUT_DIR, "top10_listings.png"), top_k=Config.TOP_K)
        plot_model_vs_baseline(
            model_ranked_df,
            baseline_ranked_df,
            os.path.join(Config.OUTPUT_DIR, "model_vs_baseline_topk.png"),
            top_k=Config.TOP_K
        )

        # 12) Save the vocabulary list
        joblib.dump(vocab, os.path.join(Config.OUTPUT_DIR, "vocab.joblib"))

        print("\nTop 5 listings (Model Ranking):")
        print(listing_score_df.head(5)[[
            "listing_id",
            "review_count",
            "avg_rating",
            "mean_pred_rating",
            "listing_score",
            "confidence",
            "final_score"
        ]])

        logging.info("🎉 All completed. The results with listing ranking as the main objective have been generated, and review-level prediction serves as the intermediate module.")

    except Exception as e:
        logging.exception(f"The program failed to run.: {e}")
        raise
