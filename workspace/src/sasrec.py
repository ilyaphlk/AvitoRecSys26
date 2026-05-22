import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

import mlflow
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from loguru import logger

import utils
from stage import BaseStage
from utils import load_config


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class _SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        h = self.norm(x)
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        h, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        return x + self.drop(h)


class _FeedForwardBlock(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class SASRecModel(nn.Module):
    def __init__(
        self,
        n_items: int,
        hidden_dim: int,
        max_seq_len: int,
        n_layers: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float,
    ):
        super().__init__()
        # index 0 is reserved for padding
        self.item_emb = nn.Embedding(n_items + 1, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.emb_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                _SelfAttentionBlock(hidden_dim, n_heads, dropout),
                _FeedForwardBlock(hidden_dim, ffn_dim, dropout),
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L) long → (B, L, D) float"""
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.emb_drop(self.item_emb(x) + self.pos_emb(pos))
        for block in self.blocks:
            h = block(h)
        return self.norm(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits (B, L, n_items) at every position, excluding the padding index."""
        h = self.encode(x)
        logits = h @ self.item_emb.weight.T  # (B, L, n_items+1)
        return logits[:, :, 1:]              # drop padding column → (B, L, n_items)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Scores over all items at the last position. (B, n_items)"""
        h = self.encode(x)[:, -1, :]        # (B, D)
        return h @ self.item_emb.weight[1:].T  # (B, n_items)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class _SASRecDataset(torch.utils.data.Dataset):
    """
    Each sample is (input_seq, target_seq) of length max_seq_len.
    input[t] → predict target[t] = input[t+1].
    Positions before the actual sequence are padded with 0.
    """

    def __init__(self, sequences: list[list[int]], max_seq_len: int):
        self.max_seq_len = max_seq_len
        self.samples = [seq for seq in sequences if len(seq) >= 2]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        seq = self.samples[idx]
        inp = seq[:-1][-self.max_seq_len:]
        tgt = seq[1:][-self.max_seq_len:]
        pad = self.max_seq_len - len(inp)
        inp = [0] * pad + inp
        tgt = [0] * pad + tgt
        return torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_sequences(
    df: pl.DataFrame,
    use_clicks_only: bool = False,
) -> tuple[dict[int, list[int]], dict[int, int], dict[int, int]]:
    """
    Build per-user item sequences sorted by timestamp.
    Item indices start at 1; index 0 is reserved for padding.
    Returns (user_sequences, item_id_to_index, user_id_to_index).
    """
    if use_clicks_only:
        df = df.filter(pl.col("is_click") == 1)

    item_ids = df["item_id"].unique().sort().to_numpy()
    user_ids = df["user_id"].unique().sort().to_numpy()

    item_id_to_index = {int(iid): int(idx) + 1 for idx, iid in enumerate(item_ids)}
    user_id_to_index = {int(uid): int(idx) for idx, uid in enumerate(user_ids)}

    seqs_raw = (
        df.sort(["user_id", "timestamp"])
        .group_by("user_id", maintain_order=True)
        .agg(pl.col("item_id").alias("item_sequence"))
    )

    user_sequences: dict[int, list[int]] = {}
    for row in seqs_raw.iter_rows(named=True):
        uid = int(row["user_id"])
        seq = [item_id_to_index[int(i)] for i in row["item_sequence"]]
        user_sequences[uid] = seq

    return user_sequences, item_id_to_index, user_id_to_index


# ──────────────────────────────────────────────────────────────────────────────
# Train
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SASRecTrainResult:
    model: Any
    model_config: Dict[str, Any]
    item_id_to_index: Dict[int, int]
    user_id_to_index: Dict[int, int]
    user_sequences: pl.DataFrame   # columns: user_id, item_sequence (list<i64>)
    popular_top: Optional[pl.DataFrame]


def train(
    df_train: pl.DataFrame,
    max_seq_len: int = 50,
    hidden_dim: int = 128,
    n_layers: int = 2,
    n_heads: int = 2,
    ffn_dim: int = 256,
    dropout: float = 0.1,
    lr: float = 1e-3,
    epochs: int = 10,
    batch_size: int = 256,
    use_clicks_only: bool = False,
    top_size: int = 100,
    make_popular_top: bool = True,
    device: str = "auto",
) -> SASRecTrainResult:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"using device: {device}")

    user_sequences, item_id_to_index, user_id_to_index = build_sequences(df_train, use_clicks_only)
    n_items = len(item_id_to_index)
    logger.info(f"n_items={n_items}, n_users={len(user_id_to_index)}")

    popular_top = None
    if make_popular_top:
        popular_top = (
            df_train.group_by("item_id")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .head(top_size)
        )

    dataset = _SASRecDataset(list(user_sequences.values()), max_seq_len)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    logger.info(f"training samples: {len(dataset)}")

    model_config = dict(
        n_items=n_items,
        hidden_dim=hidden_dim,
        max_seq_len=max_seq_len,
        n_layers=n_layers,
        n_heads=n_heads,
        ffn_dim=ffn_dim,
        dropout=dropout,
    )
    model = SASRecModel(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0

        for inp, tgt in loader:
            inp, tgt = inp.to(device), tgt.to(device)
            logits = model(inp)  # (B, L, n_items)

            # tgt has 0 for padding → shift to -1 (ignored by CE), else 0-indexed
            tgt_shifted = tgt - 1
            B, L, V = logits.shape
            loss = loss_fn(logits.reshape(B * L, V), tgt_shifted.reshape(B * L))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        logger.info(f"epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}")
        mlflow.log_metric("train_loss", avg_loss, step=epoch)

    seqs_df = pl.DataFrame({
        "user_id": list(user_sequences.keys()),
        "item_sequence": list(user_sequences.values()),
    })

    return SASRecTrainResult(
        model=model,
        model_config=model_config,
        item_id_to_index=item_id_to_index,
        user_id_to_index=user_id_to_index,
        user_sequences=seqs_df,
        popular_top=popular_top,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

def inference(
    user_to_pred: pl.DataFrame,
    model: SASRecModel,
    item_id_to_index: dict[int, int],
    user_id_to_index: dict[int, int],
    user_sequences: pl.DataFrame,
    top_size: int = 100,
    batch_size: int = 256,
    fallback_strategy: Optional[str] = None,
    popular_top: Optional[pl.DataFrame] = None,
    device: str = "auto",
) -> pl.DataFrame:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if fallback_strategy == "popular":
        assert popular_top is not None, "popular_top required when fallback_strategy='popular'"

    max_seq_len: int = model.pos_emb.num_embeddings
    index_to_item_id = {v: k for k, v in item_id_to_index.items()}

    seq_lookup: dict[int, list[int]] = dict(
        zip(user_sequences["user_id"].to_list(), user_sequences["item_sequence"].to_list())
    )

    users = user_to_pred["user_id"].to_list()
    users_warm = [u for u in users if int(u) in seq_lookup]
    users_cold = [u for u in users if int(u) not in seq_lookup]

    mlflow.log_param("users_pred_by_algo_cnt", len(users_warm))
    mlflow.log_param("users_pred_by_algo_pct", len(users_warm) / max(len(users), 1))
    if fallback_strategy is not None:
        mlflow.log_param("users_pred_by_fallback_cnt", len(users_cold))

    model = model.to(device)
    model.eval()

    warm_records: list[dict] = []
    with torch.no_grad():
        for start in range(0, len(users_warm), batch_size):
            batch_users = users_warm[start: start + batch_size]
            seqs = []
            for u in batch_users:
                seq = seq_lookup[int(u)][-max_seq_len:]
                pad = max_seq_len - len(seq)
                seqs.append([0] * pad + seq)

            x = torch.tensor(seqs, dtype=torch.long, device=device)
            scores = model.predict(x).cpu().numpy()  # (B, n_items)

            for i, uid in enumerate(batch_users):
                top_idx = np.argpartition(scores[i], -top_size)[-top_size:]
                top_idx = top_idx[np.argsort(scores[i][top_idx])[::-1]]
                warm_records.append({
                    "user_id": int(uid),
                    "item_id": [index_to_item_id[j + 1] for j in top_idx.tolist()],
                    "scores": scores[i][top_idx].tolist(),
                })

            logger.info(f"inferred {min(start + batch_size, len(users_warm))} / {len(users_warm)} warm users")

    dfs = []

    if warm_records:
        dfs.append(
            pl.DataFrame({
                "user_id": [r["user_id"] for r in warm_records],
                "item_id": [r["item_id"] for r in warm_records],
                "scores": [r["scores"] for r in warm_records],
            })
            .explode(["item_id", "scores"])
            .with_columns(
                pl.col("user_id").cast(pl.UInt32),
                pl.col("item_id").cast(pl.UInt32),
                pl.col("scores").cast(pl.Float64),
            )
        )

    if users_cold and fallback_strategy == "popular":
        logger.info(f"{len(users_cold)} cold-start users → popular fallback")
        dfs.append(
            pl.DataFrame({
                "user_id": [int(u) for u in users_cold],
                "item_id": [list(popular_top["item_id"]) for _ in users_cold],
                "scores": [list(popular_top["count"].cast(pl.Float64)) for _ in users_cold],
            })
            .explode(["item_id", "scores"])
            .with_columns(
                pl.col("user_id").cast(pl.UInt32),
                pl.col("item_id").cast(pl.UInt32),
                pl.col("scores").cast(pl.Float64),
            )
        )

    return pl.concat(dfs) if dfs else pl.DataFrame({"user_id": [], "item_id": [], "scores": []})


# ──────────────────────────────────────────────────────────────────────────────
# Stages
# ──────────────────────────────────────────────────────────────────────────────

class SASRecTrainStage(BaseStage):
    def assert_args_in_cfg(self, cfg):
        assert "in_artifacts" in cfg
        assert "train_data" in cfg["in_artifacts"]

        assert "kwargs" in cfg

        assert "out_artifacts" in cfg
        assert "artifacts_dir" in cfg["out_artifacts"]
        assert "model" in cfg["out_artifacts"]
        assert "item_id_to_index" in cfg["out_artifacts"]
        assert "user_id_to_index" in cfg["out_artifacts"]
        assert "user_sequences" in cfg["out_artifacts"]

    def parse_kwargs(self):
        return self.cfg["kwargs"]

    def load_artifacts(self):
        return {
            "df_train": utils.read_parquet(self.cfg["in_artifacts"]["train_data"]),
        }

    def write_artifacts(self, run_result: SASRecTrainResult):
        out_artifacts = self.cfg["out_artifacts"]
        make_mlflow_subdirs = out_artifacts.get("make_mlflow_artifacts_subdir", False)
        if make_mlflow_subdirs:
            mlflow_run = mlflow.active_run()
            run_id = mlflow_run.info.run_id
            exp_name = mlflow.get_experiment(mlflow_run.info.experiment_id).name
            out_artifacts["artifacts_dir"] = os.path.join(out_artifacts["artifacts_dir"], exp_name, run_id, "")

        super().write_artifacts(run_result)
        artifacts_dir = out_artifacts.get("artifacts_dir", "")

        for key in ["model", "item_id_to_index", "user_id_to_index", "user_sequences", "popular_top"]:
            if key in out_artifacts:
                out_artifacts[key] = os.path.join(artifacts_dir, out_artifacts[key])

        # Save model as a checkpoint dict so it can be loaded with weights_only=True
        checkpoint = {
            "state_dict": run_result.model.state_dict(),
            "config": run_result.model_config,
        }
        utils.save_artifact(checkpoint, out_artifacts["model"], remove_local=self.remove_local, log_artifact=self.log_artifacts)

        utils.save_artifact(
            {int(k): int(v) for k, v in run_result.item_id_to_index.items()},
            out_artifacts["item_id_to_index"],
            remove_local=self.remove_local,
            log_artifact=self.log_artifacts,
        )
        utils.save_artifact(
            {int(k): int(v) for k, v in run_result.user_id_to_index.items()},
            out_artifacts["user_id_to_index"],
            remove_local=self.remove_local,
            log_artifact=self.log_artifacts,
        )
        utils.sink_parquet(
            run_result.user_sequences,
            out_artifacts["user_sequences"],
            remove_local=self.remove_local,
            log_artifact=self.log_artifacts,
        )

        if run_result.popular_top is not None and "popular_top" in out_artifacts:
            utils.sink_parquet(
                run_result.popular_top,
                out_artifacts["popular_top"],
                remove_local=self.remove_local,
                log_artifact=self.log_artifacts,
            )


class SASRecInferenceStage(BaseStage):
    def assert_args_in_cfg(self, cfg):
        assert "in_artifacts" in cfg
        assert "eval_users" in cfg["in_artifacts"]
        assert "model" in cfg["in_artifacts"]
        assert "item_id_to_index" in cfg["in_artifacts"]
        assert "user_id_to_index" in cfg["in_artifacts"]
        assert "user_sequences" in cfg["in_artifacts"]
        assert (
            "popular_top" in cfg["in_artifacts"]
            or cfg.get("kwargs", {}).get("fallback_strategy") != "popular"
        )

        assert "out_artifacts" in cfg
        assert "submission" in cfg["out_artifacts"]

    def parse_kwargs(self):
        return self.cfg["kwargs"]

    def load_artifacts(self):
        in_artifacts = self.cfg["in_artifacts"]
        artifacts_dir = in_artifacts.get("artifacts_dir", "")
        artifacts_run_id = in_artifacts.get("artifacts_run_id", "")
        artifacts_experiment_name = in_artifacts.get("artifacts_experiment_name", "")

        for key in ["model", "item_id_to_index", "user_id_to_index", "user_sequences", "popular_top"]:
            if key in in_artifacts:
                in_artifacts[key] = os.path.join(
                    artifacts_dir, artifacts_experiment_name, artifacts_run_id, in_artifacts[key]
                )

        checkpoint = utils.load_artifact(in_artifacts["model"])
        model = SASRecModel(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        item_id_to_index = {int(k): int(v) for k, v in utils.load_artifact(in_artifacts["item_id_to_index"]).items()}
        user_id_to_index = {int(k): int(v) for k, v in utils.load_artifact(in_artifacts["user_id_to_index"]).items()}

        return {
            "user_to_pred": utils.read_csv(in_artifacts["eval_users"]),
            "model": model,
            "item_id_to_index": item_id_to_index,
            "user_id_to_index": user_id_to_index,
            "user_sequences": utils.read_parquet(in_artifacts["user_sequences"]),
            "popular_top": utils.read_parquet(in_artifacts["popular_top"]) if "popular_top" in in_artifacts else None,
        }

    def write_artifacts(self, run_result: pl.DataFrame):
        out_artifacts = self.cfg["out_artifacts"]
        make_mlflow_subdirs = out_artifacts.get("make_mlflow_artifacts_subdir", False)
        if make_mlflow_subdirs:
            mlflow_run = mlflow.active_run()
            run_id = mlflow_run.info.run_id
            exp_name = mlflow.get_experiment(mlflow_run.info.experiment_id).name
            out_artifacts["artifacts_dir"] = os.path.join(out_artifacts["artifacts_dir"], exp_name, run_id, "")

        super().write_artifacts(run_result)

        if "artifacts_dir" in out_artifacts:
            out_artifacts["submission"] = os.path.join(out_artifacts["artifacts_dir"], out_artifacts["submission"])

        utils.write_csv(
            run_result.select(pl.col("user_id"), pl.col("item_id")),
            out_artifacts["submission"],
            remove_local=self.remove_local,
            log_artifact=self.log_artifacts,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    assert len(sys.argv) == 3, "usage: sasrec.py <train_config.yml> <inference_config.yml>"
    train_config_path, inference_config_path = sys.argv[1], sys.argv[2]
    train_cfg = load_config(train_config_path)["training"]
    inference_cfg = load_config(inference_config_path)["inference"]

    logger.info("starting SASRec pipeline...")

    with mlflow.start_run(run_name="sasrec_pipeline"):
        train_stage = SASRecTrainStage(train_cfg, train)
        inference_stage = SASRecInferenceStage(inference_cfg, inference)

        train_stage.run()
        logger.info("model trained")

        inference_stage.run()
        logger.info("submission written to disk")


if __name__ == "__main__":
    main()
