import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Iterator

import mlflow
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
import mlflow
from loguru import logger

import utils
from stage import BaseStage
from utils import load_config


# ------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------

class _SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.norm(x)
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
        # index 0 reserved for padding
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
        """x: (B, L) long → (B, L, D)"""
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.emb_drop(self.item_emb(x) + self.pos_emb(pos))
        for block in self.blocks:
            h = block(h)
        return self.norm(h)

    def score_sampled(
        self,
        x: torch.Tensor,             # (B, L) input sequences
        pos_targets: torch.Tensor,   # (B, L) per-position positive item indices (0 = padding/ignore)
        neg_items: torch.Tensor,     # (N,) sampled negative item indices, shared across the batch
        pos_log_q: torch.Tensor,     # (B, L) log sampling prob of each positive
        neg_log_q: torch.Tensor,     # (N,) log sampling prob of each negative
    ) -> torch.Tensor:
        """
        Per-position sampled-softmax logits with log-Q correction.
        Returns (B, L, 1+N) — column 0 is the positive at each position.
        """
        h = self.encode(x)                              # (B, L, D)
        e_pos = self.item_emb(pos_targets)              # (B, L, D)
        e_neg = self.item_emb(neg_items)                # (N, D)

        # Positive logit per position
        pos_logits = (h * e_pos).sum(dim=-1, keepdim=True) - pos_log_q.unsqueeze(-1)  # (B, L, 1)
        # Negative logits shared across positions in the batch
        neg_logits = h @ e_neg.T - neg_log_q.view(1, 1, -1)                            # (B, L, N)
        return torch.cat([pos_logits, neg_logits], dim=-1)                             # (B, L, 1+N)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Scores over all items (excl. padding) at the last position. (B, n_items)"""
        h = self.encode(x)[:, -1, :]
        return h @ self.item_emb.weight[1:].T


# ------------------------------------------------------------------------------
# Dataset — yields full target sequence (one prediction per position)
# ------------------------------------------------------------------------------

class _SASRecIterableDataset(torch.utils.data.IterableDataset):
    """
    Streams per-user sequences from a parquet file using PyArrow batched reading
    For each user yields:
      inp: (max_seq_len,) input items, left-padded with 0
      tgt: (max_seq_len,) next-item targets per position, left-padded with 0
           (0 = ignored in loss)
    """

    def __init__(self, sequences_path: str, max_seq_len: int, chunk_size: int = 10_000):
        self.local_path = os.path.join(utils.LOCAL_DATA_DIR, sequences_path)
        self.max_seq_len = max_seq_len
        self.chunk_size = chunk_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        n_workers = 1 if worker_info is None else worker_info.num_workers
        worker_id = 0 if worker_info is None else worker_info.id

        pf = pq.ParquetFile(self.local_path)
        for batch_idx, pa_batch in enumerate(
            pf.iter_batches(batch_size=self.chunk_size, columns=["item_sequence"])
        ):
            if batch_idx % n_workers != worker_id:
                continue
            for seq in pa_batch.column("item_sequence").to_pylist():
                if len(seq) < 2:
                    continue
                # Keep at most max_seq_len + 1 items so input/target both fit in max_seq_len
                full = list(seq)[-(self.max_seq_len + 1):]
                inp = full[:-1]
                tgt = full[1:]
                pad = self.max_seq_len - len(inp)
                inp = [0] * pad + inp
                tgt = [0] * pad + tgt
                yield (
                    torch.tensor(inp, dtype=torch.long),
                    torch.tensor(tgt, dtype=torch.long),
                )


# ------------------------------------------------------------------------------
# Negative sampler — popularity^alpha proposal with log-Q correction
# ------------------------------------------------------------------------------

class _PopularityNegativeSampler:
    """
    Samples negative item indices proportional to (count + eps)^alpha
    alpha=0.0  → uniform
    alpha=0.75 → word2vec-style (recommended default)
    alpha=1.0  → strictly proportional to popularity
    """

    def __init__(
        self,
        item_counts: np.ndarray,
        alpha: float = 0.75,
        device: str = "cpu",
        eps: float = 1.0,
    ):
        weights = (item_counts.astype(np.float64) + eps) ** alpha
        probs = weights / weights.sum()
        self.probs = torch.from_numpy(probs).to(device).float()
        self.log_q = torch.log(self.probs.clamp_min(1e-30))
        self.n_items = item_counts.shape[0]
        self.device = device

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (item_indices in [1, n_items], log_q for those indices)"""
        idx = torch.multinomial(self.probs, n, replacement=True)
        item_indices = idx + 1
        return item_indices, self.log_q[idx]

    def log_q_of(self, item_indices: torch.Tensor) -> torch.Tensor:
        """log q for given item embedding indices (1-indexed). Padding (0) → 0.0 (will be masked)"""
        safe = item_indices.clamp(min=1)
        return self.log_q[safe - 1]

# ------------------------------------------------------------------------------
# Preprocess
# ------------------------------------------------------------------------------

@dataclass
class SASRecPreprocessResult:
    sequences: pl.LazyFrame  # (user_id, item_sequence)
    item_id_to_index: Dict[int, int]
    user_id_to_index: Dict[int, int]
    popular_top: Optional[pl.DataFrame]
    item_counts: Optional[pl.DataFrame]

CONTACT_EIDS = [0, 2, 4, 5, 6, 8, 9, 11, 14, 15, 16]

def preprocess(
    train_data: pl.LazyFrame,
    use_clicks_only: bool = False,
    make_popular_top: bool = True,
    make_item_counts: bool = True,
    top_size: int = 100,
) -> SASRecPreprocessResult:
    """
    Build per-user item sequences from raw events without loading the full dataset
    Unknown item IDs are silently dropped via an inner join
    Returns a lazy sequences frame that is sunk to disk by write_artifacts
    """

    if use_clicks_only:
        # train_data = train_data.filter(pl.col("is_click") == 1)
        train_data = train_data.filter(pl.col("eid").is_in(CONTACT_EIDS))

    logger.info("scanning unique item and user IDs...")
    item_ids = train_data.select("item_id").unique().collect()["item_id"].sort().to_numpy()
    user_ids = train_data.select("user_id").unique().collect()["user_id"].sort().to_numpy()
    logger.info(f"n_items={len(item_ids)}, n_users={len(user_ids)}")

    # index 0 reserved for padding
    item_id_to_index = {int(iid): int(idx) + 1 for idx, iid in enumerate(item_ids)}
    user_id_to_index = {int(uid): int(idx) for idx, uid in enumerate(user_ids)}

    item_map = pl.DataFrame({
        "item_id": list(item_id_to_index.keys()),
        "item_idx": list(item_id_to_index.values()),
    }).lazy()

    sequences_lazy = (
        train_data
        .join(item_map, on="item_id", how="inner")
        .sort(["user_id", "timestamp"])
        .group_by("user_id")
        .agg(pl.col("item_idx").alias("item_sequence"))
    )

    item_counts = None
    if make_item_counts:
        logger.info("making item counts...")
        item_counts = (
            train_data
            .group_by("item_id")
            .agg(pl.len().alias("count"))
            .collect()
        )

    popular_top = None
    if make_popular_top:
        logger.info("computing popular items...")
        if item_counts is not None:
            popular_top = item_counts.sort("count", descending=True).head(top_size)
        else:
            popular_top = (
                train_data
                .group_by("item_id")
                .agg(pl.len().alias("count"))
                .sort("count", descending=True)
                .head(top_size)
                .collect()
            )

    return SASRecPreprocessResult(
        sequences=sequences_lazy,
        item_id_to_index=item_id_to_index,
        user_id_to_index=user_id_to_index,
        popular_top=popular_top,
        item_counts=item_counts,
    )

# ------------------------------------------------------------------------------
# Train
# ------------------------------------------------------------------------------

@dataclass
class SASRecTrainResult:
    model: Any
    model_config: Dict[str, Any]


def train(
    sequences_path: str,
    item_id_to_index: Dict[int, int],
    user_id_to_index: Dict[int, int],
    item_counts: np.ndarray,
    max_seq_len: int = 50,
    hidden_dim: int = 128,
    n_layers: int = 2,
    n_heads: int = 2,
    ffn_dim: int = 256,
    dropout: float = 0.1,
    lr: float = 1e-3,
    epochs: int = 10,
    batch_size: int = 256,
    n_negatives: int = 4096,
    neg_sampling_alpha: float = 0.75,
    temperature: float = 0.1,
    chunk_size: int = 10_000,
    num_workers: int = 2,
    seed: int = 42,
    device: str = "auto",
    checkpoint_path: Optional[str] = None,
) -> SASRecTrainResult:
    assert device in {"auto", "cpu", "cuda"}, f"invalid device: {device}"
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device={device}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    mlflow.log_param("seed", seed)

    n_items = len(item_id_to_index)
    assert item_counts.shape == (n_items,), (
        f"item_counts must have shape ({n_items},), got {item_counts.shape}"
    )
    logger.info(f"n_items={n_items}, n_users={len(user_id_to_index)}")

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

    sampler = _PopularityNegativeSampler(
        item_counts=item_counts, alpha=neg_sampling_alpha, device=device,
    )

    dataset = _SASRecIterableDataset(sequences_path, max_seq_len, chunk_size)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True,
    )

    for epoch in range(epochs):
        model.train()
        total_loss, batch_idx, batches_in_epoch = 0.0, 0, 0

        for inp, tgt in loader:
            logger.info(f"processing batch {batch_idx}...")
            inp = inp.to(device)        # (B, L)
            tgt = tgt.to(device)        # (B, L)
            B, L = inp.shape

            # Sample negatives shared across the batch
            neg_items, neg_log_q = sampler.sample(n_negatives)  # (N,), (N,)
            pos_log_q = sampler.log_q_of(tgt)                   # (B, L)

            logits = model.score_sampled(
                inp, tgt, neg_items, pos_log_q, neg_log_q,
            ) / temperature                                     # (B, L, 1+N)

            collision = neg_items.view(1, 1, -1) == tgt.unsqueeze(-1)
            collision = torch.cat(
                [torch.zeros(B, L, 1, dtype=torch.bool, device=device), collision],
                dim=-1,
            )
            logits = logits.masked_fill(collision, float("-inf"))

            labels = torch.zeros(B, L, dtype=torch.long, device=device)
            valid = tgt != 0                                    # (B, L)
            labels = labels.masked_fill(~valid, -100)

            loss = F.cross_entropy(
                logits.reshape(B * L, -1),
                labels.reshape(B * L),
                ignore_index=-100,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            mlflow.log_metric("batch_train_loss", batch_loss, step=batches_in_epoch * epoch + batch_idx)

            total_loss += batch_loss
            batch_idx += 1

        batches_in_epoch = batch_idx
        logger.info(f"processed {batches_in_epoch} batches in epoch {epoch + 1}")

        avg_loss = total_loss / max(batches_in_epoch, 1)
        logger.info(f"epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}")
        mlflow.log_metric("train_loss", avg_loss, step=epoch)

        # Periodic checkpoint — cheap insurance against crashes
        if checkpoint_path is not None:
            ckpt_full = os.path.join(utils.LOCAL_DATA_DIR, checkpoint_path)
            os.makedirs(os.path.dirname(ckpt_full), exist_ok=True)
            torch.save(
                {"state_dict": model.state_dict(), "config": model_config, "epoch": epoch + 1},
                ckpt_full,
            )

    return SASRecTrainResult(model=model, model_config=model_config)


# ------------------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------------------

def _left_pad_sequences(seqs: list[list[int]], max_seq_len: int) -> np.ndarray:
    """Left-pad a list of variable-length int sequences into a (B, max_seq_len) int64 array"""
    out = np.zeros((len(seqs), max_seq_len), dtype=np.int64)
    for i, seq in enumerate(seqs):
        s = seq[-max_seq_len:]
        if s:
            out[i, max_seq_len - len(s):] = s
    return out


def _iter_eval_user_sequences(
    sequences_path: str,
    eval_users: set[int],
    chunk_size: int,
) -> Iterator[tuple[list[int], list[list[int]]]]:
    """
    Stream (user_ids, item_sequences) chunks from the sequences parquet,
    filtered to the eval user set. Each yielded chunk holds up to `chunk_size`
    matching users
    """
    local_path = os.path.join(utils.LOCAL_DATA_DIR, sequences_path)
    pf = pq.ParquetFile(local_path)
    eval_users_series = pl.Series("user_id", list(eval_users), dtype=pl.Int64)

    for pa_batch in pf.iter_batches(
        batch_size=chunk_size, columns=["user_id", "item_sequence"]
    ):
        df = pl.from_arrow(pa_batch).filter(
            pl.col("user_id").cast(pl.Int64).is_in(eval_users_series)
        )
        if len(df) == 0:
            continue
        yield (
            df["user_id"].cast(pl.Int64).to_list(),
            df["item_sequence"].to_list(),
        )


@torch.no_grad()
def _score_batch(
    model,
    seqs: list[list[int]],
    max_seq_len: int,
    top_size: int,
    n_items: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Score one model batch. Returns (top_idx, top_scores), both (B, top_size)"""
    inputs = _left_pad_sequences(seqs, max_seq_len)
    x = torch.from_numpy(inputs).to(device)
    scores = model.predict(x)  # (B, n_items)
    top_scores, top_idx = torch.topk(scores, k=min(top_size, n_items), dim=1)
    return top_idx.cpu().numpy(), top_scores.cpu().numpy()


def _build_fallback_df(
    users_without: set[int],
    popular_top: pl.DataFrame,
    top_size: int,
    chunk_users: int = 100_000,
) -> Iterator[pl.DataFrame]:
    """
    Yield popular-fallback prediction frames in chunks to avoid materializing
    (n_cold_users x top_size) rows in one allocation
    """
    pop_item_ids = popular_top["item_id"].cast(pl.Int64).to_numpy()
    pop_scores = popular_top["count"].cast(pl.Float64).to_numpy()
    n_pop = min(len(pop_item_ids), top_size)
    pop_item_ids = pop_item_ids[:n_pop]
    pop_scores = pop_scores[:n_pop]

    users_arr = np.fromiter(users_without, dtype=np.int64, count=len(users_without))
    for start in range(0, len(users_arr), chunk_users):
        chunk = users_arr[start:start + chunk_users]
        yield pl.DataFrame({
            "user_id": np.repeat(chunk, n_pop),
            "item_id": np.tile(pop_item_ids, len(chunk)),
            "scores": np.tile(pop_scores, len(chunk)),
        })


def inference(
    user_to_pred: pl.DataFrame,
    model,
    item_id_to_index: Dict[int, int],
    sequences_path: str,
    top_size: int = 100,
    batch_size: int = 64,
    chunk_size: int = 10_000,
    fallback_strategy: Optional[str] = None,
    popular_top: Optional[pl.DataFrame] = None,
    device: str = "auto",
) -> pl.DataFrame:
    """
    Score eval users against the full item catalog using the trained SASRec model
    Returns pl.DataFrame with schema (user_id, item_id, scores)
    Users with no sequence history get popular-item fallback if requested
    """
    assert device in {"auto", "cpu", "cuda"}, f"invalid device: {device}"
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"inference device={device}")

    if fallback_strategy is not None:
        assert fallback_strategy == "popular", (
            f"unsupported fallback_strategy: {fallback_strategy}"
        )
        assert popular_top is not None, "popular_top required for fallback_strategy='popular'"

    model = model.to(device)
    model.eval()

    n_items = len(item_id_to_index)
    max_seq_len: int = model.pos_emb.num_embeddings

    index_to_item_array = np.zeros(n_items + 1, dtype=np.int64)
    for iid, idx in item_id_to_index.items():
        index_to_item_array[idx] = int(iid)

    eval_user_set = set(user_to_pred["user_id"].cast(pl.Int64).to_list())
    logger.info(f"scoring {len(eval_user_set)} eval users")

    result_dfs: list[pl.DataFrame] = []
    users_with_preds: set[int] = set()

    buf_users: list[int] = []
    buf_seqs: list[list[int]] = []

    def flush():
        if not buf_users:
            return
        top_idx, top_scores = _score_batch(
            model, buf_seqs, max_seq_len, top_size, n_items, device,
        )
        top_item_ids = index_to_item_array[top_idx + 1]  # (B, top_size)
        B, K = top_item_ids.shape
        result_dfs.append(pl.DataFrame({
            "user_id": np.repeat(np.array(buf_users, dtype=np.int64), K),
            "item_id": top_item_ids.ravel(),
            "scores": top_scores.ravel().astype(np.float64),
        }))
        users_with_preds.update(buf_users)
        buf_users.clear()
        buf_seqs.clear()

    last_logged = 0
    for chunk_users, chunk_seqs in _iter_eval_user_sequences(
        sequences_path, eval_user_set, chunk_size,
    ):
        for uid, seq in zip(chunk_users, chunk_seqs):
            buf_users.append(uid)
            buf_seqs.append(list(seq))
            if len(buf_users) >= batch_size:
                flush()

        if len(users_with_preds) - last_logged >= 10000:
            logger.info(f"scored {len(users_with_preds)} / {len(eval_user_set)} users")
            last_logged = len(users_with_preds)

    flush()  # remainder

    coverage = len(users_with_preds) / max(len(eval_user_set), 1)
    mlflow.log_metric("users_pred_by_algo_cnt", len(users_with_preds))
    mlflow.log_metric("users_pred_by_algo_pct", coverage)
    logger.info(f"model coverage: {len(users_with_preds)} / {len(eval_user_set)} ({coverage:.1%})")

    users_without = eval_user_set - users_with_preds
    if users_without:
        if fallback_strategy == "popular":
            logger.info(f"applying popular fallback to {len(users_without)} users")
            for df in _build_fallback_df(users_without, popular_top, top_size):
                result_dfs.append(df)
        else:
            logger.warning(
                f"{len(users_without)} eval users have no model predictions and no fallback set"
            )

    if not result_dfs:
        return pl.DataFrame(
            schema={"user_id": pl.UInt32, "item_id": pl.UInt32, "scores": pl.Float64}
        )

    return (
        pl.concat(result_dfs)
        .with_columns(
            pl.col("user_id").cast(pl.UInt32),
            pl.col("item_id").cast(pl.UInt32),
            pl.col("scores").cast(pl.Float64),
        )
    )


def _apply_artifacts_dir(out_artifacts: dict, keys: list[str]):
    artifacts_dir = out_artifacts.get("artifacts_dir", "")
    for k in keys:
        if k in out_artifacts:
            out_artifacts[k] = os.path.join(artifacts_dir, out_artifacts[k])


def _maybe_add_mlflow_subdir(out_artifacts: dict):
    if not out_artifacts.get("make_mlflow_artifacts_subdir", False):
        return
    run = mlflow.active_run()
    run_id = run.info.run_id
    exp_name = mlflow.get_experiment(run.info.experiment_id).name
    out_artifacts["artifacts_dir"] = os.path.join(
        out_artifacts["artifacts_dir"], exp_name, run_id, ""
    )


class SASRecPreprocessStage(BaseStage):
    def assert_args_in_cfg(self, cfg):
        assert "in_artifacts" in cfg
        assert "train_data" in cfg["in_artifacts"]

        assert "out_artifacts" in cfg
        assert "artifacts_dir" in cfg["out_artifacts"]
        assert "sequences" in cfg["out_artifacts"]
        assert "item_id_to_index" in cfg["out_artifacts"]
        assert "user_id_to_index" in cfg["out_artifacts"]

    def parse_kwargs(self):
        return self.cfg.get("kwargs", {})

    def load_artifacts(self):
        return {"train_data": utils.scan_parquet(self.cfg["in_artifacts"]["filename_in"])}

    def write_artifacts(self, result: SASRecPreprocessResult):
        out = self.cfg["out_artifacts"]
        _maybe_add_mlflow_subdir(out)
        super().write_artifacts(result)
        _apply_artifacts_dir(out, ["sequences", "item_id_to_index", "user_id_to_index", "popular_top", "item_counts"])

        utils.sink_parquet(result.sequences, out["sequences"],
                           remove_local=self.remove_local, log_artifact=self.log_artifacts)

        utils.save_artifact({int(k): int(v) for k, v in result.item_id_to_index.items()},
                            out["item_id_to_index"],
                            remove_local=self.remove_local, log_artifact=self.log_artifacts)
        utils.save_artifact({int(k): int(v) for k, v in result.user_id_to_index.items()},
                            out["user_id_to_index"],
                            remove_local=self.remove_local, log_artifact=self.log_artifacts)

        if result.popular_top is not None and "popular_top" in out:
            utils.sink_parquet(result.popular_top, out["popular_top"],
                               remove_local=self.remove_local, log_artifact=self.log_artifacts)

        if result.item_counts is not None and "item_counts" in out:
            utils.sink_parquet(result.item_counts, out["item_counts"],
                               remove_local=self.remove_local, log_artifact=self.log_artifacts)

    def _func(self, *args, **kwargs):
        return preprocess(*args, **kwargs)


class SASRecTrainStage(BaseStage):
    def assert_args_in_cfg(self, cfg):
        assert "in_artifacts" in cfg
        assert "sequences" in cfg["in_artifacts"]
        assert "item_id_to_index" in cfg["in_artifacts"]
        assert "user_id_to_index" in cfg["in_artifacts"]
        assert "item_counts" in cfg["in_artifacts"]

        assert "kwargs" in cfg

        assert "out_artifacts" in cfg
        assert "artifacts_dir" in cfg["out_artifacts"]
        assert "model" in cfg["out_artifacts"]

    def parse_kwargs(self):
        return self.cfg["kwargs"]

    def load_artifacts(self):
        in_a = self.cfg["in_artifacts"]
        item_id_to_index = {int(k): int(v) for k, v in utils.load_artifact(in_a["item_id_to_index"]).items()}
        user_id_to_index = {int(k): int(v) for k, v in utils.load_artifact(in_a["user_id_to_index"]).items()}
        item_counts = utils.read_parquet(in_a["item_counts"])
        idx = item_counts["item_id"].replace(item_id_to_index).to_numpy()
        cnt = item_counts["count"].to_numpy()
        item_counts_npy = np.zeros(idx.max(), dtype=cnt.dtype)
        item_counts_npy[idx - 1] = cnt

        return {
            "sequences_path": in_a["sequences"],
            "item_id_to_index": item_id_to_index,
            "user_id_to_index": user_id_to_index,
            "item_counts": item_counts_npy
        }

    def write_artifacts(self, result: SASRecTrainResult):
        out = self.cfg["out_artifacts"]
        _maybe_add_mlflow_subdir(out)
        super().write_artifacts(result)
        _apply_artifacts_dir(out, ["model"])

        checkpoint = {
            "state_dict": result.model.state_dict(),
            "config": result.model_config,
        }
        utils.save_artifact(checkpoint, out["model"],
                            remove_local=self.remove_local, log_artifact=self.log_artifacts)

    def _func(self, *args, **kwargs):
        return train(*args, **kwargs)


class SASRecInferenceStage(BaseStage):
    def assert_args_in_cfg(self, cfg):
        assert "in_artifacts" in cfg
        assert "eval_users" in cfg["in_artifacts"]
        assert "model" in cfg["in_artifacts"]
        assert "item_id_to_index" in cfg["in_artifacts"]
        assert "sequences" in cfg["in_artifacts"]
        assert (
            "popular_top" in cfg["in_artifacts"]
            or cfg.get("kwargs", {}).get("fallback_strategy") != "popular"
        )

        assert "out_artifacts" in cfg
        assert "submission" in cfg["out_artifacts"]

    def parse_kwargs(self):
        return self.cfg["kwargs"]

    def load_artifacts(self):
        in_a = self.cfg["in_artifacts"]
        artifacts_dir = in_a.get("artifacts_dir", "")
        run_id = in_a.get("artifacts_run_id", "")
        exp_name = in_a.get("artifacts_experiment_name", "")

        for key in ["model", "item_id_to_index", "sequences", "popular_top"]:
            if key in in_a:
                in_a[key] = os.path.join(artifacts_dir, exp_name, run_id, in_a[key])

        checkpoint = utils.load_artifact(in_a["model"])
        model = SASRecModel(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        item_id_to_index = {int(k): int(v) for k, v in
                             utils.load_artifact(in_a["item_id_to_index"]).items()}

        return {
            "user_to_pred": utils.read_csv(in_a["eval_users"]),
            "model": model,
            "item_id_to_index": item_id_to_index,
            "sequences_path": in_a["sequences"],
            "popular_top": utils.read_parquet(in_a["popular_top"]) if "popular_top" in in_a else None,
        }

    def write_artifacts(self, result: pl.DataFrame):
        out = self.cfg["out_artifacts"]
        _maybe_add_mlflow_subdir(out)
        super().write_artifacts(result)

        if "artifacts_dir" in out:
            out["submission"] = os.path.join(out["artifacts_dir"], out["submission"])

        utils.write_csv(
            result.select(pl.col("user_id"), pl.col("item_id")),
            out["submission"],
            remove_local=self.remove_local,
            log_artifact=self.log_artifacts,
        )

    def _func(self, *args, **kwargs):
        return inference(*args, **kwargs)


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------

def main():
    assert len(sys.argv) == 4, (
        "usage: sasrec.py <preprocess_config.yml> <train_config.yml> <inference_config.yml>"
    )
    preprocess_cfg = load_config(sys.argv[1])["preprocessing"]
    train_cfg = load_config(sys.argv[2])["training"]
    inference_cfg = load_config(sys.argv[3])["inference"]

    with mlflow.start_run(run_name="sasrec_pipeline"):
        SASRecPreprocessStage(preprocess_cfg, preprocess).run()
        logger.info("preprocessing done")
        SASRecTrainStage(train_cfg, train).run()
        logger.info("training done")
        SASRecInferenceStage(inference_cfg, inference).run()
        logger.info("inference done")


if __name__ == "__main__":
    main()
