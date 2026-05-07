from scipy.sparse import csr_matrix
import numpy as np
import implicit
import os
import sys
import polars as pl
from loguru import logger
from debug_constants import DEBUG_ARGV_ALS, ARGV_ALS_LOCAL_SEPARATE, SUBMIT_ARGV_ALS
from pathlib import Path
import time
import psutil
from utils import load_config
from dataclasses import dataclass
from typing import Any, Dict


def ram_report():
    mem = psutil.virtual_memory()
    available_bytes = mem.available
    available_gb = available_bytes / (1024 ** 3)
    logger.info(f"RAM Available/Total/Usage: {available_gb:.2f}GB / {mem.total / (1024 ** 3):.2f}GB / {mem.percent}%")

@dataclass
class ALSTrainResult:
    model: Any
    item_id_to_index: Dict[int, int]
    user_id_to_index: Dict[int, int]
    user_matrix: Any | None
    popular_top: Any | None


def train(
        df_train,
        user_to_pred,
        show_weight=1,
        click_weight=0,
        iterations=10,
        factors=60,
        random_state=42,
        calculate_training_loss=True,
        top_size=160,
        make_popular_top=True,
        make_user_matrix=True
    ):
    user_ids = df_train["user_id"].unique().to_numpy()
    item_ids = df_train["item_id"].unique().to_numpy()

    logger.info("made unique")

    user_id_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_id_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}

    n_unique_users, n_unique_items = len(user_ids), len(item_ids)
    del user_ids, item_ids

    logger.info(f"made maps to idx. max user_idx: {len(user_id_to_index)}, max_item_idx: {len(item_id_to_index)}")
    logger.debug("deleted user_ids, item_ids.")

    rows = df_train["user_id"].replace_strict(user_id_to_index).cast(pl.UInt32).to_numpy()
    cols = df_train["item_id"].replace_strict(item_id_to_index).cast(pl.UInt32).to_numpy()

    logger.info("made rows & cols")

    values = (
        show_weight * df_train["cnt_shows_by_user_id_item_id"]
        + click_weight * df_train["cnt_clicks_by_user_id_item_id"]
    ).cast(pl.Float32).to_numpy()

    popular_top = None
    if make_popular_top:
    # for the non-als preds below
        popular_top = (
            pl.DataFrame({"item_id": df_train["item_id"]})
            .group_by("item_id")
            .agg(pl.len().alias("count"))
            .sort(by=("count"), descending=True)
            .head(top_size)
        )

    del df_train
    logger.debug("deleted df_train")
    ram_report()

    logger.info("start init matrix...")
    sparse_matrix = csr_matrix((values, (rows, cols)),shape=(n_unique_users, n_unique_items))
    del rows, cols, values
    logger.info("finish init matrix")
    ram_report()

    logger.info("start fit model...")
    model = implicit.als.AlternatingLeastSquares(
        iterations=iterations,
        factors=factors,
        random_state=random_state,
        calculate_training_loss=calculate_training_loss
    )
    model.fit(sparse_matrix, )
    logger.info("finish fit model")

    user4pred_als = np.array([user_id_to_index[i] for i in user_to_pred if i in user_id_to_index])
    user_matrix = sparse_matrix[user4pred_als] if make_user_matrix else None
    del sparse_matrix
    logger.debug("deleted full matrix")

    ram_report()

    return ALSTrainResult(
        model=model,
        item_id_to_index=item_id_to_index,
        user_id_to_index=user_id_to_index,
        user_matrix=user_matrix,
        popular_top=popular_top,
    )

def inference(
        user_to_pred,
        model,
        item_id_to_index,
        user_id_to_index,
        filter_already_liked_items=False,
        user_matrix=None,
        batch_size=100,
        top_size=160,
        fallback_strategy=None,
        popular_top=None,
    ):

    if fallback_strategy == "popular":
        assert popular_top is not None, "when using 'popular' fallback strategy, provide top popular items"
    
    if filter_already_liked_items:
        assert user_matrix is not None, "need saved user matrix for inference if filter_already_liked_items=True"

    logger.debug(f"user matrix: {user_matrix}")

    user4pred_als_idx = np.array([user_id_to_index[i] for i in user_to_pred if i in user_id_to_index])
    user4pred_fallback = np.array([i for i in user_to_pred if i not in user_id_to_index])

    all_recommendations = []
    all_scores = []

    total_batches = (len(user4pred_als_idx) + batch_size - 1) // batch_size
    for start in range(0, len(user4pred_als_idx), batch_size):
        logger.info(f"start recommending for batch {start // batch_size + 1} / {total_batches}")
        end = min(start + batch_size, len(user4pred_als_idx))
        batch_user_ids = user4pred_als_idx[start:end]
        batch_user_matrix = None
        if filter_already_liked_items:
            batch_user_matrix = user_matrix[start:end]
        else:
            csr_matrix((len(batch_user_ids), model.item_factors.shape[0]))
        logger.debug("copied batch into ram")
        ram_report()

        batch_recs, batch_scores = model.recommend(
            batch_user_ids,
            batch_user_matrix,
            N=top_size,
            filter_already_liked_items=filter_already_liked_items
        )
        logger.debug("finished recommending")
        ram_report()

        all_recommendations.append(batch_recs)
        all_scores.append(batch_scores)
        del batch_recs, batch_scores, batch_user_ids, batch_user_matrix

        logger.debug("deleted local vars explicitly")
        ram_report()
        logger.info(f"recommended for users {start}:{end} / {len(user4pred_als_idx)}")


    recommendations = np.vstack(all_recommendations)
    scores = np.vstack(all_scores)

    logger.info(f"got recs for {len(recommendations)} users seen in train")

    index_to_item_id = {v:k for k,v in item_id_to_index.items()}
    index_to_user_id = {v:k for k,v in user_id_to_index.items()}

    logger.info("made maps from idx")

    df_pred = pl.DataFrame(
        {
            'item_id': [[index_to_item_id[i] for i in i] for i in recommendations.tolist()],
            'user_id': [index_to_user_id[i] for i in user4pred_als_idx.tolist()],
            'scores': scores.tolist()
        }
    )

    logger.info("made df_pred for als recs")

    df_pred = df_pred.explode(['item_id', 'scores'])

    logger.info("exploded it")

    if fallback_strategy == "popular":
        # fallback to popular items
        
        logger.info(f"{len(user4pred_fallback)} / {len(user_to_pred)} users to pred by popularity (cold start)")

        df_pred_popular = pl.DataFrame(
            {
                'item_id': [list(popular_top["item_id"]) for _ in range(len(user4pred_fallback))],
                'user_id': user4pred_fallback,
                'scores': [list(popular_top["count"] * 1.0) for _ in range(len(user4pred_fallback))]
            }
        )
        df_pred_popular = df_pred_popular.explode(['item_id', 'scores']).with_columns(
            pl.col("item_id").cast(pl.UInt32).alias("item_id"),
            pl.col("user_id").cast(pl.UInt32).alias("user_id"),
            pl.col("scores").cast(pl.Float64).alias("scores"),
        )

        return pl.concat([df_pred, df_pred_popular])

    return df_pred


def main():
    assert len(sys.argv) == 3, "please provide a path to yaml config as arguments, (training, inference)"
    train_config_path, inference_config_path = sys.argv[1], sys.argv[2]
    cfg = load_config(train_config_path)["training"]
    cfg_inference = load_config(inference_config_path)["inference"]

    logger.info("starting pipeline...")

    train_path = Path(cfg["train_events_path"])
    if os.path.isdir(train_path):
        full_paths = [os.path.join(train_path, fn) for fn in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, fn))]
    else:
        full_paths = [train_path]
    logger.info(f"full paths to train parts: {full_paths}")

    collected_train_parts = []

    logger.info(f"collecting part {cfg['eval_users_events_path']}...")
    collected_train_parts.append(
        (
            pl.scan_parquet(cfg["eval_users_events_path"])
            .select(
                pl.col("user_id"),
                pl.col("item_id"),
                pl.col("cnt_shows_by_user_id_item_id"),
                pl.col("cnt_clicks_by_user_id_item_id")
            )
            .group_by(["user_id", "item_id"])
            .agg(pl.col("cnt_shows_by_user_id_item_id").first(), pl.col("cnt_clicks_by_user_id_item_id").first()).collect()
        )
    )

    for fp in sorted(full_paths):
        logger.info(f"collecting part {fp}...")
        collected_train_parts.append(
            (
                pl.scan_parquet(fp)
                .select(
                    pl.col("user_id"),
                    pl.col("item_id"),
                    pl.col("cnt_shows_by_user_id_item_id"),
                    pl.col("cnt_clicks_by_user_id_item_id")
                )
                .group_by(["user_id", "item_id"])
                .agg(pl.col("cnt_shows_by_user_id_item_id").first(), pl.col("cnt_clicks_by_user_id_item_id").first()).collect()
            )
        )

    logger.info("concatenating collected parts..")

    df_train = pl.concat(collected_train_parts)
    del collected_train_parts

    logger.info("concatenated successfully.")

    df_test = pl.read_csv(cfg_inference["eval_users"])
    train_result = train(
        df_train,
        df_test["user_id"],
        iterations=cfg["steps"],
        factors=cfg["hidden_dim"],
        show_weight=cfg.get("show_weight", 1),
        click_weight=cfg.get("click_weight", 0),
        random_state=cfg.get("random_state", None),
        calculate_training_loss=cfg.get("calculate_training_loss", False),
        make_popular_top=cfg.get("make_popular_top", False),
        make_user_matrix=cfg.get("make_user_matrix", False),
    )
    logger.info("got preds")

    df_pred = inference(
        df_test["user_id"],
        model=train_result.model,
        item_id_to_index=train_result.item_id_to_index,
        user_id_to_index=train_result.user_id_to_index,
        filter_already_liked_items=cfg_inference.get("filter_already_liked_items", False),
        user_matrix=train_result.user_matrix,
        batch_size=cfg_inference["batch_size"],
        top_size=cfg_inference["top_size"],
        fallback_strategy=cfg_inference.get("fallback_strategy", None),
        popular_top=train_result.popular_top,
    )

    df_pred.select(
        pl.col("user_id"),
        pl.col("item_id")
    ).write_csv(cfg_inference["eval_users_events_path"])
    logger.info("wrote submission to disk")


if __name__ == "__main__":
    main()
