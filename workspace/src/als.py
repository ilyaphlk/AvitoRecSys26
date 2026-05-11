from scipy.sparse import csr_matrix
from scipy import sparse
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
from stage import BaseStage
import json
import mlflow


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
        steps=10,
        hidden_dim=60,
        random_state=42,
        calculate_training_loss=True,
        top_size=160,
        make_popular_top=True,
        make_user_matrix=True
    ):
    user_to_pred = user_to_pred["user_id"]
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
        iterations=steps,
        factors=hidden_dim,
        random_state=random_state,
        calculate_training_loss=calculate_training_loss
    )
    model.fit(sparse_matrix, )
    logger.info("finish fit model")
    if calculate_training_loss:
        for i, loss in enumerate(model.training_loss):
            mlflow.log_metric("training_loss", loss, step=i)

    user4pred_als_idx = np.array([user_id_to_index[i] for i in user_to_pred if i in user_id_to_index])
    user_matrix = sparse_matrix[user4pred_als_idx] if make_user_matrix else None
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
    user_to_pred = user_to_pred["user_id"]

    if fallback_strategy == "popular":
        assert popular_top is not None, "when using 'popular' fallback strategy, provide top popular items"
    
    if filter_already_liked_items:
        assert user_matrix is not None, "need saved user matrix for inference if filter_already_liked_items=True"

    user4pred_als_idx = np.array([user_id_to_index[i] for i in user_to_pred if i in user_id_to_index])
    user4pred_fallback = np.array([i for i in user_to_pred if i not in user_id_to_index])

    mlflow.log_param("users_pred_by_algo_cnt", len(user4pred_als_idx))
    mlflow.log_param("users_pred_by_algo_pct", len(user4pred_als_idx) / (len(user4pred_als_idx) + len(user4pred_fallback)))
    if fallback_strategy is not None:
        mlflow.log_param("users_pred_by_fallback_cnt", len(user4pred_als_idx))
        mlflow.log_param("users_pred_by_fallback_pct", len(user4pred_fallback) / (len(user4pred_als_idx) + len(user4pred_fallback)))

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
            batch_user_matrix = csr_matrix((len(batch_user_ids), model.item_factors.shape[0]))
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
            'item_id': [[index_to_item_id[idx] for idx in row] for row in recommendations.tolist()],
            'user_id': [index_to_user_id[idx] for idx in user4pred_als_idx.tolist()],
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

def collect_train_part(fp):
    logger.info(f"collecting part {fp}...")
    return (
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

def make_train(train_events_path, eval_users_events_path):
    train_path = Path(train_events_path)
    if os.path.isdir(train_path):
        full_paths = [os.path.join(train_path, fn) for fn in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, fn))]
    else:
        full_paths = [train_path]
    logger.info(f"full paths to train parts: {full_paths}")
    full_paths = [eval_users_events_path] + sorted(full_paths)

    logger.info("concatenating collected parts..")
    return pl.concat([collect_train_part(fp) for fp in full_paths])


class ALSPreprocessStage(BaseStage):
    def assert_args_in_cfg(self):
        assert "in_artifacts" in self.cfg
        assert "train_events_path" in self.cfg["in_artifacts"]
        assert "eval_users_events_path" in self.cfg["in_artifacts"]
        assert "out_artifacts" in self.cfg
        assert "preprocessed_df_path" in self.cfg["out_artifacts"]

    def parse_kwargs(self):
        return {
            "train_events_path": self.cfg["in_artifacts"]["train_events_path"],
            "eval_users_events_path": self.cfg["in_artifacts"]["eval_users_events_path"],
        }

    def load_artifacts(self):
        return dict()

    def write_artifacts(self, run_result):
        super().write_artifacts(run_result)
        run_result.write_parquet(self.cfg["out_artifacts"]["preprocessed_df_path"])


class ALSTrainStage(BaseStage):
    def assert_args_in_cfg(self):
        assert "in_artifacts" in self.cfg
        assert "train_data" in self.cfg["in_artifacts"]
        assert "eval_users" in self.cfg["in_artifacts"]

        assert "kwargs" in self.cfg
        assert "steps" in self.cfg["kwargs"]
        assert "hidden_dim" in self.cfg["kwargs"]
                    
        assert "out_artifacts" in self.cfg
        assert "model" in self.cfg["out_artifacts"]
        assert "item_id_to_index" in self.cfg["out_artifacts"]
        assert "user_id_to_index" in self.cfg["out_artifacts"]

    def parse_kwargs(self):
        return self.cfg["kwargs"]

    def load_artifacts(self):
        return {
            "df_train": pl.read_parquet(self.cfg["in_artifacts"]["train_data"]),
            "user_to_pred": pl.read_csv(self.cfg["in_artifacts"]["eval_users"]),
        }

    def write_artifacts(self, run_result):
        super().write_artifacts(run_result)

        run_result.model.save(self.cfg["out_artifacts"]["model"])

        with open(self.cfg["out_artifacts"]["item_id_to_index"], "w") as f:
            json.dump({int(k): int(v) for k, v in run_result.item_id_to_index.items()}, f)
        with open(self.cfg["out_artifacts"]["user_id_to_index"], "w") as f:
            json.dump({int(k): int(v) for k, v in run_result.user_id_to_index.items()}, f)
        
        if self.cfg["kwargs"].get("make_popular_top", False):
            run_result.popular_top.write_parquet(self.cfg["out_artifacts"]["popular_top"])
        
        if self.cfg["kwargs"].get("make_user_matrix", False):
            sparse.save_npz(self.cfg["out_artifacts"]["user_matrix"], run_result.user_matrix)


class ALSInferenceStage(BaseStage):
    def assert_args_in_cfg(self):
        assert "in_artifacts" in self.cfg
        assert "eval_users" in self.cfg["in_artifacts"]
        assert "model" in self.cfg["in_artifacts"]
        assert "item_id_to_index" in self.cfg["in_artifacts"]
        assert "user_id_to_index" in self.cfg["in_artifacts"]
        assert "user_matrix" in self.cfg["in_artifacts"] or not self.cfg["kwargs"].get("filter_already_liked_items", False)
        assert "popular_top" in self.cfg["in_artifacts"] or not self.cfg["kwargs"].get("fallback_strategy") == "popular"

        assert "out_artifacts" in self.cfg
        assert "submission" in self.cfg["out_artifacts"]
        

    def load_artifacts(self):
        in_artifacts = self.cfg["in_artifacts"]
        with open(in_artifacts["item_id_to_index"]) as f_i2idx, open(in_artifacts["user_id_to_index"]) as f_u2idx:
            return {
                "user_to_pred": pl.read_csv(in_artifacts["eval_users"]),
                "model": implicit.als.AlternatingLeastSquares().load(in_artifacts["model"]),
                "item_id_to_index": json.load(f_i2idx, object_hook=lambda d: {int(k): v for k, v in d.items()}),
                "user_id_to_index": json.load(f_u2idx, object_hook=lambda d: {int(k): v for k, v in d.items()}),
                "user_matrix": sparse.load_npz(in_artifacts["user_matrix"]) if "user_matrix" in in_artifacts else None,
                "popular_top": pl.read_parquet(in_artifacts["popular_top"]) if "popular_top" in in_artifacts else None,
            }

    def parse_kwargs(self):
        return self.cfg["kwargs"]
    
    def write_artifacts(self, run_result):
        super().write_artifacts(run_result)
        run_result.select(
            pl.col("user_id"),
            pl.col("item_id")
        ).write_csv(self.cfg["out_artifacts"]["submission"])



def main():
    assert len(sys.argv) == 3, "please provide a path to yaml config as arguments, (training, inference)"
    train_config_path, inference_config_path = sys.argv[1], sys.argv[2]
    preprocess_cfg = load_config(train_config_path)["preprocessing"]
    train_cfg = load_config(train_config_path)["training"]
    inference_cfg = load_config(inference_config_path)["inference"]

    logger.info("starting pipeline...")

    preproc_stage = ALSPreprocessStage(preprocess_cfg, make_train)
    train_stage = ALSTrainStage(train_cfg, train)
    inference_stage = ALSInferenceStage(inference_cfg, inference)

    preproc_stage.run()
    logger.info("made train successfully.")

    train_stage.run()
    logger.info("trained model")

    inference_stage.run()
    logger.info("wrote submission to disk")


if __name__ == "__main__":
    main()
