from scipy.sparse import csr_matrix
from scipy import sparse
import numpy as np
import implicit
import os
import sys
import polars as pl
from loguru import logger
from pathlib import Path
import time
import psutil
from utils import load_config
from dataclasses import dataclass
from typing import Any, Dict
from stage import BaseStage
import json
import mlflow
import utils
from transform_data import parse_path_in


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
        show_weight * df_train["events_cnt_by_item_id_user_id"]
        + click_weight * df_train["clicks_cnt_by_item_id_user_id"]
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

    loss_history = []
    def store_loss(output_list):
        def inner(iteration, elapsed, loss):
            output_list.append(loss)
        return inner

    model = implicit.als.AlternatingLeastSquares(
        iterations=steps,
        factors=hidden_dim,
        random_state=random_state,
        calculate_training_loss=calculate_training_loss
    )
    model.fit_callback = store_loss(loss_history)
    model.fit(sparse_matrix, )
    logger.info("finish fit model")
    if calculate_training_loss and steps > 0:
        for i, loss in enumerate(loss_history):
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
        resume_from_run_id=None,
        artifacts_dir="mlflow/artifacts",
    ):
    if resume_from_run_id is not None:
        logger.info(f"inference will be resumed from run_id {resume_from_run_id}")
    else:
        resume_from_run_id = mlflow.active_run().info.run_id

    user_to_pred = user_to_pred["user_id"].sort()

    if fallback_strategy == "popular":
        assert popular_top is not None, "when using 'popular' fallback strategy, provide top popular items"
    
    if filter_already_liked_items:
        assert user_matrix is not None, "need saved user matrix for inference if filter_already_liked_items=True"

    user4pred_als_idx = np.array([user_id_to_index[i] for i in user_to_pred if i in user_id_to_index])
    user4pred_fallback = np.array([i for i in user_to_pred if i not in user_id_to_index])

    mlflow.log_param("users_pred_by_algo_cnt", len(user4pred_als_idx))
    mlflow.log_param("users_pred_by_algo_pct", len(user4pred_als_idx) / (len(user4pred_als_idx) + len(user4pred_fallback)))
    if fallback_strategy is not None:
        mlflow.log_param("users_pred_by_fallback_cnt", len(user4pred_fallback))
        mlflow.log_param("users_pred_by_fallback_pct", len(user4pred_fallback) / (len(user4pred_als_idx) + len(user4pred_fallback)))

    batches_dir = os.path.join("/project", artifacts_dir, resume_from_run_id, "inference_batches")  # todo rework for aws
    Path(batches_dir).mkdir(exist_ok=True, parents=True)
    client = mlflow.MlflowClient()

    last_batch = int(client.get_run(resume_from_run_id).data.tags.get("last_completed_batch", -1))

    total_batches = (len(user4pred_als_idx) + batch_size - 1) // batch_size
    for batch_id, start in enumerate(range(0, len(user4pred_als_idx), batch_size)):
        batch_path = os.path.join(batches_dir, f"batch_{batch_id}.npy")
        scores_path = os.path.join(batches_dir, f"scores_{batch_id}.npy")

        if batch_id <= last_batch:
            # assumes that batch_size is consistent across runs
            assert os.path.exists(batch_path), f"missing artifact for completed batch: {batch_path}"
            assert os.path.exists(scores_path), f"missing artifact for completed batch: {scores_path}"
            logger.info(f"skip inferencing batch {batch_id}, artifact exists")
            continue

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

        np.save(batch_path, batch_recs)
        np.save(scores_path, batch_scores)

        del batch_recs, batch_scores, batch_user_ids, batch_user_matrix

        logger.debug("deleted local vars explicitly")
        ram_report()
        logger.info(f"recommended for users {start}:{end} / {len(user4pred_als_idx)}")
        mlflow.log_artifact(str(batch_path), artifact_path="batch_results")
        mlflow.log_artifact(str(scores_path), artifact_path="batch_scores")
        mlflow.log_metric("last_completed_batch", batch_id, step=batch_id)
        mlflow.set_tag("last_completed_batch", batch_id)


    # assumes that batch_size is consistent across runs
    recommendations = np.vstack([np.load(os.path.join(batches_dir, f"batch_{i}.npy")) for i in range(total_batches)])
    scores = np.vstack([np.load(os.path.join(batches_dir, f"scores_{i}.npy")) for i in range(total_batches)])

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

    df_pred = df_pred.explode(['item_id', 'scores']).with_columns(
        pl.col("item_id").cast(pl.UInt32).alias("item_id"),
        pl.col("user_id").cast(pl.UInt32).alias("user_id"),
        pl.col("scores").cast(pl.Float64).alias("scores"),
    )

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

def collect_train_part_from_joined(fp):
    logger.info(f"collecting part {fp}...")
    return (
        utils.scan_parquet(fp)
        .select(
            pl.col("user_id"),
            pl.col("item_id"),
            pl.col("events_cnt_by_item_id_user_id"),
            pl.col("clicks_cnt_by_item_id_user_id")
        )
        .group_by(["user_id", "item_id"])
        .agg(pl.col("events_cnt_by_item_id_user_id").first(), pl.col("clicks_cnt_by_item_id_user_id").first()).collect()
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
    return pl.concat([collect_train_part_from_joined(fp) for fp in full_paths])


class ALSPreprocessStage(BaseStage):
    def assert_args_in_cfg(self, cfg):
        assert "in_artifacts" in cfg
        assert "train_events_path" in cfg["in_artifacts"]
        assert "eval_users_events_path" in cfg["in_artifacts"]
        assert "out_artifacts" in cfg
        assert "preprocessed_df_path" in cfg["out_artifacts"]

    def parse_kwargs(self):
        return {
            "train_events_path": self.cfg["in_artifacts"]["train_events_path"],
            "eval_users_events_path": self.cfg["in_artifacts"]["eval_users_events_path"],
        }

    def load_artifacts(self):
        return dict()

    def write_artifacts(self, run_result):
        super().write_artifacts(run_result)
        utils.sink_parquet(
            run_result,
            self.cfg["out_artifacts"]["preprocessed_df_path"],
            remove_local=self.remove_local,
            log_artifact=self.log_artifacts
        )


class ALSTrainStage(BaseStage):
    def assert_args_in_cfg(self, cfg):
        assert "in_artifacts" in cfg
        assert "train_data" in cfg["in_artifacts"]
        assert "eval_users" in cfg["in_artifacts"]

        assert "kwargs" in cfg
        assert "steps" in cfg["kwargs"]
        assert "hidden_dim" in cfg["kwargs"]
                    
        assert "out_artifacts" in cfg
        assert "artifacts_dir" in cfg["out_artifacts"]
        assert "model" in cfg["out_artifacts"]
        assert "item_id_to_index" in cfg["out_artifacts"]
        assert "user_id_to_index" in cfg["out_artifacts"]

    def parse_kwargs(self):
        return self.cfg["kwargs"]

    def load_artifacts(self):

        def file_parts_from_paths(paths_in):
            if isinstance(paths_in, str):
                paths_in = [paths_in]

            parts = []
            for path_in in paths_in:
                dir_in, part_filenames = parse_path_in(path_in)
                for part_filename in part_filenames:
                    logger.debug(f"scanning {part_filename} from {dir_in}...")
                    read_path = os.path.join(dir_in, part_filename)
                    parts.append(read_path)
            return parts

        return {
            "df_train": utils.read_parquet(file_parts_from_paths(self.cfg["in_artifacts"]["train_data"])),
            "user_to_pred": utils.read_csv(file_parts_from_paths(self.cfg["in_artifacts"]["eval_users"])),
        }

    def write_artifacts(self, run_result):
        out_artifacts = self.cfg["out_artifacts"]
        make_mlflow_subdirs = self.cfg["out_artifacts"].get("make_mlflow_artifacts_subdir", False)
        if make_mlflow_subdirs:
            mlflow_run = mlflow.active_run()
            run_id, exp_name = mlflow_run.info.run_id, mlflow.get_experiment(mlflow_run.info.experiment_id).name
            out_artifacts["artifacts_dir"] = os.path.join(out_artifacts["artifacts_dir"], exp_name, run_id, "")

        super().write_artifacts(run_result)

        artifacts_dir = self.cfg["out_artifacts"].get("artifacts_dir", "")

        for artifact_key in ["model", "item_id_to_index", "user_id_to_index", "user_matrix", "popular_top"]:
            if artifact_key in out_artifacts:
                out_artifacts[artifact_key] = os.path.join(artifacts_dir, out_artifacts[artifact_key])

        utils.save_artifact(run_result.model, out_artifacts["model"])

        utils.save_artifact({int(k): int(v) for k, v in run_result.item_id_to_index.items()}, out_artifacts["item_id_to_index"])
        utils.save_artifact({int(k): int(v) for k, v in run_result.user_id_to_index.items()}, out_artifacts["user_id_to_index"])
        
        if self.cfg["kwargs"].get("make_popular_top", False):
            utils.sink_parquet(
                run_result.popular_top,
                out_artifacts["popular_top"],
                remove_local=self.remove_local,
                log_artifact=self.log_artifacts
            )
        
        if self.cfg["kwargs"].get("make_user_matrix", False):
            utils.save_artifact(run_result.user_matrix, out_artifacts["user_matrix"])


class ALSInferenceStage(BaseStage):
    def assert_args_in_cfg(self, cfg):
        assert "in_artifacts" in cfg
        assert "eval_users" in cfg["in_artifacts"]
        assert "model" in cfg["in_artifacts"]
        assert "item_id_to_index" in cfg["in_artifacts"]
        assert "user_id_to_index" in cfg["in_artifacts"]
        assert "user_matrix" in cfg["in_artifacts"] or not cfg["kwargs"].get("filter_already_liked_items", False)
        assert "popular_top" in cfg["in_artifacts"] or not cfg["kwargs"].get("fallback_strategy") == "popular"

        assert "out_artifacts" in cfg
        assert "submission" in cfg["out_artifacts"]

        assert "make_mlflow_artifacts_subdir" not in cfg["out_artifacts"] or "artifacts_dir" in cfg["out_artifacts"]
        

    def load_artifacts(self):
        in_artifacts = self.cfg["in_artifacts"]
        artifacts_dir = self.cfg["in_artifacts"].get("artifacts_dir", "")

        artifacts_run_id = self.cfg["in_artifacts"].get("artifacts_run_id", "")
        artifacts_experiment_name = self.cfg["in_artifacts"].get("artifacts_experiment_name", "")

        for artifact_key in ["model", "item_id_to_index", "user_id_to_index", "user_matrix", "popular_top"]:
            if artifact_key in in_artifacts:
                in_artifacts[artifact_key] = os.path.join(artifacts_dir, artifacts_experiment_name, artifacts_run_id, in_artifacts[artifact_key])

        item_id_to_index = utils.load_artifact(in_artifacts["item_id_to_index"])
        user_id_to_index = utils.load_artifact(in_artifacts["user_id_to_index"])
        item_id_to_index, user_id_to_index = {int(k): v for k, v in item_id_to_index.items()}, {int(k): v for k, v in user_id_to_index.items()}

        return {
            "user_to_pred": utils.read_csv(in_artifacts["eval_users"]),
            "model": utils.load_artifact(in_artifacts["model"]),
            "item_id_to_index": item_id_to_index,
            "user_id_to_index": user_id_to_index,
            "user_matrix": utils.load_artifact(in_artifacts["user_matrix"]) if "user_matrix" in in_artifacts else None,
            "popular_top": utils.read_parquet(in_artifacts["popular_top"]) if "popular_top" in in_artifacts else None
        }

    def parse_kwargs(self):
        return self.cfg["kwargs"]
    
    def write_artifacts(self, run_result):
        make_mlflow_subdirs = self.cfg["out_artifacts"].get("make_mlflow_artifacts_subdir", False)

        out_artifacts = self.cfg["out_artifacts"]
        if make_mlflow_subdirs:
            mlflow_run = mlflow.active_run()
            run_id, exp_name = mlflow_run.info.run_id, mlflow.get_experiment(mlflow_run.info.experiment_id).name
            out_artifacts["artifacts_dir"] = os.path.join(out_artifacts["artifacts_dir"], exp_name, run_id, "")

        super().write_artifacts(run_result)

        if "artifacts_dir" in out_artifacts:
            artifacts_dir = out_artifacts["artifacts_dir"]
            out_artifacts["submission"] = os.path.join(artifacts_dir, out_artifacts["submission"])
            out_artifacts["submission_w_scores"] = os.path.join(artifacts_dir, out_artifacts["submission_w_scores"])

        utils.write_csv(
            run_result.select(pl.col("user_id"), pl.col("item_id")),
            out_artifacts["submission"],
            remove_local=self.remove_local,
            log_artifact=self.log_artifacts
        )

        utils.write_csv(
            run_result.select(pl.col("user_id"), pl.col("item_id"), pl.col("scores")),
            out_artifacts["submission_w_scores"],
            remove_local=self.remove_local,
            log_artifact=self.log_artifacts
        )



def main():
    assert len(sys.argv) == 4, "please provide a path to yaml config as arguments, (preprocess, training, inference)"
    preprocess_config_path, train_config_path, inference_config_path = sys.argv[1], sys.argv[2], sys.argv[3]
    preprocess_cfg = load_config(preprocess_config_path)["preprocessing"]
    train_cfg = load_config(train_config_path)["training"]
    inference_cfg = load_config(inference_config_path)["inference"]

    logger.info("starting pipeline...")

    with mlflow.start_run(run_name="als_pipeline"):
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
