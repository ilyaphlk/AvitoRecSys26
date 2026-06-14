"""
prepare_local_eval.py — competitor-facing template for building a local
validation set out of df_train.

The official `eval.csv` (private/public ground truth) is not shipped to
competitors. This script reproduces the official eval pipeline against a
synthetic threshold inside train.

Pipeline (constants below match the official v4 eval — DO NOT change them
unless you know what you're doing):

  synthetic_train := events with timestamp <  synth_threshold
  synthetic_eval  := events with timestamp >= synth_threshold + 12h gap

  Filters on synthetic_eval:
    1. eid in contact_eids.csv,
    2. item_id appears in synthetic_train with >= 2 unique users,
    3. (user_id, item_id) ∉ synthetic_train (novelty filter).

  Bucket sampling (5 vertical × 10k + 5 holdout × 10k):
    For each of the 5 named verticals, sample 10k users whose unique-train
    items in that vertical's mapped_vertical_id(s) cover >= 90% of their
    total. Then 5 stratified holdout pools of 10k from the remaining
    eligible users.

  Per-user target = first 160 chronologically-unique novel contact items.

Usage (only the dataset paths and where to write are configurable):
    python -m bin.datafest_2026_v2.prepare_local_eval \\
        --train data/train.parquet \\
        --item-features data/item_features.parquet \\
        --contact-eids data/contact_eids.csv \\
        --out data/local_eval.csv

Output:
  * local_eval.csv        — flat (user_id, item_id), score with calc_metric.py
  * local_eval_users.csv  — user_id → bucket map (per-bucket recall analysis)
"""

import argparse
from datetime import datetime, timedelta

import polars as pl
from loguru import logger

import os
from pathlib import Path

from utils import load_config
import sys
from stage import BaseStage
import mlflow
from transform_data import SequentialStage, parse_path_in

import utils

# ── Constants frozen by the official v4 eval spec ─────────────────────────
DEFAULT_SYNTH_THRESHOLD = "2026-04-08T00:00:00"  # 1 week before real threshold
GAP_HOURS = 12
K = 160
MIN_USERS_PER_ITEM = 2
QUOTA_PER_VERTICAL = 10_000
HOLDOUT_QUOTA = 10_000
N_HOLDOUT_BUCKETS = 5
VERTICAL_THRESHOLD = 0.9
# 5 named buckets: 4 single verticals + 1 concat of two related verticals.
# vertical_id values come from item_features.parquet's mapped_vertical_id column.
BUCKET_SPECS: tuple[tuple[str, frozenset[int]], ...] = (
    ("v0", frozenset({0})),
    ("v2", frozenset({2})),
    ("v3", frozenset({3})),
    ("v4", frozenset({4})),
    ("v57", frozenset({5, 7})),
)


def _build_candidates(
    train_path: str,
    contact_eids: list[int],
    threshold_ms: int,
    eval_start_ms: int,
    items_blacklist_path: str | None = None
) -> pl.DataFrame:
    logger.debug("start building candidates...")
    train = utils.scan_parquet(train_path)
    synth_train = train.filter(pl.col("timestamp") < threshold_ms)

    seen = synth_train.select(["user_id", "item_id"]).unique()  # for each user select item_ids already seen

    candidates = None
    if items_blacklist_path:
        candidates = (
            train.filter(pl.col("timestamp") >= eval_start_ms)  # select events from eval time range,
            .filter(pl.col("eid").is_in(contact_eids))          # only contact (target) ones
            .join(utils.scan_parquet(items_blacklist_path), on="item_id", how="anti")    # only items above popularity thr
            .join(seen, on=["user_id", "item_id"], how="anti")  # and only unseen by users from synth_train
            .collect()
        )
    else:
        train_items_2u = (                   # select only those items, that have at least MIN_USERS_PER_ITEM unique
            synth_train.group_by("item_id")  # users interacted with them. The list is unique by item_id
            .agg(pl.col("user_id").n_unique().alias("n_users"))
            .filter(pl.col("n_users") >= MIN_USERS_PER_ITEM)
            .select("item_id")
        )
        candidates = (
            train.filter(pl.col("timestamp") >= eval_start_ms)  # select events from eval time range,
            .filter(pl.col("eid").is_in(contact_eids))          # only contact (target) ones
            .join(train_items_2u, on="item_id", how="semi")    # only items above popularity thr
            .join(seen, on=["user_id", "item_id"], how="anti")  # and only unseen by users from synth_train
            .collect()
        )
    n_rows = candidates.height
    n_eligible_users = candidates['user_id'].n_unique()
    logger.info(
        f"Candidate events: {n_rows:,} rows, "
        f"{n_eligible_users:,} eligible users"
    )
    mlflow.log_metrics({"n_rows_eligible": n_rows, "n_users_eligible": n_eligible_users})
    return candidates


def _build_user_sample(
    train_path: str,
    item_features_path: str,
    threshold_ms: int,
    eligible_users: pl.DataFrame,
    vertical_quotas: dict[str, int] | None = None,
) -> pl.DataFrame:
    synth_train = utils.scan_parquet(train_path).filter(
        pl.col("timestamp") < threshold_ms
    )
    items_v = utils.scan_parquet(item_features_path).select(["item_id", "vertical_id"])  # for each item get its slice (vertical)

    logger.info("start making vertical stats...")
    user_vertical = (
        synth_train.select(["user_id", "item_id"])
        .unique()
        .join(items_v, on="item_id", how="inner")                # join vertical_id
        .group_by(["user_id", "vertical_id"])
        .agg(pl.len().alias("n_in_v"))                           # how many events in each vertical for each user
        .join(eligible_users.lazy(), on="user_id", how="inner")  # but only for users from candidate events
        .collect()
    )
    user_total = user_vertical.group_by("user_id").agg(          # count all events by user
        pl.col("n_in_v").sum().alias("n_total")
    )

    parts: list[pl.DataFrame] = []
    bucket_stats = dict()
    for bucket_name, vertical_ids in BUCKET_SPECS:
        logger.info(f"start processing bucket {bucket_name}")
        bucket_n = (
            user_vertical.filter(pl.col("vertical_id").is_in(list(vertical_ids)))  # select only events in current bucket
            .group_by("user_id")
            .agg(pl.col("n_in_v").sum().alias("n_in_bucket"))  # basically the same, need only for v57 support
        )

        focused = (                                    # select unique user_ids which have >= VERTICAL_THRESHOLD of all
            bucket_n.join(user_total, on="user_id")    # their events belong to the current bucket + sort them
            .with_columns((pl.col("n_in_bucket") / pl.col("n_total")).alias("frac"))
            .filter(pl.col("frac") >= VERTICAL_THRESHOLD)
            .select("user_id")
            .unique()
            .sort("user_id")
        )

        quota = vertical_quotas[bucket_name] if vertical_quotas is not None else QUOTA_PER_VERTICAL
        n = min(quota, focused.height)    # sample at max QUOTA_PER_VERTICAL users from focused
        seed = 42 + sum(int(v) for v in vertical_ids)
        sample = focused.sample(n=n, seed=seed)
        logger.info(
            f"  bucket {bucket_name} (verticals={sorted(vertical_ids)}): "
            f"{n} / {focused.height} eligible focused users (seed={seed})"
        )
        bucket_stats[bucket_name] = {
            "n_selected": n,
            "n_eligible": focused.height,
            "seed": seed
        }
        parts.append(sample.with_columns(pl.lit(bucket_name).alias("bucket")))

    bucketed = pl.concat([p.select("user_id") for p in parts]).unique()

    holdout_pool = (
        eligible_users.join(bucketed, on="user_id", how="anti").sort("user_id")
    )
    logger.info(
        f"Holdout pool: {holdout_pool.height:,} eligible users not in any vertical bucket"
    )

    for i in range(N_HOLDOUT_BUCKETS):  # iteratively select n random non-overlapping buckets from all the other users
        if holdout_pool.height == 0:
            break
        bucket_name = f"h{i}"
        quota = vertical_quotas[bucket_name] if vertical_quotas is not None else HOLDOUT_QUOTA
        n = min(quota, holdout_pool.height)
        seed = 100 + i
        sample = holdout_pool.sample(n=n, seed=seed)
        logger.info(f"  bucket {bucket_name}: {n} / {holdout_pool.height} (seed={seed})")
        bucket_stats[bucket_name] = {
            "n_selected": n,
            "n_eligible": holdout_pool.height,
            "seed": seed
        }
        parts.append(sample.with_columns(pl.lit(bucket_name).alias("bucket")))
        holdout_pool = holdout_pool.join(sample, on="user_id", how="anti").sort(
            "user_id"
        )

    mlflow.log_dict(bucket_stats, "stats/bucket_stats.json")

    combined = pl.concat(parts)
    n_unique = combined["user_id"].n_unique()
    logger.info(f"Total sampled: {combined.height:,} rows, {n_unique:,} unique users")
    mlflow.log_metrics({"total_sampled": combined.height, "total_unique_users": n_unique})
    if n_unique != combined.height:
        logger.warning(f"{combined.height - n_unique} duplicate user→bucket assignments")
    return combined  # contains only `user_id` and `bucket` columns


def _build_eval_rows(candidates: pl.DataFrame, sampled: pl.DataFrame) -> pl.DataFrame:
    """
        return top-K unique items from candidates dataframe, filtered by user_id from sampled
        top-K by being earliest by timestamp
    """
    return (
        candidates.join(sampled.select("user_id"), on="user_id", how="inner")
        .sort(["user_id", "timestamp"])
        .group_by("user_id", maintain_order=True)
        .agg(pl.col("item_id").unique(maintain_order=True).head(K))
        .explode("item_id")
    )


def prepare_local_eval(
    train_path: str,
    item_features_path: str,
    contact_eids_path: str,
    synth_threshold: str = DEFAULT_SYNTH_THRESHOLD,
    write_train_part: bool = False,
    items_blacklist_path: str | None = None,
    vertical_quotas: dict[str, int] | None = None,
):
    """
        train_path - path to a single train file to split
        item_features_path - path to a file with item features (need this for vertical_id)
        contact_eids_path - path to a file with contact event types (to filter out only contact events)
        synth_threshold - train cutoff dt to split by (in the YYYY-MM-DDTHH:mm:ss format)

        writes <out_path>.csv - pairs of user_id, item_id (ground truth); <out_path>_users.csv (only users)
    """
    threshold_date = datetime.fromisoformat(synth_threshold)
    eval_start_date = threshold_date + timedelta(hours=GAP_HOURS)
    threshold_ms = int(threshold_date.timestamp() * 1000)  # multiply by 1000, so it is consistent with training data (in ms)
    eval_start_ms = int(eval_start_date.timestamp() * 1000)
    logger.info(
        f"synth_train: timestamp < {threshold_ms} ({threshold_date}) | "
        f"synth_eval: timestamp >= {eval_start_ms} ({eval_start_date}, gap {GAP_HOURS}h)"
    )

    contact_eids = utils.read_csv(contact_eids_path).get_column("mapped_eid").to_list()  # materialize only contact eids

    candidates = _build_candidates(  # returns events from train eligible for eval (by date and popularity thr, unseen in synth_train)
        train_path, contact_eids, threshold_ms, eval_start_ms, items_blacklist_path
    )
    eligible_users = candidates.select("user_id").unique().sort("user_id")  # unique users from that

    sampled = _build_user_sample(  # user_id, bucket - df sampled by bucket
        train_path, item_features_path, threshold_ms, eligible_users, vertical_quotas
    )
    eval_df = _build_eval_rows(candidates, sampled)  # ground truth

    res = {
        "sampled": sampled,
        "ground_truth": eval_df,
        "eval_user_events": None,
        "other_user_events": None,
    }

    if write_train_part:
        # write events for selected sample / all other users separately
        eval_user_events = (
            utils.scan_parquet(train_path)
            .join(eval_df.lazy(), on="user_id", how="semi")
            .filter(pl.col("timestamp") < threshold_ms)
        )
        res["eval_user_events"] = eval_user_events

        other_user_events = (
            utils.scan_parquet(train_path)
            .join(eval_df.lazy(), on="user_id", how="anti")
            .filter(pl.col("timestamp") < threshold_ms)
        )
        res["other_user_events"] = other_user_events

    return res


class PrepareLocalEvalStage(BaseStage):
    def assert_args_in_cfg(self, cfg):
        assert "in_artifacts" in self.cfg
        assert "filename_in" in self.cfg["in_artifacts"]
        assert "item_features_path" in self.cfg["in_artifacts"]
        assert "contact_eids_path" in self.cfg["in_artifacts"]

        assert "out_artifacts" in self.cfg
        assert "filename_out" in self.cfg["out_artifacts"]

    def load_artifacts(self):
        artifacts = {
            "train_path": self.cfg["in_artifacts"]["filename_in"],
            "item_features_path": self.cfg["in_artifacts"]["item_features_path"],
            "contact_eids_path": self.cfg["in_artifacts"]["contact_eids_path"],
        }
        # mlflow.log_artifact(artifacts["train_path"])
        # mlflow.log_artifact(artifacts["item_features_path"])
        # mlflow.log_artifact(artifacts["contact_eids_path"])

        return artifacts

    def parse_kwargs(self):
        return self.cfg.get("kwargs", dict())
    
    def write_artifacts(self, run_result):
        super().write_artifacts(run_result)

        out_path = Path(self.cfg["out_artifacts"]["filename_out"]).with_suffix(".csv")
        utils.write_csv(run_result["ground_truth"], out_path, remove_local=False, log_artifact=True)
        n_rows = run_result["ground_truth"].height
        n_unique_users = run_result["ground_truth"]['user_id'].n_unique()
        logger.info(
            f"{out_path}: {n_rows:,} rows, "
            f"{n_unique_users:,} users with >=1 target"
        )

        users_path = Path(os.path.join(out_path.parent, "users", out_path.name))
        utils.write_csv(run_result["sampled"], users_path, remove_local=False, log_artifact=True)
        logger.info(f"User → bucket map saved to {users_path}")

        if self.kwargs.get("write_train_part", False):
            def write_user_events(key):
                """
                    key in ["eval_user_events", "other_user_events"]
                """
                subdirs = {"eval_user_events": key, "other_user_events": os.path.join("..", "train", "other_user_events")}
                user_events_path = Path(
                    os.path.join(out_path.parent, subdirs[key], out_path.name)
                ).with_suffix(".pq")

                utils.sink_parquet(run_result[key], user_events_path, remove_local=False, log_artifact=True)
                n_rows_user_events = run_result[key].select(pl.len()).collect().item()
                n_unique_items_user_events = run_result[key].select(pl.col('item_id').n_unique()).collect().item()
                logger.info(f"{user_events_path}: {n_rows_user_events} rows, {n_unique_items_user_events} unique items.")
                mlflow.log_metrics({f"n_rows_{key}": n_rows_user_events, f"n_unique_items_{key}": n_unique_items_user_events})

            write_user_events("eval_user_events")
            write_user_events("other_user_events")

    def _func(self, *args, **kwargs):
        return prepare_local_eval(*args, **kwargs)


if __name__ == "__main__":
    assert len(sys.argv) == 2, "please provide a path to yaml config"
    cfg_path = sys.argv[1]
    # cfg_path = "/project/workspace/config/data/eval/debug_stage.yml"

    cfg = load_config(cfg_path)["prepare_eval"]

    mlflow.set_tracking_uri(f"http://localhost:{os.getenv('MLFLOW_PORT', '5000')}")
    mlflow.set_experiment("prepare_local_eval")

    stage = SequentialStage(cfg, PrepareLocalEvalStage, prepare_local_eval)
    stage.run()