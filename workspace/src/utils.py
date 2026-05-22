import polars as pl
import yaml
import argparse
import os
import boto3
import io
from loguru import logger
from pathlib import Path
import mlflow
from enum import Enum
import glob
import fnmatch
from typing import Any
import re
import numpy as np
import scipy.sparse as sparse
import json
import torch
from implicit.als import AlternatingLeastSquares


STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local")  # "local" or "s3"
S3_BUCKET = os.getenv("S3_BUCKET", None)
AWS_DEFAULT_REGION=os.getenv("AWS_DEFAULT_REGION", None)
LOCAL_DATA_DIR = os.getenv("LOCAL_DATA_DIR", "/project/data")
S3_DATA_DIR = os.getenv("S3_DATA_DIR", "data")

_s3_client = None

def get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3", region_name=AWS_DEFAULT_REGION)
    return _s3_client

def parse_partition_args(yaml_partition_args=None):
    def transform_key(part):
        return tuple(sorted(part)) if isinstance(part, (list, tuple)) else part
    return {transform_key(elem["part"]): elem["args"] for elem in yaml_partition_args} if yaml_partition_args else None

def sink_with_partition(df: pl.LazyFrame, root_path: str, key: str, mod=100, prefix="part_"):
    """
        df: a dataframe to sink
        root_path: directory to sink to
        key: integer column to partition by
        mod: integer modulus, each partition id is a pl.col(key) % mod
        prefix: prefix for each filename
    """
    n_digits = len(str(mod - 1)) + 1
    alias = "__mod"

    written_files = set()

    def fp_provider(fp_args: pl.FileProviderArgs):
        part_val = fp_args.partition_keys[alias].item()
        full_path = f"{prefix}{part_val:0{n_digits}d}.parquet"
        written_files.add(full_path)
        logger.debug(f"called fp_provider, returning {full_path}...")
        return full_path

    df.with_columns(
        (pl.col(key) % mod).alias(alias)
    ).sink_parquet(
        pl.PartitionBy(
            root_path,
            key = alias,
            include_key=False,
            #file_path_provider=lambda part_id, _: f"{prefix}{part_id[alias]:0{n_digits}d}.parquet"
            file_path_provider=fp_provider
        )
    )

    return sorted(list(written_files))

def sink_parquet(
        df: pl.LazyFrame | pl.DataFrame,
        path: str,
        remove_local=True,
        log_artifact=False,
        partition_args: dict[str, Any] | None = None,
    ):
    """
        path is relative, e.g. 'features/train.parquet'
        if partition_args is not None, then path must be a dir
    """
    assert path_type(path) == PathType.IS_FILE or partition_args is not None
    df = df.lazy() if isinstance(df, pl.DataFrame) else df

    local_path = os.path.join(LOCAL_DATA_DIR, path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    dir_out, written_files = None, None
    if path_type(local_path) == PathType.IS_FILE:
        df.sink_parquet(local_path)
        dir_out, written_files = Path(local_path).parent, [Path(local_path).name]
    else:
        written_files = sink_with_partition(df, local_path, **partition_args)
        dir_out = local_path

    logger.debug(f"sink_parquet got {path} as arg and wrote to: {written_files}")

    if log_artifact:
        for f_out in written_files:
            mlflow.log_artifact(os.path.join(dir_out, f_out))
    if STORAGE_BACKEND == "s3":
        for f_out in written_files:
            full_path = os.path.join(dir_out, f_out)
            s3_path = os.path.join(S3_DATA_DIR, path if path_type(path) == PathType.IS_DIR else Path(path).parent, f_out)
            get_s3_client().upload_file(full_path, S3_BUCKET, s3_path)
            if remove_local:
                os.remove(full_path)

def read_parquet(path: str | list[str]) -> pl.DataFrame:
    if isinstance(path, str):
        path = [path]

    if STORAGE_BACKEND == "s3":
        s3_prefix = "s3://{S3_BUCKET}/{S3_DATA_DIR}/{path}"
        w_prefix = [s3_prefix.format(S3_BUCKET=S3_BUCKET, S3_DATA_DIR=S3_DATA_DIR, path=elem) for elem in path]
        return pl.read_parquet(w_prefix, storage_options={"aws_region": AWS_DEFAULT_REGION})
    else:
        return pl.read_parquet([os.path.join(LOCAL_DATA_DIR, elem) for elem in path])


def scan_parquet(path: str | list[str]) -> pl.LazyFrame:
    if isinstance(path, str):
        path = [path]

    if STORAGE_BACKEND == "s3":
        s3_prefix = "s3://{S3_BUCKET}/{S3_DATA_DIR}/{path}"
        w_prefix = [s3_prefix.format(S3_BUCKET=S3_BUCKET, S3_DATA_DIR=S3_DATA_DIR, path=elem) for elem in path]
        return pl.scan_parquet(w_prefix, storage_options={"aws_region": AWS_DEFAULT_REGION})
    else:
        return pl.scan_parquet([os.path.join(LOCAL_DATA_DIR, elem) for elem in path])


def write_csv(df: pl.DataFrame, path: str, remove_local=True, log_artifact=False):
    """path is relative, e.g. 'features/train.csv'"""
    full_path = os.path.join(LOCAL_DATA_DIR, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    df.write_csv(full_path)
    if log_artifact:
        mlflow.log_artifact(full_path)
    if STORAGE_BACKEND == "s3":
        get_s3_client().upload_file(full_path, S3_BUCKET, f"{S3_DATA_DIR}/{path}")
        if remove_local:
            os.remove(full_path)

def read_csv(path: str | list[str]) -> pl.DataFrame:
    if isinstance(path, str):
        path = [path]

    if STORAGE_BACKEND == "s3":
        s3_prefix = "s3://{S3_BUCKET}/{S3_DATA_DIR}/{path}"
        w_prefix = [s3_prefix.format(S3_BUCKET=S3_BUCKET, S3_DATA_DIR=S3_DATA_DIR, path=elem) for elem in path]
        return pl.read_csv(w_prefix)
    else:
        return pl.read_csv([os.path.join(LOCAL_DATA_DIR, elem) for elem in path])

def als_save(obj: Any, path: str):
    """
        helper function to remove the mandatory .npz file extension upon saving the file.
    """
    obj.save(path)
    os.rename(f"{path}.npz", path)

def als_load(path):
    """
        add .npz extension to the .als model filename so that the ALS loader does not freak out
    """
    pnpz = f"{path}.npz"
    os.rename(path, pnpz)
    model = AlternatingLeastSquares().load(pnpz)
    os.rename(pnpz, path)
    return model

def save_artifact(obj: Any, path: str, remove_local: bool = True, log_artifact: bool = False):
    """
    path is relative, e.g. 'models/rec.keras' or 'data/matrix.npz'
    Dispatches serialization by file extension, then uploads to S3 if configured.
    """
    local_path = os.path.join(LOCAL_DATA_DIR, path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    ext = Path(path).suffix.lower()
    _WRITERS = {
        ".npy":   lambda o, p: np.save(p, o),
        ".npz":   lambda o, p: sparse.save_npz(p, o),
        ".json":  lambda o, p: Path(p).write_text(json.dumps(o)),
        ".pt":    lambda o, p: torch.save(o, p),
        ".als":   lambda o, p: als_save(o, p)
    }

    writer = _WRITERS.get(ext)
    if writer is None:
        raise ValueError(f"No writer registered for extension '{ext}'")
    writer(obj, local_path)

    if log_artifact:
        mlflow.log_artifact(local_path)

    if STORAGE_BACKEND == "s3":
        s3_path = os.path.join(S3_DATA_DIR, path)
        get_s3_client().upload_file(local_path, S3_BUCKET, s3_path)
        if remove_local:
            os.remove(local_path)

def load_artifact(path: str) -> Any:
    """
    path is relative, e.g. 'models/rec.keras' or 'data/matrix.npz'
    If S3 backend, downloads first, then deserializes by extension.
    """
    local_path = os.path.join(LOCAL_DATA_DIR, path)

    if STORAGE_BACKEND == "s3":
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3_path = os.path.join(S3_DATA_DIR, path)
        get_s3_client().download_file(S3_BUCKET, s3_path, local_path)

    ext = Path(path).suffix.lower()
    _READERS = {
        ".npy":   lambda p: np.load(p, allow_pickle=False),
        ".npz":   lambda p: sparse.load_npz(p),
        ".json":  lambda p: json.loads(Path(p).read_text()),
        ".pt":    lambda p: torch.load(p, weights_only=True),
        ".als":   lambda p: als_load(p)
    }

    reader = _READERS.get(ext)
    if reader is None:
        raise ValueError(f"No reader registered for extension '{ext}'")
    return reader(local_path)

def is_dirlike(path):
    return os.path.split(path)[-1] == ""

class PathType(Enum):
    IS_FILE = 0
    IS_DIR = 1
    IS_GLOB = 2

def path_type(path):
    if is_dirlike(path):
        return PathType.IS_DIR
    if glob.has_magic(path):
        return PathType.IS_GLOB
    return PathType.IS_FILE

def listdir(path: str, glob_pattern=None) -> list[str]:
    # todo handle case where path is file-like
    if STORAGE_BACKEND == "s3":
        logger.debug(f"listing s3 files in {S3_DATA_DIR}/{path}")
        contents = get_s3_client().list_objects_v2(Bucket=S3_BUCKET, Prefix=f"{S3_DATA_DIR}/{path}").get("Contents", [])
        logger.debug(f"returned contents: {contents}")
        part_filenames = [Path(obj["Key"]).name for obj in contents if (Path(obj["Key"]).parent == Path(S3_DATA_DIR, path))]  # filter only files, no subdirs
        if glob_pattern is not None:
            part_filenames = fnmatch.filter(part_filenames, glob_pattern)
        logger.debug(f"part_filenames: {part_filenames}")
        return part_filenames
    
    search_dir = os.path.join(LOCAL_DATA_DIR, path)
    return os.listdir(search_dir) if glob_pattern is None else glob.glob(glob_pattern, root_dir=search_dir)


def calc_metric(df_true, df_pred, df_users=None, df_item_verticals=None):
    """
        calculate mean recall across all eval (true) users
        df_true: user_id, item_id
        df_pred: user_id, item_id
        df_users: user_id, vertical_id
    """
    # assert that sets of users are the same
    assert set(df_true["user_id"]) == set(df_pred["user_id"]), "sets of users in eval and pred are different"

    # assert that all recommendations are unique for each user
    count_unique_preds = df_pred.group_by("user_id").agg(
        pl.col("item_id").len().alias("count"),
        pl.col("item_id").n_unique().alias("count_unique")
    )
    assert(all(count_unique_preds["count"] == count_unique_preds["count_unique"])), "pred has users with non-unique items"

    joined = df_true.join(df_pred, on=("user_id", "item_id"))
    total_by_user = df_true.group_by("user_id").agg(pl.len().alias("total_items"))
    retrieved_by_user = joined.group_by("user_id").agg(pl.len().alias("retrieved_items"))
    total_by_user = total_by_user.join(retrieved_by_user, on=("user_id"), how="left").fill_null(0)
    recall_by_user = total_by_user.select(
        pl.col("user_id"),
        (pl.col("retrieved_items") / pl.col("total_items")).alias("recall")
    )

    res = dict()

    res["overall_recall"] = recall_by_user["recall"].mean()

    if df_users is not None:
        recall_with_buckets = recall_by_user.join(df_users, on="user_id")
        res["per_bucket_recall"] = recall_with_buckets.group_by("bucket").agg(pl.col("recall").mean()).sort(by="bucket")

        if df_item_verticals is not None:
            filtered_items = df_item_verticals.select(["item_id", "vertical_id"]).join(df_pred.lazy(), on="item_id", how="semi")
            preds_with_verticals = df_pred.lazy().join(filtered_items, on="item_id", how="left")
            bucket_v_id_counts = (
                preds_with_verticals
                .join(df_users.lazy(), on="user_id", how="left")
                .group_by(["bucket", "vertical_id"]).agg(pl.len().alias("cnt"))
                .collect()
            )
            total_by_bucket = bucket_v_id_counts.group_by("bucket").agg(pl.col("cnt").sum().alias("total"))
            total_by_v_id = bucket_v_id_counts.group_by("vertical_id").agg(pl.col("cnt").sum().alias("total"))

            bucket_v_id_counts_norm_by_bucket = bucket_v_id_counts.join(
                total_by_bucket,on="bucket", how="left"
            ).select(
                pl.col("bucket"),
                pl.col("vertical_id"),
                (100. * pl.col("cnt") / pl.col("total")).alias("pct")
            )

            bucket_v_id_counts_norm_by_v_id = bucket_v_id_counts.join(
                total_by_v_id,on="vertical_id", how="left"
            ).select(
                pl.col("bucket"),
                pl.col("vertical_id"),
                (100. * pl.col("cnt") / pl.col("total")).alias("pct")
            )

            confusion_matrix = bucket_v_id_counts.pivot(on="vertical_id", index="bucket", values="cnt")
            confusion_matrix_pct_by_bucket = bucket_v_id_counts_norm_by_bucket.pivot(on="vertical_id", index="bucket", values="pct")
            confusion_matrix_pct_by_v_id = bucket_v_id_counts_norm_by_v_id.pivot(on="bucket", index="vertical_id", values="pct")

            res["confusion_matrix"] = confusion_matrix.sort(by="bucket")
            res["confusion_matrix_pct_by_bucket"] = confusion_matrix_pct_by_bucket.sort(by="bucket")
            res["confusion_matrix_pct_by_v_id"] = confusion_matrix_pct_by_v_id.sort(by="vertical_id")

    return res


def check_submission(df_true_filename, df_pred_filename, df_users_filename=None, df_item_verticals_filename=None):
    df_true = read_csv(df_true_filename)
    df_pred = read_csv(df_pred_filename)
    df_users = read_csv(df_users_filename) if df_users_filename is not None else None
    df_item_verticals = scan_parquet(df_item_verticals_filename) if df_item_verticals_filename is not None else None

    return calc_metric(df_true, df_pred, df_users, df_item_verticals)


def resolve_constants(cfg: dict, constants: dict = None) -> dict:
    """Replace '$NAME' strings with their value from cfg['constants']."""
    constants = dict() if constants is None else constants
    pattern = r'\$\$(.*?)\$\$'

    def replace_with_const(match):
        const_name = match.group(1)
        if const_name not in constants:
            raise ValueError(f"Undefined constant '{const_name}'")
        return str(constants[const_name])

    def resolve(obj):
        if isinstance(obj, str) and obj.startswith("$") and not re.findall(pattern, obj):
            key = obj[1:]
            if key not in constants:
                raise ValueError(f"Undefined constant '{key}'")
            return constants[key]

        if isinstance(obj, str):
            return re.sub(pattern, replace_with_const, obj)

        if isinstance(obj, dict):
            return {k: resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [resolve(v) for v in obj]
        return obj

    return resolve(cfg)


def load_config(config_path: str, constants_path: str | list[str] = None) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    constants = cfg.get("constants", {})
    if constants_path is None:
        return resolve_constants(cfg, constants)    

    if isinstance(constants_path, str):
        constants_path = [constants_path]

    for c in constants_path:
        with open(c) as f:
            constants_part = yaml.safe_load(f)["constants"]
            constants.update(constants_part)

    return resolve_constants(cfg, constants)

def deep_merge(base: dict, update: dict) -> dict:
    result = base.copy()
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def deep_merge_inplace(base: dict, update: dict):
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge_inplace(base[key], value)
        else:
            base[key] = value

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to YAML config file.",
    )

    return parser.parse_args(argv)
