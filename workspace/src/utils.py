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

def write_parquet(df: pl.DataFrame, path: str, remove_local=True, log_artifact=False):
    """path is relative, e.g. 'features/train.parquet'"""
    full_path = os.path.join(LOCAL_DATA_DIR, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    df.write_parquet(full_path)
    if log_artifact:
        mlflow.log_artifact(full_path)
    if STORAGE_BACKEND == "s3":
        get_s3_client().upload_file(full_path, S3_BUCKET, f"{S3_DATA_DIR}/{path}")
        if remove_local:
            os.remove(full_path)

def sink_parquet(df: pl.LazyFrame, path: str, remove_local=True, log_artifact=False):
    """path is relative, e.g. 'features/train.parquet'"""
    full_path = os.path.join(LOCAL_DATA_DIR, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    df.sink_parquet(full_path)
    if log_artifact:
        mlflow.log_artifact(full_path)
    if STORAGE_BACKEND == "s3":
        get_s3_client().upload_file(full_path, S3_BUCKET, f"{S3_DATA_DIR}/{path}")
        if remove_local:
            os.remove(full_path)

def read_parquet(path: str) -> pl.DataFrame:
    if STORAGE_BACKEND == "s3":
        return pl.read_parquet(f"s3://{S3_BUCKET}/{S3_DATA_DIR}/{path}",
                               storage_options={"aws_region": AWS_DEFAULT_REGION})
    else:
        return pl.read_parquet(os.path.join(LOCAL_DATA_DIR, path))


def scan_parquet(path: str) -> pl.LazyFrame:
    if STORAGE_BACKEND == "s3":
        return pl.scan_parquet(f"s3://{S3_BUCKET}/{S3_DATA_DIR}/{path}",
                               storage_options={"aws_region": AWS_DEFAULT_REGION})
    else:
        return pl.scan_parquet(os.path.join(LOCAL_DATA_DIR, path))

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

def read_csv(path: str) -> pl.DataFrame:
    if STORAGE_BACKEND == "s3":
        return pl.read_csv(f"s3://{S3_BUCKET}/{S3_DATA_DIR}/{path}",
                               #storage_options={"aws_region": AWS_DEFAULT_REGION}
                               )
    else:
        return pl.read_csv(os.path.join(LOCAL_DATA_DIR, path))

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


def calc_metric(df_true, df_pred):
    """
        calculate mean recall across all eval (true) users
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
    df_true_by_user = df_true.group_by("user_id").agg(pl.len().alias("total_items"))
    joined_by_user = joined.group_by("user_id").agg(pl.len().alias("retrieved_items"))
    df_true_by_user = df_true_by_user.join(joined_by_user, on=("user_id"), how="left").fill_null(0)
    df_true_by_user = df_true_by_user.select(
        (pl.col("retrieved_items") / pl.col("total_items")).alias("recall")
    )
    return df_true_by_user["recall"].mean()


def check_submission(df_true_filename, df_pred_filename):
    df_true = pl.read_csv(df_true_filename)
    df_pred = pl.read_csv(df_pred_filename)

    return calc_metric(df_true, df_pred)


def resolve_constants(cfg: dict) -> dict:
    """Replace '$NAME' strings with their value from cfg['constants']."""
    constants = cfg.get("constants", {})

    def resolve(obj):
        if isinstance(obj, str) and obj.startswith("$"):
            key = obj[1:]
            if key not in constants:
                raise ValueError(f"Undefined constant '{key}'")
            return constants[key]
        if isinstance(obj, dict):
            return {k: resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [resolve(v) for v in obj]
        return obj

    return resolve(cfg)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return resolve_constants(cfg)

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to YAML config file.",
    )

    return parser.parse_args(argv)