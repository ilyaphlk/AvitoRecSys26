import polars as pl
import os
import argparse
from pathlib import Path
from loguru import logger
from datetime import datetime
from debug_constants import DEBUG_ARGV_MAKE_TRAIN
import yaml

DEFAULT_SYNTH_THRESHOLD = "2026-04-08T00:00:00"
CONTACT_EIDS = [0, 2, 4, 5, 6, 8, 9, 11, 14, 15, 16]

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

PRED_OPS = {
    "<":  lambda col, val: col < val,
    ">":  lambda col, val: col > val,
    "<=": lambda col, val: col <= val,
    ">=": lambda col, val: col >= val,
    "==": lambda col, val: col == val,
    "is_in": lambda col, val: col.is_in(val)
}

AGG_FUNCS = {
    "count":    lambda col: col.count(),
    "n_unique": lambda col: col.n_unique(),
    "sum":      lambda col: col.sum(),
    "mean":     lambda col: col.mean(),
    "min":      lambda col: col.min(),
    "max":      lambda col: col.max(),
    "first":    lambda col: col.first(),
    "last":     lambda col: col.last(),
}

def make_agg_expr(agg_item: dict, keys: list[str]) -> pl.Expr:
    col_name = agg_item["col"]
    func = agg_item["func"]
    alias = agg_item.get("alias", f"{func}_{col_name}_by_{'_'.join(keys)}")

    expr = pl.col(col_name)
    if "filter" in agg_item:
        filter_expr = make_predicate(agg_item["filter"])
        expr = expr.filter(filter_expr)

    if func not in AGG_FUNCS:
        raise ValueError(f"Unknown aggregation function '{func}'")

    return AGG_FUNCS[func](expr).alias(alias)

def make_aggregations(df: pl.LazyFrame, cfg: dict) -> dict[tuple[str], pl.LazyFrame]:
    if "features" not in cfg or "aggregations" not in cfg["features"]:
        return dict()

    agg_frames = dict()
    for agg_block in cfg["features"]["aggregations"]:
        keys = agg_block["group_by"]
        exprs = [make_agg_expr(item, keys) for item in agg_block["agg"]]
        agg_frames[tuple(keys)] = df.group_by(keys).agg(exprs)

    return agg_frames

def make_predicate(col_name, val_range_item):
    pred, val = val_range_item
    if pred not in PRED_OPS:
        raise ValueError(f"Unknown predicate '{pred}' for column '{col_name}'")
    return PRED_OPS[pred](pl.col(col_name), val)

def transform_date_to_ms(cfg):
    cfg["filters"]["timestamp"] = {
        op: int(datetime.fromisoformat(val).timestamp() * 1000) for op, val in cfg["filters"]["date_thr"].items()
    }
    cfg["filters"].pop("date_thr")

def make_filters(cfg):
    if "filters" not in cfg:
        return []

    if "date_thr" in cfg["filters"]:
        transform_date_to_ms(cfg)

    filters = []
    for col_name, val_range in cfg["filters"].items():
        for val_range_item in val_range.items():
            filters.append(make_predicate(col_name, val_range_item))
    return filters

def apply_filters(df, filter_expressions):
    return df.filter(*filter_expressions)

def make_train(cfg, filename_in, filename_out):
    out_path = Path(filename_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pl.scan_parquet(filename_in)

    agg_frames = make_aggregations(df, cfg)

    for keys, agg_frame in agg_frames.items():
        df = df.join(agg_frame, on=keys)

    filters = make_filters(cfg)
    df = df.filter(*filters)

    df.sink_parquet(out_path)


def main():
    args = parse_args(DEBUG_ARGV_MAKE_TRAIN)
    cfg = load_config(args.config)["data"]

    filename_in = cfg["files"]["in"]
    filename_out = cfg["files"]["out"]

    assert os.path.isfile(filename_in) == os.path.isfile(filename_out) or not os.path.exists(filename_out)

    if os.path.isfile(filename_in) or "*" in filename_in:
        make_train(
            cfg=cfg,
            filename_in=filename_in,
            filename_out=filename_out,
        )
    else:
        logger.info(f"Processing multiple files in {filename_in}..")
        part_filenames = list(filter(lambda fn: fn.startswith("part_"), os.listdir(filename_in)))
        newline = "\n"
        logger.info(f"Filenames to process:\n{newline.join(part_filenames)}")

        for part_filename in part_filenames:
            logger.info(f"{'#'*20}\nProcessing {part_filename}...\n")
            make_train(
                cfg=cfg,
                filename_in=os.path.join(filename_in, part_filename),
                filename_out=os.path.join(filename_out, part_filename),
            )


if __name__ == "__main__":
    main()