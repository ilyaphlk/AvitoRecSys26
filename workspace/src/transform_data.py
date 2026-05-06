import polars as pl
import os
import argparse
from pathlib import Path
from loguru import logger
from datetime import datetime
from debug_constants import DEBUG_ARGV_MAKE_TRAIN, ARGV_MAKE_TRAIN_SEPARATE
from utils import load_config, parse_args

DEFAULT_SYNTH_THRESHOLD = "2026-04-08T00:00:00"
CONTACT_EIDS = [0, 2, 4, 5, 6, 8, 9, 11, 14, 15, 16]


PRED_OPS = {
    "<":  lambda col, val: col < val,
    ">":  lambda col, val: col > val,
    "<=": lambda col, val: col <= val,
    ">=": lambda col, val: col >= val,
    "==": lambda col, val: col == val,
    "!=": lambda col, val: col != val,
    "is_in": lambda col, val: col.is_in(val)
}

AGG_FUNCS = {
    "count":    lambda col, **kw: col.count(),
    "n_unique": lambda col, **kw: col.n_unique(),
    "sum":      lambda col, **kw: col.sum(),
    "mean":     lambda col, **kw: col.mean(),
    "min":      lambda col, **kw: col.min(),
    "max":      lambda col, **kw: col.max(),
    "first":    lambda col, **kw: col.first(),
    "last":     lambda col, **kw: col.last(),
    "unique":   lambda col, **kw: col.unique(),
    "count_if": lambda col, val_range, **kw: pl.all_horizontal(
        [make_predicate(col, val_range_item) for val_range_item in val_range.items()]).sum()
}

def make_agg_expr(agg_item: dict, keys: list[str]) -> pl.Expr:
    col_name = agg_item["col"]
    func = agg_item["func"]
    alias = agg_item.get("alias", f"{func}_{col_name}_by_{'_'.join(keys)}")
    condition = agg_item.get("condition", None)

    expr = pl.col(col_name)
    if "filters" in agg_item:
        filters = []
        for filter in agg_item["filters"]:
            for filter_col_name, val_range in filter.items():
                for val_range_item in val_range.items():
                    filters.append(make_predicate(pl.col(filter_col_name), val_range_item))
        expr = expr.filter(*filters)

    if func not in AGG_FUNCS:
        raise ValueError(f"Unknown aggregation function '{func}'")

    return AGG_FUNCS[func](expr, val_range=condition).alias(alias)

def make_aggregations(df: pl.LazyFrame, cfg: dict) -> dict[tuple[str], pl.LazyFrame]:
    if "features" not in cfg or "aggregations" not in cfg["features"]:
        return dict()

    agg_frames = dict()
    for agg_block in cfg["features"]["aggregations"]:
        keys = agg_block["group_by"]
        exprs = [make_agg_expr(item, keys) for item in agg_block["agg"]]
        agg_frames[tuple(keys)] = df.group_by(keys).agg(exprs)

    return agg_frames

def make_predicate(expr, val_range_item):
    pred, val = val_range_item
    if pred not in PRED_OPS:
        raise ValueError(f"Unknown predicate '{pred}'")
    return PRED_OPS[pred](expr, val)

def transform_date_to_ms(cfg):
    cfg["filters"]["timestamp"] = {
        op: int(datetime.fromisoformat(val).timestamp() * 1000) for op, val in cfg["filters"]["date_thr"].items()
    }
    cfg["filters"].pop("date_thr")

def make_filters(cfg):
    if "filters" not in cfg:
        return dict()

    if "date_thr" in cfg["filters"]:
        transform_date_to_ms(cfg)

    filters = dict()
    for col_name, val_range in cfg["filters"].items():
        for val_range_item in val_range.items():
            filters[col_name] = make_predicate(pl.col(col_name), val_range_item)
    return filters

def apply_filters(df, filter_expressions):
    return df.filter(*filter_expressions)

def make_train(cfg, filename_in, filename_out):
    out_path = Path(filename_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pl.scan_parquet(filename_in)

    agg_frames = make_aggregations(df, cfg)
    filters = make_filters(cfg)

    collected_agg_frames = dict()
    for keys, agg_frame in agg_frames.items():
        valid_filters = {fname: f for fname, f in filters.items() if fname in agg_frame.collect_schema().names()}
        agg_frame = agg_frame.filter(*valid_filters.values())
        collected_agg_frames[keys] = agg_frame.collect()

    for keys, agg_frame in collected_agg_frames.items():
        df = df.join(agg_frame.lazy(), on=keys, how='semi')

    for keys, agg_frame in collected_agg_frames.items():
        df = df.join(agg_frame.lazy(), on=keys, how='inner')

    df = df.filter(*filters.values())

    df.sink_parquet(out_path)


def main():
    args = parse_args(ARGV_MAKE_TRAIN_SEPARATE)
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
        part_filenames = sorted(list(filter(lambda fn: fn.startswith("part_"), os.listdir(filename_in))))
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
