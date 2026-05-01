import polars as pl
import os
import argparse
from pathlib import Path
from loguru import logger
from datetime import datetime
from debug_constants import DEBUG_ARGV_MAKE_TRAIN
import yaml

PRED_OPS = {
    "<":  lambda col, val: col < val,
    ">":  lambda col, val: col > val,
    "<=": lambda col, val: col <= val,
    ">=": lambda col, val: col >= val,
    "==": lambda col, val: col == val,
}

DEFAULT_SYNTH_THRESHOLD = "2026-04-08T00:00:00"

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to YAML config file.",
    )

    return parser.parse_args(argv)

def make_predicate(col_name, val_range_item):
    pred, val = val_range_item
    if pred not in PRED_OPS:
        raise ValueError(f"Unknown predicate '{pred}' for column '{col_name}'")
    return PRED_OPS[pred](pl.col(col_name), val)

def transform_date_to_ms(cfg):
    cfg["filters"]["timestamp"] = {
        op: int(datetime.fromisoformat(val).timestamp() * 1000) for op, val in cfg["filters"]["date_thr"]
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

    filters = make_filters(cfg)

    pl.scan_parquet(filename_in).filter(*filters).sink_parquet(out_path)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    filename_in = cfg["files"]["in"]
    filename_out = cfg["files"]["out"]

    assert os.path.isfile(filename_in) == os.path.isfile(filename_out)

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