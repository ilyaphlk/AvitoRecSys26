import polars as pl
import os
import argparse
from pathlib import Path
from loguru import logger
from datetime import datetime
from debug_constants import DEBUG_ARGV_MAKE_TRAIN
import yaml


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

def make_train(filename_in, filename_out, threshold_date):
    threshold_ms = int(threshold_date.timestamp() * 1000)
    out_path = Path(filename_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pl.scan_parquet(filename_in).filter(
        pl.col("timestamp") < threshold_ms
    ).sink_parquet(out_path)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    filename_in = cfg["files"]["in"]
    filename_out = cfg["files"]["out"]
    threshold_date = datetime.fromisoformat(cfg["filters"]["date_thr"])

    assert os.path.isfile(filename_in) == os.path.isfile(filename_out)

    if os.path.isfile(filename_in) or "*" in filename_in:
        make_train(
            filename_in=filename_in,
            filename_out=filename_out,
            threshold_date=threshold_date
        )
    else:
        logger.info(f"Processing multiple files in {filename_in}..")
        part_filenames = list(filter(lambda fn: fn.startswith("part_"), os.listdir(filename_in)))
        newline = "\n"
        logger.info(f"Filenames to process:\n{newline.join(part_filenames)}")

        for part_filename in part_filenames:
            logger.info(f"{'#'*20}\nProcessing {part_filename}...\n")
            make_train(
                filename_in=os.path.join(filename_in, part_filename),
                filename_out=os.path.join(filename_out, part_filename),
                threshold_date=threshold_date,
            )


if __name__ == "__main__":
    main()