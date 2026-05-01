import polars as pl
import os
import argparse
from pathlib import Path
from loguru import logger
from debug_constants import DEBUG_ARGV_MAKE_TRAIN
from datetime import datetime, timedelta

DEFAULT_SYNTH_THRESHOLD = "2026-04-08T00:00:00" 

def make_train(filename_in, filename_out, threshold_date):
    threshold_ms = int(threshold_date.timestamp() * 1000)
    out_path = Path(filename_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pl.scan_parquet(filename_in).filter(
        pl.col("timestamp") < threshold_ms
    ).sink_parquet(out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train", type=str, required=True,
        help="Path to train file/dir — full pre-threshold clickstream.",
    )
    parser.add_argument(
        "--out", type=str, required=True,
        help="Output filename/dir.",
    )
    parser.add_argument(
        "--synth-threshold", type=str, default=DEFAULT_SYNTH_THRESHOLD,
        help=(
            "Synthetic threshold date (ISO). Events strictly before this go to "
            "synth_train, events at/after this + 12h go to synth_eval. Default "
            f"is {DEFAULT_SYNTH_THRESHOLD} (one week before the official eval)."
        ),
    )

    args = parser.parse_args()

    assert os.path.isfile(args.train) == os.path.isfile(args.out)  # either both are files or directories

    threshold_date = datetime.fromisoformat(args.synth_threshold)

    if os.path.isfile(args.train) or "*" in args.train:  # process wildcard pattern as one merged file
        make_train(
            filename_in=args.train,
            filename_out=args.out,
            threshold_date=threshold_date
        )
    else:
        logger.info(f"processing multiple files in the directory {args.train}..")
        part_filenames = list(filter(lambda fn: fn.startswith("part_"), os.listdir(args.train)))
        newline = "\n"  # py3.11 workaround
        logger.info(f"filenames to be processed: {newline.join(part_filenames)}")
        for part_filename in part_filenames:
            logger.info(f"{'#'*20}{newline}start processing {part_filename}...{newline}")
            train_path = os.path.join(args.train, part_filename)
            out_filename = part_filename
            out_path = os.path.join(args.out, out_filename)
            make_train(
                filename_in=train_path,
                filename_out=out_path,
                threshold_date=threshold_date
            )

if __name__ == "__main__":
    main()