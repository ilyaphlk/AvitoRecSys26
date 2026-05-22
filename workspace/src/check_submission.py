from utils import check_submission
import argparse
from loguru import logger
import polars as pl

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gt-path", type=str, required=True,
        help="Path to ground truth csv file",
    )
    parser.add_argument(
        "--preds-path", type=str, required=True,
        help="Path to submission csv file",
    )
    parser.add_argument(
        "--users-path", type=str, required=False, default=None,
        help="Path to users csv file (user_id, bucket)",
    )
    parser.add_argument(
        "--items-path", type=str, required=False, default=None,
        help="Path to items parquet file (item_id, vertical_id)",
    )
    args = parser.parse_args()

    submission_res = check_submission(
        args.gt_path,
        args.preds_path,
        args.users_path,
        args.items_path
    )

    logger.info(f"recall on {args.gt_path}")
    with pl.Config(tbl_cols=-1):
        for k, v in submission_res.items():
            logger.info(f"{k}: {v}")