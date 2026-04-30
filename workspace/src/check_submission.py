from utils import check_submission
import argparse
from loguru import logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ground-truth-path", type=str, required=True,
        help="Path to ground truth csv file",
    )
    parser.add_argument(
        "--submission-path", type=str, required=True,
        help="Path to submission csv file",
    )
    args = parser.parse_args()

    avg_recall = check_submission(args.ground_truth_path, args.submission_path)

    logger.info(f"avg recall on {args.ground_truth_path}: {avg_recall}")