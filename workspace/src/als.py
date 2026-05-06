from scipy.sparse import csr_matrix
import numpy as np
import implicit
import os
import polars as pl
import argparse
from loguru import logger
from debug_constants import DEBUG_ARGV_ALS, ARGV_ALS_LOCAL_SEPARATE, SUBMIT_ARGV_ALS
from pathlib import Path
import time
import psutil

def ram_report():
    mem = psutil.virtual_memory()
    available_bytes = mem.available
    available_gb = available_bytes / (1024 ** 3)
    logger.info(f"RAM Available/Total/Usage: {available_gb:.2f}GB / {mem.total / (1024 ** 3):.2f}GB / {mem.percent}%")


def get_als_pred(df_train, user_to_pred, N=160, batch_size=100):
    user_ids = df_train["user_id"].unique().to_numpy()
    item_ids = df_train["item_id"].unique().to_numpy()

    logger.info("made unique")

    user_id_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_id_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}

    user4pred_popular = np.array(list(set(user_to_pred) - set(user_ids)))
    n_unique_users, n_unique_items = len(user_ids), len(item_ids)
    del user_ids, item_ids

    logger.info(f"made maps to idx. max user_idx: {len(user_id_to_index)}, max_item_idx: {len(item_id_to_index)}")
    logger.debug("deleted user_ids, item_ids.")

    rows = df_train["user_id"].replace_strict(user_id_to_index).cast(pl.UInt32).to_numpy()
    cols = df_train["item_id"].replace_strict(item_id_to_index).cast(pl.UInt32).to_numpy()

    logger.info("made rows & cols")

    values = (df_train["cnt_shows_by_user_id_item_id"] + 10 * df_train["cnt_clicks_by_user_id_item_id"]).cast(pl.Float32).to_numpy()

    # for the non-als preds below
    popular_top = (
        pl.DataFrame({"item_id": df_train["item_id"]})
        .group_by("item_id")
        .agg(pl.len().alias("count"))
        .sort(by=("count"), descending=True)
        .head(N)
    )
    del df_train
    logger.debug("deleted df_train")
    ram_report()

    logger.info("start init matrix...")
    sparse_matrix = csr_matrix((values, (rows, cols)),shape=(n_unique_users, n_unique_items))
    del rows, cols, values
    logger.info("finish init matrix")
    ram_report()

    logger.info("start fit model...")
    model = implicit.als.AlternatingLeastSquares(iterations=10, factors=60, random_state=42, calculate_training_loss=True)
    model.fit(sparse_matrix, )
    logger.info("finish fit model")

    user4pred_als = np.array([user_id_to_index[i] for i in user_to_pred if i in user_id_to_index])
    user_matrix = sparse_matrix[user4pred_als]
    del sparse_matrix
    logger.debug("deleted full matrix")

    ram_report()

    #recommendations, scores = model.recommend(user4pred_als, user_matrix, N=160, filter_already_liked_items=True)
    all_recommendations = []
    all_scores = []

    total_batches = (len(user4pred_als) + batch_size - 1) // batch_size
    for start in range(0, len(user4pred_als), batch_size):
        logger.info(f"start recommending for batch {start // batch_size + 1} / {total_batches}")
        end = min(start + batch_size, len(user4pred_als))
        batch_user_ids = user4pred_als[start:end]
        batch_user_matrix = user_matrix[start:end]
        logger.debug("copied batch into ram")
        ram_report()

        batch_recs, batch_scores = model.recommend(
            batch_user_ids,
            batch_user_matrix,
            N=N,
            filter_already_liked_items=True
        )
        logger.debug("finished recommending")
        ram_report()

        all_recommendations.append(batch_recs)
        all_scores.append(batch_scores)
        del batch_recs, batch_scores, batch_user_ids, batch_user_matrix

        logger.debug("deleted local vars explicitly")
        ram_report()
        logger.info(f"recommended for users {start}:{end} / {len(user4pred_als)}")


    recommendations = np.vstack(all_recommendations)
    scores = np.vstack(all_scores)

    logger.info(f"got recs for {len(recommendations)} users seen in train")

    index_to_item_id = {v:k for k,v in item_id_to_index.items()}
    index_to_user_id = {v:k for k,v in user_id_to_index.items()}

    logger.info("made maps from idx")

    df_pred = pl.DataFrame(
        {
            'item_id': [
                [index_to_item_id[i] for i in i] for i in recommendations.tolist()
            ],
            'user_id': [
                index_to_user_id[i] for i in user4pred_als.tolist()
            ],
            'scores': scores.tolist()
        }
    )

    logger.info("made df_pred for als recs")

    df_pred = df_pred.explode(['item_id', 'scores'])

    logger.info("exploded it")

    # fallback to popular items
    
    logger.info(f"{len(user4pred_popular)} users to pred by popularity (cold start)")

    logger.info(f"computed popular_top")

    df_pred_popular = pl.DataFrame(
        {
            'item_id': [list(popular_top["item_id"]) for _ in range(len(user4pred_popular))],
            'user_id': user4pred_popular,
            'scores': [list(popular_top["count"] * 1.0) for _ in range(len(user4pred_popular))]
        }
    )
    df_pred_popular = df_pred_popular.explode(['item_id', 'scores']).with_columns(
        pl.col("item_id").cast(pl.UInt32).alias("item_id"),
        pl.col("user_id").cast(pl.UInt32).alias("user_id"),
        pl.col("scores").cast(pl.Float64).alias("scores"),
    )

    return pl.concat([df_pred, df_pred_popular])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train", type=str, required=True,
        help="Path to train file",
    )
    parser.add_argument(
        "--eval-user-events", type=str, required=True,
        help="Path to eval_user_events.pq.",
    )
    parser.add_argument(
        "--item-features", type=str, required=True,
        help="Path to item_features.parquet (used for vertical filter).",
    )
    parser.add_argument(
        "--eval-users", type=str, required=True,
        help="Path to eval_users.csv (single-column user_id list).",
    )
    parser.add_argument(
        "--out", type=str, default="submission_als.csv",
        help="Output CSV path with (user_id, item_id) pairs.",
    )
    parser.add_argument(
        "--k", type=int, default=160,
        help="Items per user (matches the Recall@160 cap).",
    )
    parser.add_argument(
        "--items-popularity-thr", type=int, default=3,
        help="All items with less than N occurences will be removed from train.",
    )
    args = parser.parse_args(SUBMIT_ARGV_ALS)

    logger.info("starting pipeline...")

    train_path = Path(args.train)
    if os.path.isdir(train_path):
        full_paths = [os.path.join(train_path, fn) for fn in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, fn))]
    else:
        full_paths = [train_path]
    logger.info(f"full paths to train parts: {full_paths}")

    collected_train_parts = []

    logger.info(f"collecting part {args.eval_user_events}...")
    collected_train_parts.append(
        (
            pl.scan_parquet(args.eval_user_events)
            .select(
                pl.col("user_id"),
                pl.col("item_id"),
                pl.col("cnt_shows_by_user_id_item_id"),
                pl.col("cnt_clicks_by_user_id_item_id")
            )
            .group_by(["user_id", "item_id"])
            .agg(pl.col("cnt_shows_by_user_id_item_id").first(), pl.col("cnt_clicks_by_user_id_item_id").first()).collect()
        )
    )

    for fp in sorted(full_paths):
        logger.info(f"collecting part {fp}...")
        collected_train_parts.append(
            (
                pl.scan_parquet(fp)
                .select(
                    pl.col("user_id"),
                    pl.col("item_id"),
                    pl.col("cnt_shows_by_user_id_item_id"),
                    pl.col("cnt_clicks_by_user_id_item_id")
                )
                .group_by(["user_id", "item_id"])
                .agg(pl.col("cnt_shows_by_user_id_item_id").first(), pl.col("cnt_clicks_by_user_id_item_id").first()).collect()
            )
        )

    logger.info("concatenating collected parts..")

    df_train = pl.concat(collected_train_parts)
    del collected_train_parts

    logger.info("concatenated successfully.")

    df_test = pl.read_csv(args.eval_users)
    df_pred = get_als_pred(df_train, df_test["user_id"], N=args.k)
    logger.info("got preds")

    df_pred.select(
        pl.col("user_id"),
        pl.col("item_id")
    ).write_csv(args.out)
    logger.info("wrote submission to disk")


if __name__ == "__main__":
    main()
