from scipy.sparse import csr_matrix
import numpy as np
import implicit
import polars as pl
import argparse
from loguru import logger



def get_als_pred(users, items, user_to_pred, N=160):
    user_ids = users.unique().to_list()
    item_ids = items.unique().to_list()

    user_id_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_id_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
    index_to_item_id = {v:k for k,v in item_id_to_index.items()}
    index_to_user_id = {v:k for k,v in user_id_to_index.items()}

    rows = users.replace_strict(user_id_to_index).to_list()
    cols = items.replace_strict(item_id_to_index).to_list()

    values = [1] * len(users)

    sparse_matrix = csr_matrix((values, (rows, cols)), shape=(len(user_ids), len(item_ids)))

    logger.info("init matrix")

    model = implicit.als.AlternatingLeastSquares(iterations=10, factors=60)
    model.fit(sparse_matrix, )

    logger.info("fit model")

    user4pred_als = np.array([user_id_to_index[i] for i in user_to_pred if i in user_id_to_index])
    recommendations, scores = model.recommend(user4pred_als, sparse_matrix[user4pred_als], N=160, filter_already_liked_items=True)

    logger.info(f"got recs for {len(recommendations)} users seen in train")
    
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

    df_pred = df_pred.explode(['item_id', 'scores'])

    # fallback to popular items
    user4pred_popular = np.array(list(set(user_to_pred) - set(users)))
    logger.info(f"{len(user4pred_popular)} users to pred by popularity (cold start)")
    popular_top = (
        pl.DataFrame({"item_id":items})
        .group_by("item_id")
        .agg(pl.len().alias("count"))
        .sort(by=("count"), descending=True)
        .head(N)
    )

    logger.info(f"computed popular_top")

    df_pred_popular = pl.DataFrame(
        {
            'item_id': [list(popular_top["item_id"]) for _ in range(len(user4pred_popular))],
            'user_id': user4pred_popular,
            'scores': [list(popular_top["count"] * 1.0) for _ in range(len(user4pred_popular))]
        }
    )
    df_pred_popular = df_pred_popular.explode(['item_id', 'scores'])

    return pl.concat([df_pred, df_pred_popular])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
    args = parser.parse_args()


    df_train = pl.scan_parquet(args.eval_user_events).select(pl.col("user_id"), pl.col("item_id"))


    ### stats

    quantiles = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

    cnt_items = df_train.group_by("item_id").agg(pl.len().alias("count"))
    quantiles_items = cnt_items.select(
        [pl.col("count").quantile(q).alias(f"{int(100*q)}q") for q in quantiles],
    )
    logger.info(f"item_id popularity quantiles:\n{quantiles_items.collect()}\n")
    
    cnt_events = df_train.group_by("user_id").agg(pl.len().alias("count"))
    quantiles_events = cnt_events.select(
        [pl.col("count").quantile(q).alias(f"{int(100*q)}q") for q in quantiles],
    )
    logger.info(f"event count per user quantiles:\n{quantiles_events.collect()}\n")

    top_items = cnt_items.filter(pl.col("count") >= args.items_popularity_thr)
    top_items_cnt, total_items_cnt = top_items.select(pl.len()).collect().item(), cnt_items.select(pl.len()).collect().item()
    logger.info(f"{top_items_cnt} / {total_items_cnt} = {100*top_items_cnt/total_items_cnt:2f}% items above threshold")


    ### filtering
    df_train = df_train.join(top_items, on=("item_id"), how="semi").collect()

    logger.info("filtered unpopular items")

    df_test = pl.read_csv(args.eval_users)
    df_pred = get_als_pred(df_train["user_id"], df_train["item_id"], df_test["user_id"], N=args.k)
    logger.info("got preds")

    df_pred.select(
        pl.col("user_id"),
        pl.col("item_id")
    ).write_csv(args.out)
    logger.info("wrote submission to disk")


if __name__ == "__main__":
    main()