import polars as pl

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