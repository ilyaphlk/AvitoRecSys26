import polars as pl

def calc_metric(df_true, df_pred):
    # todo: assert that submit is valid:
    # 1) all users from true are in pred and no other;
    # 2) for each user all items are different

    # assert that sets of users are the same
    assert set(df_true["user_id"]) == set(df_pred["user_id"]), "sets of users in eval and pred are different"


    joined = df_true.join(df_pred, on=("user_id", "item_id"))
    df_true_by_user = df_true.group_by("user_id").agg(pl.len().alias("total_items"))
    joined_by_user = joined.group_by("user_id").agg(pl.len().alias("retrieved_items"))
    df_true_by_user = df_true_by_user.join(joined_by_user, on=("user_id"), how="left").fill_null(0)
    df_true_by_user = df_true_by_user.select(
        (pl.col("retrieved_items") / pl.col("total_items")).alias("recall")
    )
    return df_true_by_user["recall"].mean()