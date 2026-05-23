import os
import mlflow
import utils
import argparse
from loguru import logger
import polars as pl

from stage import BaseStage

def calc_metric(df_true, df_pred, df_users=None, df_item_verticals=None, top_sizes: list[int] = None):
    """
        calculate mean recall across all eval (true) users
        df_true: user_id, item_id
        df_pred: user_id, item_id
        df_users: user_id, vertical_id
        top_sizes: list of sizes for which to calculate recall@ k
    """
    # assert that sets of users are the same
    assert set(df_true["user_id"]) == set(df_pred["user_id"]), "sets of users in eval and pred are different"

    # assert that all recommendations are unique for each user
    count_unique_preds = df_pred.group_by("user_id").agg(
        pl.col("item_id").len().alias("count"),
        pl.col("item_id").n_unique().alias("count_unique")
    )
    assert(all(count_unique_preds["count"] == count_unique_preds["count_unique"])), "pred has users with non-unique items"

    top_sizes = [160] if top_sizes is None else sorted(list(set(top_sizes) | set([160])))

    recall_at_k = dict()
    for top_size in top_sizes:
        logger.info(f"calculating recall@{top_size}")
        df_pred_top = (
            df_pred
            .with_columns(
                pl.col("scores")
                .rank(method="dense", descending=True)
                .over("user_id")
                .alias("score_rank")
            )
            .filter(pl.col("score_rank") <= top_size)    
        )
        joined = df_true.join(df_pred_top, on=("user_id", "item_id"))
        total_by_user = df_true.group_by("user_id").agg(pl.len().alias("total_items"))
        retrieved_by_user = joined.group_by("user_id").agg(pl.len().alias("retrieved_items"))
        total_by_user = total_by_user.join(retrieved_by_user, on=("user_id"), how="left").fill_null(0)
        recall_by_user = total_by_user.select(
            pl.col("user_id"),
            (pl.col("retrieved_items") / pl.col("total_items")).alias("recall")
        )
        recall_at_k[str(top_size)] = recall_by_user["recall"].mean()

    res = dict()
    res["overall_recall"] = recall_at_k[str(160)]
    res["recall_at_k"] = pl.DataFrame(recall_at_k)

    if df_users is not None:
        recall_with_buckets = recall_by_user.join(df_users, on="user_id")
        res["per_bucket_recall"] = recall_with_buckets.group_by("bucket").agg(pl.col("recall").mean()).sort(by="bucket")

        if df_item_verticals is not None:
            filtered_items = df_item_verticals.select(["item_id", "vertical_id"]).join(df_pred.lazy(), on="item_id", how="semi")
            preds_with_verticals = df_pred.lazy().join(filtered_items, on="item_id", how="left")
            bucket_v_id_counts = (
                preds_with_verticals
                .join(df_users.lazy(), on="user_id", how="left")
                .group_by(["bucket", "vertical_id"]).agg(pl.len().alias("cnt"))
                .collect()
            )
            total_by_bucket = bucket_v_id_counts.group_by("bucket").agg(pl.col("cnt").sum().alias("total"))
            total_by_v_id = bucket_v_id_counts.group_by("vertical_id").agg(pl.col("cnt").sum().alias("total"))

            bucket_v_id_counts_norm_by_bucket = bucket_v_id_counts.join(
                total_by_bucket,on="bucket", how="left"
            ).select(
                pl.col("bucket"),
                pl.col("vertical_id"),
                (100. * pl.col("cnt") / pl.col("total")).alias("pct")
            )

            bucket_v_id_counts_norm_by_v_id = bucket_v_id_counts.join(
                total_by_v_id,on="vertical_id", how="left"
            ).select(
                pl.col("bucket"),
                pl.col("vertical_id"),
                (100. * pl.col("cnt") / pl.col("total")).alias("pct")
            )

            confusion_matrix = bucket_v_id_counts.pivot(on="vertical_id", index="bucket", values="cnt")
            confusion_matrix_pct_by_bucket = bucket_v_id_counts_norm_by_bucket.pivot(on="vertical_id", index="bucket", values="pct")
            confusion_matrix_pct_by_v_id = bucket_v_id_counts_norm_by_v_id.pivot(on="bucket", index="vertical_id", values="pct")

            res["confusion_matrix"] = confusion_matrix.sort(by="bucket")
            res["confusion_matrix_pct_by_bucket"] = confusion_matrix_pct_by_bucket.sort(by="bucket")
            res["confusion_matrix_pct_by_v_id"] = confusion_matrix_pct_by_v_id.sort(by="vertical_id")

    return res


def check_submission(df_true_filename, df_pred_filename, df_users_filename=None, df_item_verticals_filename=None, top_sizes=None):
    df_true = utils.read_csv(df_true_filename)
    df_pred = utils.read_csv(df_pred_filename)
    df_users = utils.read_csv(df_users_filename) if df_users_filename is not None else None
    df_item_verticals = utils.scan_parquet(df_item_verticals_filename) if df_item_verticals_filename is not None else None

    return calc_metric(df_true, df_pred, df_users, df_item_verticals, top_sizes)



class CheckSubmissionStage(BaseStage):
    def assert_args_in_cfg(self, cfg):
        assert "in_artifacts" in cfg
        assert "ground_truth_path" in cfg["in_artifacts"]
        assert "preds_path" in cfg["in_artifacts"]
        

        assert "out_artifacts" in cfg
        assert "overall_recall" in cfg["out_artifacts"]
        assert "per_bucket_recall" in cfg["out_artifacts"]
        assert "confusion_matrix" in cfg["out_artifacts"]
        assert "confusion_matrix_pct_by_bucket" in cfg["out_artifacts"]
        assert "confusion_matrix_pct_by_v_id" in cfg["out_artifacts"]

        assert "make_mlflow_artifacts_subdir" not in cfg["out_artifacts"] or "artifacts_dir" in cfg["out_artifacts"]

    def parse_kwargs(self):
        return self.cfg.get("kwargs", {})

    def load_artifacts(self):
        in_artifacts = self.cfg["in_artifacts"]
        artifacts_dir = in_artifacts.get("artifacts_dir", "")
        artifacts_run_id = in_artifacts.get("artifacts_run_id", "")
        artifacts_experiment_name = in_artifacts.get("artifacts_experiment_name", "")

        preds_path = os.path.join(
            artifacts_dir, artifacts_experiment_name, artifacts_run_id, in_artifacts["preds_path"]
        )

        return {
            "df_true": utils.read_csv(in_artifacts["ground_truth_path"]),
            "df_pred": utils.read_csv(preds_path),
            "df_users": utils.read_csv(in_artifacts["users_path"]) if "users_path" in in_artifacts else None,
            "df_item_verticals": utils.scan_parquet(in_artifacts["items_path"]) if "items_path" in in_artifacts else None,
        }

    def write_artifacts(self, run_result):
        make_mlflow_subdirs = self.cfg["out_artifacts"].get("make_mlflow_artifacts_subdir", False)
        out_artifacts = self.cfg["out_artifacts"]

        if make_mlflow_subdirs:
            mlflow_run = mlflow.active_run()
            run_id = mlflow_run.info.run_id
            exp_name = mlflow.get_experiment(mlflow_run.info.experiment_id).name
            out_artifacts["artifacts_dir"] = os.path.join(
                out_artifacts["artifacts_dir"], exp_name, run_id, ""
            )

        super().write_artifacts(run_result)

        artifacts_dir = out_artifacts.get("artifacts_dir", "")

        mlflow.log_metric("overall_recall", run_result["overall_recall"])

        # Resolve full output paths under artifacts_dir
        section_to_full_path = {}
        for section in out_artifacts.keys():
            if section in ["make_mlflow_artifacts_subdir", "artifacts_dir"]:
                continue
            out_artifacts[section] = os.path.join(artifacts_dir, out_artifacts[section])

        for section, df in run_result.items():
            if isinstance(df, (int, float)):
                df = pl.DataFrame({section: [df]})
            utils.write_csv(
                df,
                out_artifacts[section],
                remove_local=self.remove_local,
                log_artifact=self.log_artifacts,
            )


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
    parser.add_argument(
        "--top-sizes", nargs='+', type=int, required=False, default=None,
        help="A list of top sizes for which to compute recall@k",
    )
    args = parser.parse_args()

    submission_res = check_submission(
        args.gt_path,
        args.preds_path,
        args.users_path,
        args.items_path,
        args.top_sizes,
    )

    logger.info(f"recall on {args.gt_path}")
    with pl.Config(tbl_cols=-1):
        for k, v in submission_res.items():
            logger.info(f"{k}: {v}")
