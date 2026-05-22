import os
import mlflow
import utils
from utils import check_submission
import argparse
from loguru import logger
import polars as pl

from stage import BaseStage


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
