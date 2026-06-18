import als
import transform_data
import prepare_local_eval
import sys
from utils import load_config
import mlflow
from loguru import logger
import os
import check_submission
import sasrec


STAGES_DICT = {
    "ALSPreprocessStage": als.ALSPreprocessStage,
    "ALSTrainStage": als.ALSTrainStage,
    "ALSInferenceStage": als.ALSInferenceStage,
    "SASRecPreprocessStage": sasrec.SASRecPreprocessStage,
    "SASRecTrainStage": sasrec.SASRecTrainStage,
    "SASRecInferenceStage": sasrec.SASRecInferenceStage,
    "DataTransformStage": transform_data.DataTransformStage,
    "MakeAccumStage": transform_data.MakeAccumStage,
    "SequentialStage": transform_data.SequentialStage,
    "PrepareLocalEvalStage": prepare_local_eval.PrepareLocalEvalStage,
    "JoinTablesStage": transform_data.JoinTablesStage,
    "CheckSubmissionStage": check_submission.CheckSubmissionStage,
}

def make_stage_object(stage_dict):
    cfg_path = stage_dict["config_path"]
    constants_path = stage_dict.get("constants_path", None)
    cfg = load_config(cfg_path, constants_path)[stage_dict["stage_name"]]
    stage_class = STAGES_DICT[stage_dict["stage_class"]]
    child_stage_class = STAGES_DICT[stage_dict["child_stage_class"]] if "child_stage_class" in stage_dict else None
    if child_stage_class is not None:
        return stage_class(cfg, child_stage_class)
    return stage_class(cfg)

def maybe_update_stage_config_from_prev_stages(stage, runs_info):
    cfg = stage["object"].cfg
    get_artifacts_from_stage = cfg["in_artifacts"].get("get_artifacts_from_stage", None) if "in_artifacts" in cfg else None
    if get_artifacts_from_stage is not None:
        assert get_artifacts_from_stage in runs_info, f"cannot update config with run_id, exp_name from {get_artifacts_from_stage}, stage has not run yet"
        new_part = {
            "in_artifacts": {
                "artifacts_run_id": runs_info[get_artifacts_from_stage]["run_id"],
                "artifacts_experiment_name": runs_info[get_artifacts_from_stage]["experiment_name"]
        }}
        stage["object"].update_cfg(new_part)
        logger.info(f"updated stage config of {stage['name']} with run info from {get_artifacts_from_stage}.")

def run_pipeline():
    assert len(sys.argv) == 2, "please provide a path to yaml config with pipeline args"
    pipeline_config_path = sys.argv[1]
    # pipeline_config_path = "/project/workspace/config/pipeline/als/local_eval/click_10/train_inference_160_min_user_min_item.yml"

    pipeline_cfg = load_config(pipeline_config_path)["pipeline"]

    experiment_name = pipeline_cfg["experiment_name"]
    run_name = pipeline_cfg.get("run_name", None)

    stages = []
    for stage_dict in pipeline_cfg["stages"]:
        stages.append(
            {
                "name": stage_dict["stage_name"],
                "object": make_stage_object(stage_dict),
            }
        )
    
    logger.debug("setting mlflow uri...")
    mlflow.set_tracking_uri(f"http://localhost:{os.getenv('MLFLOW_PORT', '5000')}")
    logger.debug("setting mlflow exp...")
    mlflow.set_experiment(experiment_name=experiment_name)

    logger.info("starting pipeline...")
    runs_info = dict()
    with mlflow.start_run(run_name=run_name):
        for stage in stages:
            maybe_update_stage_config_from_prev_stages(stage, runs_info)
            logger.info(f"starting stage {stage['name']}...")
            run_info = stage["object"].run()
            runs_info[stage["name"]] = run_info
            logger.info(f"ran stage {stage['name']}, with run_info: {run_info}")

if __name__ == "__main__":
    run_pipeline()
