import als
import transform_data
import prepare_local_eval
import sys
from utils import load_config
import mlflow
from loguru import logger

STAGES_DICT = {
    "ALSPreprocessStage": als.ALSPreprocessStage,
    "ALSTrainStage": als.ALSTrainStage,
    "ALSInferenceStage": als.ALSInferenceStage,
    "DataTransformStage": transform_data.DataTransformStage,
    "MakeAccumStage": transform_data.MakeAccumStage,
    "SequentialStage": transform_data.SequentialStage,
    "PrepareLocalEvalStage": prepare_local_eval.PrepareLocalEvalStage,
}

FUNCS_DICT = {
    "als_make_train": als.make_train,
    "als_train": als.train,
    "als_inference": als.inference,
    "process_data": transform_data.process_data,
    "make_empty_df": transform_data.make_empty_df,
    "prepare_local_eval": prepare_local_eval.prepare_local_eval
}

def make_stage_object(stage_dict):
    cfg = load_config(stage_dict["config_path"])[stage_dict["stage_name"]]
    stage_class = STAGES_DICT[stage_dict["stage_class"]]
    stage_func = FUNCS_DICT[stage_dict["stage_func"]]
    child_stage_class = STAGES_DICT[stage_dict["child_stage_class"]] if "child_stage_class" in stage_dict else None
    if child_stage_class is not None:
        return stage_class(cfg, child_stage_class, stage_func)
    return stage_class(cfg, stage_func)

def run_pipeline():
    assert len(sys.argv) == 2, "please provide a path to yaml config with pipeline args"
    pipeline_config_path = sys.argv[1]
    # pipeline_config_path = "/project/workspace/config/pipeline/make_item_blacklist_by_user_cnt.yml"

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
    mlflow.set_tracking_uri("http://localhost:5000")
    logger.debug("setting mlflow exp...")
    mlflow.set_experiment(experiment_name=experiment_name)

    logger.info("starting pipeline...")
    with mlflow.start_run(run_name=run_name):
        for stage in stages:
            logger.info(f"starting stage {stage['name']}...")
            stage["object"].run()
            logger.info(f"ran stage {stage['name']}.")

if __name__ == "__main__":
    run_pipeline()