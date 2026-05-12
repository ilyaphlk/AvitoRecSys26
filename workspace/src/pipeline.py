import als
import transform_data
import sys
from utils import load_config
import mlflow
from loguru import logger

STAGES_DICT = {
    "ALSPreprocessStage": (lambda cfg: als.ALSPreprocessStage(cfg, als.make_train)),
    "ALSTrainStage": (lambda cfg: als.ALSTrainStage(cfg, als.train)),
    "ALSInferenceStage": (lambda cfg: als.ALSInferenceStage(cfg, als.inference)),
    "DataTransformStage": (lambda cfg: transform_data.DataTransformStage(cfg, transform_data.process_data))
}

def run_pipeline():
    assert len(sys.argv) == 2, "please provide a path to yaml config with pipeline args"
    pipeline_config_path = sys.argv[1]
    # pipeline_config_path = "/project/workspace/config/pipeline/unique_users_cnt_by_item_id.yml"

    pipeline_cfg = load_config(pipeline_config_path)["pipeline"]

    experiment_name = pipeline_cfg["experiment_name"]
    run_name = pipeline_cfg.get("run_name", None)

    stages = []
    for stage_dict in pipeline_cfg["stages"]:
        cfg = load_config(stage_dict["config_path"])[stage_dict["stage_name"]]
        stages.append(
            {
                "name": stage_dict["stage_name"],
                "object": STAGES_DICT[stage_dict["stage_class"]](cfg),
            }
        )
    
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name=experiment_name)

    logger.info("starting pipeline...")
    with mlflow.start_run(run_name=run_name):
        for stage in stages:
            logger.info(f"starting stage {stage['name']}...")
            stage["object"].run()
            logger.info(f"ran stage {stage['name']}.")

if __name__ == "__main__":
    run_pipeline()