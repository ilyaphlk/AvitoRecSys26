# pipeline_scenarios.py

from collections import OrderedDict
from dataclasses import dataclass
from loguru import logger
import mlflow
import os

from utils import load_config

from transform_data import DataTransformStage, MakeAccumStage, SequentialStage, JoinTablesStage


@dataclass
class PipelineScenario:
    name: str
    stages: OrderedDict
    experiment: str = "transform_data"

def build_aggregate_combine() -> PipelineScenario:
    cfg = load_config("/project/workspace/config/data/eval/unique_users_cnt_by_item_id.yml")
    return PipelineScenario(
        name="aggregate_combine",
        stages=OrderedDict([
            ("aggregate", DataTransformStage(cfg["aggregate"])),
            ("make_accum", MakeAccumStage(cfg["make_accum"])),
            ("combine",   SequentialStage(cfg["combine"], DataTransformStage)),
        ])
    )

def build_filter_df() -> PipelineScenario:
    cfg = load_config("/project/workspace/config/data/eval/dt_filter_raw_train_debug.yml")
    return PipelineScenario(
        name="filter_df",
        stages=OrderedDict([
            ("filter", SequentialStage(cfg["filter_by_dt"], DataTransformStage)),
        ])
    )

def build_full_whitelist() -> PipelineScenario:
    cfg = load_config("/project/workspace/config/data/features/counters_local_shows_clicks.yml")
    return PipelineScenario(
        name="full_whitelist",
        stages=OrderedDict([
            ("dt_filter",       SequentialStage(cfg["filter_by_dt"],                   DataTransformStage)),
            ("make_agg",        SequentialStage(cfg["aggregate_partitions"],            DataTransformStage)),
            ("make_accum_item", MakeAccumStage( cfg["make_accum_item_id"])),
            ("blacklist_item",  SequentialStage(cfg["make_item_id_blacklist_sequential"], DataTransformStage)),
            ("make_accum_user", MakeAccumStage( cfg["make_accum_user_id"])),
            ("blacklist_user",  SequentialStage(cfg["make_user_id_blacklist_sequential"], DataTransformStage)),
            ("make_whitelist",  SequentialStage(cfg["filter_by_blacklists"],            JoinTablesStage)),
        ])
    )

SCENARIOS = {
    "aggregate_combine": build_aggregate_combine,
    "filter_df":         build_filter_df,
    "full_whitelist":    build_full_whitelist,
}


MLFLOW_URI = f"http://localhost:{os.getenv('MLFLOW_PORT', '5000')}"

def run_scenario(scenario: PipelineScenario):
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(scenario.experiment)

    with mlflow.start_run(run_name=scenario.name):
        for name, stage in scenario.stages.items():
            logger.info(f"running stage: {name}")
            stage.run()

if __name__ == "__main__":
    import sys
    scenario_name = sys.argv[1] if len(sys.argv) > 1 else "filter_df"
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario_name}'. Available: {list(SCENARIOS)}")
    run_scenario(SCENARIOS[scenario_name]())