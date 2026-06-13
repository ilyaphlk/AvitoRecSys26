from enum import Enum
from typing import Dict, Any
from pathlib import Path
import mlflow
import os
import utils
from loguru import logger

class StageStatus(Enum):
    NOT_STARTED = 0
    FINISHED = 1
    FAILED = 2

class BaseStage:
    def __init__(self, cfg, func=None, run_name=None, run_name_suffix=None):
        """
            `cfg` - config object
            `func` - callable function, returns a result which is then written as artifacts to disk
        """
        self.cfg = cfg
        self.func = func if func is not None else self._func
        self.run_name = run_name if run_name is not None else self.__class__.__name__
        self.run_name = self.run_name + run_name_suffix if run_name_suffix is not None else self.run_name
        self.kwargs = self.parse_kwargs()
        self.assert_args_in_cfg(self.cfg)
        self.status = StageStatus.NOT_STARTED
        self.log_artifacts = self.cfg.get("mlflow_log_artifacts", True)
        self.remove_local = self.cfg.get("remove_local", False)
    
    def assert_args_in_cfg(self, cfg):
        """
            assert that all required args are in the cfg
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement assert_args_in_cfg")
    
    def parse_kwargs(self):
        """
            parse function kwargs from cfg
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement parse_kwargs")

    def update_cfg(self, new_part: dict[str: Any], ignore_on_assert_failure=False):
        new_cfg = utils.deep_merge(self.cfg, new_part)

        try:
            self.assert_args_in_cfg(cfg=new_cfg)
            self.cfg = new_cfg
        except AssertionError as e:
            if ignore_on_assert_failure:
                logger.warning("asserts failed when trying to update the stage config, ignoring...")
            else:
                logger.error("error when updating stage config")
                raise
        except Exception as e:
            raise

    
    def load_artifacts(self) -> Dict[str, Any]:
        """
            load artifacts needed for running from disk, return as dict
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement load_artifacts")
    
    def write_artifacts(self, run_result):
        """
            write run artifacts to disk (locally)
        """
        write_artifacts_kwargs = {"partition_args", "artifacts_dir", "make_mlflow_artifacts_subdir"}
        artifacts_dir = self.cfg["out_artifacts"].get("artifacts_dir", "")
        for section_name, fp in self.cfg["out_artifacts"].items():
            if section_name in write_artifacts_kwargs:
                continue
            if os.path.split(fp)[-1] == "":  # check if dir-like
                fp = os.path.join(fp, "placeholder")
            p = Path(utils.LOCAL_DATA_DIR, artifacts_dir, fp)
            p.parent.mkdir(parents=True, exist_ok=True)

    def _func(*args, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__} must implement _func")
    
    def run(self):
        with mlflow.start_run(run_name=self.run_name, nested=True):
            mlflow.log_params(self.kwargs)
            mlflow.log_dict(self.cfg, artifact_file="configs/stage_config.json")
            try:
                input_artifacts = self.load_artifacts()
                res = self.func(**{**self.kwargs, **input_artifacts})
                self.write_artifacts(res)
                self.status = StageStatus.FINISHED
                mlflow.set_tag("status", "finished")
            except Exception as e:
                self.status = StageStatus.FAILED
                mlflow.set_tag("status", "failed")
                mlflow.set_tag("error", str(e))
                raise

            mlflow_run = mlflow.active_run()
            return {
                "run_name": self.run_name,
                "run_id": mlflow_run.info.run_id,
                "experiment_id": mlflow_run.info.experiment_id,
                "experiment_name": mlflow.get_experiment(mlflow_run.info.experiment_id).name
            }
