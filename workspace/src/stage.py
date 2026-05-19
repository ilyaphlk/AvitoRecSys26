from enum import Enum
from typing import Dict, Any
from pathlib import Path
import mlflow
import os
import utils


class StageStatus(Enum):
    NOT_STARTED = 0
    FINISHED = 1
    FAILED = 2

class BaseStage:
    def __init__(self, cfg, func, run_name=None, run_name_suffix=None):
        """
            `cfg` - config object
            `func` - callable function, returns a result which is then written as artifacts to disk
        """
        self.cfg = cfg
        self.func = func
        self.run_name = run_name if run_name is not None else self.__class__.__name__
        self.run_name = self.run_name + run_name_suffix if run_name_suffix is not None else self.run_name
        self.kwargs = self.parse_kwargs()
        self.assert_args_in_cfg()
        self.status = StageStatus.NOT_STARTED
    
    def assert_args_in_cfg(self):
        """
            assert that all required args are in the cfg
        """
        raise NotImplementedError
    
    def parse_kwargs(self):
        """
            parse function kwargs from cfg
        """
        raise NotImplementedError
    
    def load_artifacts(self) -> Dict[str, Any]:
        """
            load artifacts needed for running from disk, return as dict
        """
        raise NotImplementedError
    
    def write_artifacts(self, run_result):
        """
            write run artifacts to disk (locally)
        """
        write_artifacts_kwargs = ["partition_args"]
        for section_name, fp in self.cfg["out_artifacts"].items():
            if section_name in write_artifacts_kwargs:
                continue
            if os.path.split(fp)[-1] == "":  # check if dir-like
                fp = os.path.join(fp, "placeholder")
            p = Path(utils.LOCAL_DATA_DIR, fp)
            p.parent.mkdir(parents=True, exist_ok=True)
    
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
