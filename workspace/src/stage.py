from enum import Enum
from typing import Dict, Any

class StageStatus(Enum):
    NOT_STARTED = 0
    FINISHED = 1
    FAILED = 2

class BaseStage:
    def __init__(self, cfg, func):
        """
            `cfg` - config object
            `func` - callable function, returns a result which is then written as artifacts to disk
        """
        self.cfg = cfg
        self.func = func
        self.kwargs = self.parse_kwargs()
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
            write run artifacts to disk
        """
        raise NotImplementedError
    
    def run(self):
        self.assert_args_in_cfg()
        try:
            input_artifacts = self.load_artifacts()
            res = self.func(**{**self.kwargs, **input_artifacts})
            self.write_artifacts(res)
        except Exception as e:
            self.status = StageStatus.FAILED
            raise e
        self.status = StageStatus.FINISHED