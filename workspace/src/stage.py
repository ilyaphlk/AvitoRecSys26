from enum import Enum
from typing import Dict, Any

class StageStatus(Enum):
    NOT_STARTED = 0
    FINISHED = 1

class BaseStage:
    def __init__(self, cfg, func):
        self.cfg = cfg
        self.func = func
        self.kwargs = self.parse_kwargs()
        self.status = StageStatus.NOT_STARTED
    
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
        input_artifacts = self.load_artifacts()
        res = self.func(**{**self.kwargs, **input_artifacts})
        self.write_artifacts(res)
        self.status = StageStatus.FINISHED