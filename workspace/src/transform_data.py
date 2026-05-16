import polars as pl
import os
import sys
from pathlib import Path
from loguru import logger
from datetime import datetime
from debug_constants import DEBUG_ARGV_MAKE_TRAIN, ARGV_MAKE_TRAIN_SEPARATE
from utils import load_config
from stage import BaseStage, StageStatus
import mlflow
import copy
import utils


PRED_OPS = {
    "<":  lambda col, val: col < val,
    ">":  lambda col, val: col > val,
    "<=": lambda col, val: col <= val,
    ">=": lambda col, val: col >= val,
    "==": lambda col, val: col == val,
    "!=": lambda col, val: col != val,
    "is_in": lambda col, val: col.is_in(val)
}

AGG_FUNCS = {
    "count":    lambda col, **kw: col.count(),
    "n_unique": lambda col, **kw: col.n_unique(),
    "sum":      lambda col, **kw: col.sum(),
    "mean":     lambda col, **kw: col.mean(),
    "min":      lambda col, **kw: col.min(),
    "max":      lambda col, **kw: col.max(),
    "first":    lambda col, **kw: col.first(),
    "last":     lambda col, **kw: col.last(),
    "unique":   lambda col, **kw: col.unique(),
    "count_if": lambda col, val_range, **kw: pl.all_horizontal(
        [make_predicate(col, val_range_item) for val_range_item in val_range.items()]).sum()
}

def make_agg_expr(agg_item: dict, keys: list[str]) -> pl.Expr:
    col_name = agg_item["col"]
    func = agg_item["func"]
    alias = agg_item.get("alias", f"{func}_{col_name}_by_{'_'.join(keys)}")
    condition = agg_item.get("condition", None)

    expr = pl.col(col_name)
    if "filters" in agg_item:
        filters = []
        for filter in agg_item["filters"]:
            for filter_col_name, val_range in filter.items():
                for val_range_item in val_range.items():
                    filters.append(make_predicate(pl.col(filter_col_name), val_range_item))
        expr = expr.filter(*filters)

    if func not in AGG_FUNCS:
        raise ValueError(f"Unknown aggregation function '{func}'")

    return AGG_FUNCS[func](expr, val_range=condition).alias(alias)

def make_aggregations(df: pl.LazyFrame, cfg: dict) -> dict[tuple[str], pl.LazyFrame]:
    if "features" not in cfg or "aggregations" not in cfg["features"]:
        return dict()

    agg_frames = dict()
    for agg_block in cfg["features"]["aggregations"]:
        keys = agg_block["group_by"]
        exprs = [make_agg_expr(item, keys) for item in agg_block["agg"]]
        agg_frames[tuple(keys)] = df.group_by(keys).agg(exprs)

    return agg_frames

def make_predicate(expr, val_range_item):
    pred, val = val_range_item
    if pred not in PRED_OPS:
        raise ValueError(f"Unknown predicate '{pred}'")
    return PRED_OPS[pred](expr, val)

def transform_date_to_ms(cfg):
    cfg["filters"]["timestamp"] = {
        op: int(datetime.fromisoformat(val).timestamp() * 1000) for op, val in cfg["filters"]["date_thr"].items()
    }
    cfg["filters"].pop("date_thr")

def make_filters(cfg):
    if "filters" not in cfg:
        return dict()

    if "date_thr" in cfg["filters"]:
        transform_date_to_ms(cfg)

    filters = dict()
    for col_name, val_range in cfg["filters"].items():
        for val_range_item in val_range.items():
            filters[col_name] = make_predicate(pl.col(col_name), val_range_item)
    return filters

def apply_filters(df, filter_expressions):
    return df.filter(*filter_expressions)

def make_df(df, cfg, df_accum=None):
    agg_frames = make_aggregations(df, cfg)
    filters = make_filters(cfg)

    filtered_agg_frames = dict()
    for keys, agg_frame in agg_frames.items():
        valid_filters = {fname: f for fname, f in filters.items() if fname in agg_frame.collect_schema().names()}
        agg_frame = agg_frame.filter(*valid_filters.values())
        filtered_agg_frames[keys] = agg_frame

    if cfg.get("eager_execution", False):
        filtered_agg_frames = {k: v.collect() for k, v in filtered_agg_frames.items()}

    if cfg.get("join_back", True):
        for keys, agg_frame in filtered_agg_frames.items():
            agg_frame = agg_frame.lazy() if isinstance(agg_frame, pl.DataFrame) else agg_frame
            df = df.join(agg_frame, on=keys, how='semi')

        for keys, agg_frame in filtered_agg_frames.items():
            agg_frame = agg_frame.lazy() if isinstance(agg_frame, pl.DataFrame) else agg_frame
            df = df.join(agg_frame, on=keys, how='inner')

        df = df.filter(*filters.values())

        return df

    return filtered_agg_frames

def process_data(frames: list[pl.LazyFrame], cfg, df_accum=None):
    assert df_accum is None or (cfg.get("incremental_accum", False) and len(frames) == 1)
    
    if df_accum is not None and cfg.get("incremental_accum", False) and len(frames) == 1:
        return [make_df(pl.concat([frames[0], df_accum]), cfg)]

    return [make_df(elem, cfg) for elem in frames]


def parse_path_in(path_in):
        if os.path.isdir(path_in):
            return path_in, lambda s: s.startswith("part_")

        return str(Path(path_in).parent), lambda s: s == str(Path(path_in).name)


class DataTransformStage(BaseStage):
    def assert_args_in_cfg(self):
        assert "in_artifacts" in self.cfg
        assert "filename_in" in self.cfg["in_artifacts"]

        assert "kwargs" in self.cfg
        assert "cfg" in self.cfg["kwargs"]

        assert "out_artifacts" in self.cfg
        assert "filename_out" in self.cfg["out_artifacts"]

        assert (  #either both are files or both are dirs; out dir must end in a "/"
            not(os.path.isdir(self.cfg["in_artifacts"]["filename_in"]) ^ (os.path.split(self.cfg["out_artifacts"]["filename_out"])[-1] == ""))
            or (os.path.isdir(self.cfg["in_artifacts"]["filename_in"]) and self.cfg["in_artifacts"].get("filename_accum", None) is not None)
        )    

    def load_artifacts(self):
        path_in = self.cfg["in_artifacts"]["filename_in"]
        df_accum = None
        if "filename_accum" in self.cfg["in_artifacts"]:
            filename_accum = self.cfg["in_artifacts"]["filename_accum"]
            df_accum = utils.scan_parquet(filename_accum)
            #mlflow.log_artifact(filename_accum)
        res = {"df_accum": df_accum}

        dir_in, filename_filter = parse_path_in(path_in)
        parts = []
        for part_filename in sorted(list(filter(filename_filter, os.listdir(dir_in)))):
            logger.debug(f"scanning {part_filename} from {dir_in}...")
            read_path = os.path.join(dir_in, part_filename)
            parts.append(utils.scan_parquet(read_path))
            #mlflow.log_artifact(read_path)
        return {**res, "frames": parts}

    
    def write_artifacts(self, res: list[pl.LazyFrame | pl.DataFrame] | list[dict[tuple, pl.LazyFrame | pl.DataFrame]]):
        super().write_artifacts(res)
        path_in = self.cfg["in_artifacts"]["filename_in"]
        path_out = self.cfg["out_artifacts"]["filename_out"]
        need_keys = self.cfg["kwargs"]["cfg"].get("append_keys_to_filename", True)

        dir_in, filename_filter = parse_path_in(path_in)
        part_filenames = sorted(list(filter(filename_filter, os.listdir(dir_in))))
        for elem, part_filename in zip(res, part_filenames):
            logger.info(f"{'#'*20}\nProcessing {part_filename} from {dir_in}...\n")
            if isinstance(elem, dict):
                # case of join_back: False
                for join_keys, df in elem.items():
                    p = Path(part_filename)
                    part_filename_keys = "_".join([str(p.stem), *sorted(join_keys)]) + p.suffix if need_keys else part_filename
                    write_path = os.path.join(path_out, part_filename_keys) if os.path.isdir(path_out) else path_out
                    utils.sink_parquet(df, write_path, remove_local=False) if isinstance(df, pl.LazyFrame) else utils.write_parquet(df, write_path, remove_local=False)
                    mlflow.log_artifact(write_path)
            else:
                write_path = os.path.join(path_out, part_filename) if os.path.isdir(path_out) else path_out
                utils.sink_parquet(elem, write_path, remove_local=False) if isinstance(elem, pl.LazyFrame) else utils.write_parquet(elem, write_path, remove_local=False)
                mlflow.log_artifact(write_path)


    def parse_kwargs(self):
        return self.cfg["kwargs"]


class SequentialStage(BaseStage):
    def __init__(self, cfg, stage_class, func, run_name=None):
        """
            `cfg` - config object
            `func` - callable function, returns a result which is then written as artifacts to disk
            `stage_class` - class to build with
        """
        self.stage_class = stage_class
        super().__init__(cfg, func, run_name)

    def assert_args_in_cfg(self):
        return self.stage_class.assert_args_in_cfg(self)

    def parse_kwargs(self):
        return self.stage_class.parse_kwargs(self)

    def make_children_stages(self):
        path_in = self.cfg["in_artifacts"]["filename_in"]
        path_out = self.cfg["out_artifacts"]["filename_out"]

        dir_in, filename_filter = parse_path_in(path_in)
        dir_out = path_out if os.path.split(path_out)[-1] == "" else str(Path(path_out).parent)
        part_filenames = sorted(list(filter(filename_filter, os.listdir(dir_in))))
        logger.debug(f"making children stages for running on directory: {dir_in}, files: {part_filenames}")
        children_stages = list()
        for part_filename in part_filenames:
            logger.debug(f"making children stage {part_filename} from {dir_in}...")
            full_filename_in = os.path.join(dir_in, part_filename)
            full_filename_out = os.path.join(dir_out, part_filename) if dir_out == path_out else path_out
            cfg_copy = copy.deepcopy(self.cfg)
            cfg_copy["in_artifacts"]["filename_in"] = full_filename_in
            cfg_copy["out_artifacts"]["filename_out"] = full_filename_out
            children_stages.append(self.stage_class(cfg_copy, self.func, run_name_suffix="_"+Path(part_filename).stem))
        
        return children_stages
    
    def run(self):
        with mlflow.start_run(run_name=self.run_name, nested=True):
            mlflow.log_params(self.kwargs)
            mlflow.log_dict(self.cfg, artifact_file="configs/stage_config.json")
            children_stages = self.make_children_stages()
            try:
                for stage in children_stages:
                    stage.run()
                mlflow.set_tag("status", "success")
            except Exception as e:
                self.status = StageStatus.FAILED
                mlflow.set_tag("status", "failed")
                mlflow.set_tag("error", str(e))
                raise


def make_empty_df(schema: dict[str, str]):
    schema = {k: getattr(pl, v) for k, v in schema.items()}
    return pl.LazyFrame(schema=schema)

class MakeAccumStage(BaseStage):
    def assert_args_in_cfg(self):
        assert "out_artifacts" in self.cfg
        assert "filename_accum" in self.cfg["out_artifacts"]

        assert "kwargs" in self.cfg
        assert "schema" in self.cfg["kwargs"]

    def load_artifacts(self):
        return dict()

    def parse_kwargs(self):
        return self.cfg["kwargs"]
    
    def write_artifacts(self, run_result):
        super().write_artifacts(run_result)
        path_out = self.cfg["out_artifacts"]["filename_accum"]
        utils.sink_parquet(run_result, path_out, remove_local=False)
        mlflow.log_artifact(path_out)

def main():
    # assert len(sys.argv) == 2, "please provide path to stage yaml config as an argument"
    # preprocess_config_path = sys.argv[1]
    preprocess_config_path = "/project/workspace/config/data/eval/unique_users_cnt_by_item_id.yml"

    aggregate_cfg = load_config(preprocess_config_path)["aggregate"]
    accum_cfg = load_config(preprocess_config_path)["make_accum"]
    combine_cfg = load_config(preprocess_config_path)["combine"]

    logger.info("starting pipeline...")

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("transform_data")

    stages = [
        DataTransformStage(aggregate_cfg, process_data),
        MakeAccumStage(accum_cfg, make_empty_df),
        SequentialStage(combine_cfg, DataTransformStage, process_data)
    ]

    with mlflow.start_run(run_name="data_transform_pipeline"):
        logger.info(f"total stages: {len(stages)}")
        for idx, stage in enumerate(stages):
            logger.info(f"running stage idx={idx}")
            stage.run()
        logger.info("transformed data successfully.")


if __name__ == "__main__":
    main()
