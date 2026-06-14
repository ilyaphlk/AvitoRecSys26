import polars as pl
import os
import sys
from pathlib import Path
from loguru import logger
from datetime import datetime
from utils import load_config
from stage import BaseStage, StageStatus
import mlflow
import copy
import utils
from collections import OrderedDict
from typing import Any

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

def parse_filters_cfg(cfg):
    if "filters" not in cfg:
        return {"pre": [pl.lit(True)], "post": [pl.lit(True)]}

    res = dict()
    res["pre"] = make_filters(cfg["filters"]["pre"]) if "pre" in cfg["filters"] else [pl.lit(True)]
    res["post"] = make_filters(cfg["filters"]["post"]) if "post" in cfg["filters"] else [pl.lit(True)]

    return res

def parse_select_cfg(cfg):
    if "select" not in cfg:
        return {"pre": [pl.all()], "post": [pl.all()]}
    
    res = dict()
    res["pre"] = cfg["select"].get("pre", [pl.all()])
    res["post"] = cfg["select"].get("post", [pl.all()])
    return res


def make_aggregations(df: pl.LazyFrame, cfg: dict) -> dict[tuple[str], pl.LazyFrame]:
    if "features" not in cfg or "aggregations" not in cfg["features"]:
        return dict()

    agg_frames = dict()
    for agg_block in cfg["features"]["aggregations"]:
        filters_block = parse_filters_cfg(agg_block)
        select_block = parse_select_cfg(agg_block)
        keys = agg_block["group_by"]
        exprs = [make_agg_expr(item, keys) for item in agg_block["agg"]]
        agg_frames[tuple(sorted(keys))] = (
            df.filter(filters_block["pre"]).select(select_block["pre"])
            .group_by(keys).agg(exprs)
            .filter(filters_block["post"]).select(select_block["post"])
        )

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
    assert "reduce_as" in cfg and cfg["reduce_as"] in ["all", "any"]
    filters = []
    if isinstance(cfg["filters"], list):
        filters = [make_filters(elem) for elem in cfg["filters"]]
    else:
        for col_name, val_range in cfg["filters"].items():
            filters.append(
                pl.all_horizontal([
                    make_predicate(pl.col(col_name), val_range_item) for val_range_item in val_range.items()
                ])
            )

    return pl.all_horizontal(filters) if cfg["reduce_as"] == "all" else pl.any_horizontal(filters)

def make_df(df, cfg, df_accum=None):
    overwrite_df_in=cfg.get("overwrite_df_in", False) or cfg.get("incremental_accum", False)  # todo check whether incremental_accum should be here
    assert not(overwrite_df_in and cfg.get("join_back", True)), "either overwrite df with the agg frame or join frames to original"

    full_df_filters = parse_filters_cfg(cfg)
    full_df_select = parse_select_cfg(cfg)
    df = df.filter(full_df_filters["pre"]).select(full_df_select["pre"])
    
    agg_frames = make_aggregations(df, cfg)

    if cfg.get("eager_execution", False):
        agg_frames = {k: v.collect() for k, v in agg_frames.items()}

    if overwrite_df_in:  # return (the only) agg frame as main df
        join_key = next(iter(agg_frames))
        return {"df": agg_frames[join_key].filter(full_df_filters["post"]).select(full_df_select["post"])}

    if cfg.get("join_back", True):
        for keys, agg_frame in agg_frames.items():
            agg_frame = agg_frame.lazy() if isinstance(agg_frame, pl.DataFrame) else agg_frame
            df = df.join(agg_frame, on=keys, how='semi')

        for keys, agg_frame in agg_frames.items():
            agg_frame = agg_frame.lazy() if isinstance(agg_frame, pl.DataFrame) else agg_frame
            df = df.join(agg_frame, on=keys, how='inner')

        return {"df": df.filter(full_df_filters["post"]).select(full_df_select["post"])}

    if "filters" not in cfg:  # means that no changes made to input df
        return {"agg_frames": agg_frames}

    return {"df": df.filter(full_df_filters["post"]).select(full_df_select["post"]), "agg_frames": agg_frames}

def process_data(frames: list[pl.LazyFrame], cfg, df_accum=None):
    assert df_accum is None or (cfg.get("incremental_accum", False) and len(frames) == 1)
    
    if df_accum is not None and cfg.get("incremental_accum", False) and len(frames) == 1:
        return [make_df(pl.concat([frames[0], df_accum]), cfg)]

    return [make_df(elem, cfg) for elem in frames]


def parse_path_in(path_in) -> tuple[str, list[str]]:
    """
        returns a tuple of (dir_in, filename_parts): dir prefix and individual filenames
    """
    path_in_type = utils.path_type(path_in)
    if path_in_type == utils.PathType.IS_FILE:
        return Path(path_in).parent, [Path(path_in).name]
    if path_in_type == utils.PathType.IS_DIR:
        return path_in, sorted(list(filter(lambda s: s.startswith("part_"), utils.listdir(path_in))))
    
    dir_in = Path(path_in).parent
    return dir_in, sorted(utils.listdir(dir_in, glob_pattern=Path(path_in).name))  # todo support glob pattern in listdir


def assert_paths_type_match(cfg):
    fin, fout = cfg["in_artifacts"]["filename_in"], cfg["out_artifacts"]["filename_out"]
    merge_inputs = cfg["in_artifacts"].get("do_merge", False)

    # todo: implement writing to multiple directories
    assert isinstance(fout, str)

    partition_outputs = (cfg["out_artifacts"].get("partition_args", None) is not None)
    do_partition = partition_outputs is not None

    # treat fin as dir in case of multidir input for asserts' purpose
    fin_type = utils.PathType.IS_DIR if isinstance(fin, list) else utils.path_type(fin)
    fout_type = utils.path_type(fout)

    
    both_files = (fin_type == utils.PathType.IS_FILE and fout_type == utils.PathType.IS_FILE)
    both_dirs = (
        (fin_type == utils.PathType.IS_DIR or fin_type == utils.PathType.IS_GLOB)
        and fout_type == utils.PathType.IS_DIR
        and not(merge_inputs ^ do_partition)
    )
    merged_to_file = (
        (fin_type == utils.PathType.IS_DIR or fin_type == utils.PathType.IS_GLOB)
        and fout_type == utils.PathType.IS_FILE
        and merge_inputs
    )
    partition_to_dir = (
        (((fin_type == utils.PathType.IS_DIR or fin_type == utils.PathType.IS_GLOB) and merge_inputs) or fin_type == utils.PathType.IS_FILE)
        and fout_type == utils.PathType.IS_DIR
        and do_partition
    )

    assert both_files or both_dirs or merged_to_file or partition_to_dir, "filename_in/filename_out type mismatch"


class DataTransformStage(BaseStage):
    def assert_args_in_cfg(self, cfg):
        assert "in_artifacts" in self.cfg
        assert "filename_in" in self.cfg["in_artifacts"]

        assert "kwargs" in self.cfg
        assert "cfg" in self.cfg["kwargs"]

        assert "out_artifacts" in self.cfg
        assert "filename_out" in self.cfg["out_artifacts"]

        assert_paths_type_match(self.cfg)

        has_exactly_one_agg = (
            "features" in self.cfg["kwargs"]["cfg"]
            and (self.cfg["kwargs"]["cfg"]["features"].get("aggregations", False)
            and len(self.cfg["kwargs"]["cfg"]["features"]["aggregations"]) == 1)
        )
        overwrite_df_in = self.cfg["kwargs"]["cfg"].get("overwrite_df_in", False)
        incremental_accum = self.kwargs["cfg"].get("incremental_accum", False)

        # if either is true, require exactly one aggregation frame description
        assert (
            (not overwrite_df_in and not incremental_accum) or has_exactly_one_agg
        )

        # check that if either is set, then all is set
        filename_accum = self.cfg["in_artifacts"].get("filename_accum", None)
        assert (
            (not incremental_accum and filename_accum is None) or (filename_accum is not None and filename_accum == self.cfg["out_artifacts"]["filename_out"])
        )

    def load_artifacts(self):
        paths_in = self.cfg["in_artifacts"]["filename_in"]
        df_accum = None
        if "filename_accum" in self.cfg["in_artifacts"]:
            filename_accum = self.cfg["in_artifacts"]["filename_accum"]
            df_accum = utils.scan_parquet(filename_accum)
        res = {"df_accum": df_accum}

        if isinstance(paths_in, str):
            paths_in = [paths_in]

        parts = []
        for path_in in paths_in:
            dir_in, part_filenames = parse_path_in(path_in)
            for part_filename in part_filenames:
                logger.debug(f"scanning {part_filename} from {dir_in}...")
                read_path = os.path.join(dir_in, part_filename)
                if self.cfg["in_artifacts"].get("do_merge", False):
                    parts.append(read_path)
                else:
                    parts.append(utils.scan_parquet(read_path))
        
        frames = [utils.scan_parquet(parts)] if self.cfg["in_artifacts"].get("do_merge", False) else parts

        return {**res, "frames": frames}

    
    def write_artifacts(self, res: list[pl.LazyFrame | pl.DataFrame] | list[dict[tuple, pl.LazyFrame | pl.DataFrame]]):
        super().write_artifacts(res)
        paths_in = self.cfg["in_artifacts"]["filename_in"]
        path_out = self.cfg["out_artifacts"]["filename_out"]
        partition_args = self.cfg["out_artifacts"].get("partition_args", None)
        partition_args = utils.parse_partition_args(partition_args)

        if isinstance(paths_in, str):
            paths_in = [paths_in]

        part_filenames = []
        for path_in in paths_in:
            _, cur_part_filenames = parse_path_in(path_in)
            part_filenames.extend(cur_part_filenames)

        dir_out, out_filenames = path_out, part_filenames
        if utils.path_type(path_out) == utils.PathType.IS_FILE:
            dir_out, out_filenames = Path(path_out).parent, [Path(path_out).name]

        if utils.path_type(path_out) == utils.PathType.IS_DIR and partition_args is not None:
            dir_out, out_filenames = path_out, [""]

        # check case where paths_in is a list
        # todo implement writing to multiple directories
        assert len(out_filenames) == len(set(out_filenames)), "duplicate filenames are not allowed when using multidir input"
  
        for elem, out_filename in zip(res, out_filenames):
            write_path = os.path.join(dir_out, out_filename)
            logger.info(f"{'#'*20}\nProcessing {write_path}...\n")

            if "df" in elem:  # otherwise means no changes made to df -> skip writing it
                slice_partition_args = partition_args["df"] if partition_args else None
                utils.sink_parquet(elem["df"], write_path, remove_local=self.remove_local, log_artifact=self.log_artifacts, partition_args=slice_partition_args)

            if "agg_frames" in elem:
                # case of join_back: False
                for join_keys, df in elem["agg_frames"].items():
                    logger.debug(f"processing {join_keys} agg part...")
                    keys_subdir = "_".join(sorted(join_keys))
                    write_path = os.path.join(dir_out, keys_subdir, out_filename)
                    slice_partition_args = partition_args[join_keys] if partition_args else None
                    utils.sink_parquet(df, write_path, remove_local=self.remove_local, log_artifact=self.log_artifacts, partition_args=partition_args[join_keys] if partition_args else None)

    def parse_kwargs(self):
        return self.cfg["kwargs"]

    def _func(self, *args, **kwargs):
        return process_data(*args, **kwargs)

def maybe_collect(df, eager=False):
    return df.collect() if eager and isinstance(df, pl.LazyFrame) else df

def join_tables(frames: list[pl.DataFrame | pl.LazyFrame], join_tables: OrderedDict[str, dict[str, Any]], eager_execution=False, transform_join_tables=None):
    if transform_join_tables is not None:
        for join_id, tr in transform_join_tables.items():
            selects = parse_select_cfg(tr)
            filters = parse_filters_cfg(tr)
            join_tables[join_id]["df"] = (
                join_tables[join_id]["df"]
                .filter(filters["pre"]).select(selects["pre"])
                .filter(filters["post"]).select(selects["post"])
            )

    for idx in range(len(frames)):
        for join_id, jt in join_tables.items():
            frames[idx] = maybe_collect(frames[idx], eager_execution).join(
                maybe_collect(jt["df"], eager_execution), on=jt["join_key"], how=jt["join_type"]
            )
    return frames      


class JoinTablesStage(BaseStage):
    def assert_args_in_cfg(self, cfg):
        assert "in_artifacts" in self.cfg
        assert "filename_in" in self.cfg["in_artifacts"]
        assert "join_tables" in self.cfg["in_artifacts"]
        assert isinstance(self.cfg["in_artifacts"]["join_tables"], list), "must be a list of dicts"

        assert "kwargs" in self.cfg

        assert "out_artifacts" in self.cfg
        assert "filename_out" in self.cfg["out_artifacts"]

        assert_paths_type_match(self.cfg)

    def load_artifacts(self):
        path_in = self.cfg["in_artifacts"]["filename_in"]

        res = {
            "join_tables": OrderedDict([
                (
                    jt["join_id"],
                    {
                        "df": utils.scan_parquet(jt["table_path"]),
                        "join_key": jt["join_key"],
                        "join_type": jt["join_type"],
                    }
                )
                for jt in self.cfg["in_artifacts"]["join_tables"]
            ])
        }

        dir_in, part_filenames = parse_path_in(path_in)
        parts = []
        for part_filename in part_filenames:
            logger.debug(f"scanning {part_filename} from {dir_in}...")
            read_path = os.path.join(dir_in, part_filename)
            if self.cfg["in_artifacts"].get("do_merge", False):
                parts.append(read_path)
            else:
                parts.append(utils.scan_parquet(read_path))
        
        frames = [utils.scan_parquet(parts)] if self.cfg["in_artifacts"].get("do_merge", False) else parts

        return {**res, "frames": frames}

    
    def write_artifacts(self, res: list[pl.LazyFrame | pl.DataFrame] | list[dict[tuple, pl.LazyFrame | pl.DataFrame]]):
        super().write_artifacts(res)
        path_in = self.cfg["in_artifacts"]["filename_in"]
        path_out = self.cfg["out_artifacts"]["filename_out"]
        partition_args = self.cfg["out_artifacts"].get("partition_args", None)
        partition_args = utils.parse_partition_args(partition_args)

        _, part_filenames = parse_path_in(path_in)
        dir_out, out_filenames = path_out, part_filenames
        if utils.path_type(path_out) == utils.PathType.IS_FILE:
            dir_out, out_filenames = Path(path_out).parent, [Path(path_out).name]

        if utils.path_type(path_out) == utils.PathType.IS_DIR and partition_args is not None:
            dir_out, out_filenames = path_out, [""]
            
        for elem, out_filename in zip(res, out_filenames):
            write_path = os.path.join(dir_out, out_filename)
            logger.info(f"{'#'*20}\nProcessing {write_path}...\n")
            utils.sink_parquet(elem, write_path, remove_local=self.remove_local, log_artifact=self.log_artifacts, partition_args=partition_args["df"] if partition_args else None)

    def parse_kwargs(self):
        return self.cfg["kwargs"]

    def _func(self, *args, **kwargs):
        return join_tables(*args, **kwargs)


class SequentialStage(BaseStage):
    def __init__(self, cfg, stage_class, func=None, run_name=None):
        """
            `cfg` - config object
            `func` - callable function, returns a result which is then written as artifacts to disk
            `stage_class` - class to build with
        """
        self.stage_class = stage_class
        super().__init__(cfg, func, run_name)

    def assert_args_in_cfg(self, cfg):
        return self.stage_class.assert_args_in_cfg(self, cfg)

    def parse_kwargs(self):
        return self.stage_class.parse_kwargs(self)

    def make_children_stages(self):
        path_in = self.cfg["in_artifacts"]["filename_in"]
        path_out = self.cfg["out_artifacts"]["filename_out"]

        dir_in, part_filenames = parse_path_in(path_in)
        dir_out = Path(path_out).parent if utils.path_type(path_out) == utils.PathType.IS_FILE else path_out

        logger.debug(f"making children stages for running on directory: {dir_in}, files: {part_filenames}")
        children_stages = []
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
    def assert_args_in_cfg(self, cfg):
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
        utils.sink_parquet(run_result, path_out, remove_local=self.remove_local, log_artifact=self.log_artifacts)

    def _func(self, *args, **kwargs):
        return make_empty_df(*args, **kwargs)
