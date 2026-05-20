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
        agg_frames[tuple(sorted(keys))] = df.group_by(keys).agg(exprs)

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

    agg_frames = make_aggregations(df, cfg)
    filters = make_filters(cfg["filters"]) if "filters" in cfg else [pl.lit(True)]  # .. else do not filter anything

    if cfg.get("eager_execution", False):
        agg_frames = {k: v.collect() for k, v in agg_frames.items()}

    if cfg.get("join_back", True):
        for keys, agg_frame in agg_frames.items():
            agg_frame = agg_frame.lazy() if isinstance(agg_frame, pl.DataFrame) else agg_frame
            df = df.join(agg_frame, on=keys, how='semi')

        for keys, agg_frame in agg_frames.items():
            agg_frame = agg_frame.lazy() if isinstance(agg_frame, pl.DataFrame) else agg_frame
            df = df.join(agg_frame, on=keys, how='inner')

        df = df.filter(filters)

        return {"df": df}
    
    if overwrite_df_in:
        join_key = next(iter(agg_frames))
        return {"df": agg_frames[join_key].filter(filters)}
    
    df = df.filter(filters)

    return {"df": df, "agg_frames": agg_frames}

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
    partition_outputs = (cfg["out_artifacts"].get("partition_args", None) is not None)
    do_partition = partition_outputs is not None
    fin_type, fout_type = utils.path_type(fin), utils.path_type(fout)

    
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
    def assert_args_in_cfg(self):
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
        path_in = self.cfg["in_artifacts"]["filename_in"]
        df_accum = None
        if "filename_accum" in self.cfg["in_artifacts"]:
            filename_accum = self.cfg["in_artifacts"]["filename_accum"]
            df_accum = utils.scan_parquet(filename_accum)
        res = {"df_accum": df_accum}

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
            slice_partition_args = partition_args["df"] if partition_args else None
            utils.sink_parquet(elem["df"], write_path, remove_local=False, log_artifact=True, partition_args=slice_partition_args)

            if "agg_frames" in elem:
                # case of join_back: False
                for join_keys, df in elem["agg_frames"].items():
                    logger.debug(f"processing {join_keys} agg part...")
                    keys_subdir = "_".join(sorted(join_keys))
                    write_path = os.path.join(dir_out, keys_subdir, out_filename)
                    slice_partition_args = partition_args[join_keys] if partition_args else None
                    utils.sink_parquet(df, write_path, remove_local=False, log_artifact=True, partition_args=partition_args[join_keys] if partition_args else None)

    def parse_kwargs(self):
        return self.cfg["kwargs"]

def maybe_collect(df, eager=False):
    return df.collect() if eager and isinstance(df, pl.LazyFrame) else df

def join_tables(frames, join_tables, eager_execution=False):
    for idx in range(len(frames)):
        for jt in join_tables:
            frames[idx] = maybe_collect(frames[idx], eager_execution).join(
                maybe_collect(jt["df"], eager_execution), on=jt["join_key"], how=jt["join_type"]
            )
    return frames      


class JoinTablesStage(BaseStage):
    def assert_args_in_cfg(self):
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
            "join_tables": [
                {"df": utils.scan_parquet(jt["name"]), "join_key": jt["join_key"], "join_type": jt["join_type"]}
                for jt in self.cfg["in_artifacts"]["join_tables"]
            ]
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
            utils.sink_parquet(elem, write_path, remove_local=False, log_artifact=True)

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
        utils.sink_parquet(run_result, path_out, remove_local=False, log_artifact=True)

def test_aggregate_combine():
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

    return stages, test_aggregate_combine.__name__

def test_dilter_df():
    config_path = "/project/workspace/config/data/eval/dt_filter_raw_train_debug.yml"
    filter_cfg = load_config(config_path)["filter_by_dt"]

    stages = [
        SequentialStage(filter_cfg, DataTransformStage, process_data)
    ]

    return stages, test_dilter_df.__name__

def test_aggregate_tables():
    config_path = "/project/workspace/config/data/features/counters_local_shows_clicks_debug.yml"
    agg_cfg = load_config(config_path)["aggregate_partitions"]

    stages = [
        SequentialStage(agg_cfg, DataTransformStage, process_data)
    ]

    return stages, test_aggregate_tables.__name__

def test_make_blacklist():
    config_path = "/project/workspace/config/data/features/counters_local_shows_clicks_debug.yml"
    accum_cfg = load_config(config_path)["make_accum_item_id"]
    blacklist_cfg = load_config(config_path)["make_item_id_blacklist_sequential"]

    stages = [
        MakeAccumStage(accum_cfg, make_empty_df),
        SequentialStage(blacklist_cfg, DataTransformStage, process_data),
    ]

    return stages, test_make_blacklist.__name__

def test_join_tables():
    config_path = "/project/workspace/config/data/features/counters_local_shows_clicks_debug.yml"
    join_cfg = load_config(config_path)["filter_by_blacklists"]

    stages = [
        SequentialStage(join_cfg, JoinTablesStage, join_tables),
    ]

    return stages, test_join_tables.__name__

def full_whitelist_pipeline():
    filter_config_path = "/project/workspace/config/data/eval/dt_filter_raw_train.yml"
    filter_cfg = load_config(filter_config_path)["filter_by_dt"]

    config_path = "/project/workspace/config/data/features/counters_local_shows_clicks.yml"
    agg_cfg = load_config(config_path)["aggregate_partitions"]
    accum_item_cfg = load_config(config_path)["make_accum_item_id"]
    blacklist_item_cfg = load_config(config_path)["make_item_id_blacklist_sequential"]
    accum_user_cfg = load_config(config_path)["make_accum_user_id"]
    blacklist_user_cfg = load_config(config_path)["make_user_id_blacklist_sequential"]
    whitelist_by_antijoin_cfg = load_config(config_path)["filter_by_blacklists"]

    stages = OrderedDict([
        ("dt_filter", SequentialStage(filter_cfg, DataTransformStage, process_data, run_name="dt_filter")),
        ("make_agg", SequentialStage(agg_cfg, DataTransformStage, process_data, run_name="make_agg")),
        ("make_accum_item", MakeAccumStage(accum_item_cfg, make_empty_df, run_name="make_accum_item")),
        ("blacklist_item", SequentialStage(blacklist_item_cfg, DataTransformStage, process_data, run_name="blacklist_item")),
        ("make_accum_user", MakeAccumStage(accum_user_cfg, make_empty_df, run_name="make_accum_user")),
        ("blacklist_user", SequentialStage(blacklist_user_cfg, DataTransformStage, process_data, run_name="blacklist_user")),
        ("make_whitelist", SequentialStage(whitelist_by_antijoin_cfg, JoinTablesStage, join_tables, run_name="make_whitelist")),
    ])

    return stages, full_whitelist_pipeline.__name__

def full_whitelist_pipeline_aws():
    filter_config_path = "/project/workspace/config/data/eval/dt_filter_raw_train.yml"
    filter_cfg = load_config(filter_config_path)["filter_by_dt"]

    config_path = "/project/workspace/config/data/features/counters_local_shows_clicks_aws.yml"
    agg_cfg = load_config(config_path)["aggregate_partitions"]
    accum_item_cfg = load_config(config_path)["make_accum_item_id"]
    blacklist_item_cfg = load_config(config_path)["make_item_id_blacklist"]
    accum_user_cfg = load_config(config_path)["make_accum_user_id"]
    blacklist_user_cfg = load_config(config_path)["make_user_id_blacklist"]
    whitelist_by_antijoin_cfg = load_config(config_path)["filter_by_blacklists"]

    stages = OrderedDict([
        #("dt_filter", SequentialStage(filter_cfg, DataTransformStage, process_data, run_name="dt_filter")),
        #("make_agg", SequentialStage(agg_cfg, DataTransformStage, process_data, run_name="make_agg")),
        #("make_accum_item", MakeAccumStage(accum_item_cfg, make_empty_df, run_name="make_accum_item")),
        ("blacklist_item", DataTransformStage(blacklist_item_cfg, process_data, run_name="blacklist_item")),
        #("make_accum_user", MakeAccumStage(accum_user_cfg, make_empty_df, run_name="make_accum_user")),
        ("blacklist_user", DataTransformStage(blacklist_user_cfg, process_data, run_name="blacklist_user")),
        ("make_whitelist", SequentialStage(whitelist_by_antijoin_cfg, JoinTablesStage, join_tables, run_name="make_whitelist")),
    ])

    return stages, full_whitelist_pipeline_aws.__name__

def full_whitelist_pipeline_aws_debug():
    # filter_config_path = "/project/workspace/config/data/eval/dt_filter_raw_train_debug.yml"
    # filter_cfg = load_config(filter_config_path)["filter_by_dt"]

    config_path = "/project/workspace/config/data/features/counters_local_shows_clicks_aws_debug.yml"
    agg_cfg = load_config(config_path)["aggregate_partitions"]
    accum_item_cfg = load_config(config_path)["make_accum_item_id"]
    blacklist_item_cfg = load_config(config_path)["make_item_id_blacklist"]
    accum_user_cfg = load_config(config_path)["make_accum_user_id"]
    blacklist_user_cfg = load_config(config_path)["make_user_id_blacklist"]
    whitelist_by_antijoin_cfg = load_config(config_path)["filter_by_blacklists"]

    stages = OrderedDict([
        #("dt_filter", SequentialStage(filter_cfg, DataTransformStage, process_data, run_name="dt_filter")),
        #("make_agg", SequentialStage(agg_cfg, DataTransformStage, process_data, run_name="make_agg")),
        ("make_agg", DataTransformStage(agg_cfg, process_data, run_name="make_agg")),
        #("make_accum_item", MakeAccumStage(accum_item_cfg, make_empty_df, run_name="make_accum_item")),
        #("blacklist_item", DataTransformStage(blacklist_item_cfg, process_data, run_name="blacklist_item")),
        #("make_accum_user", MakeAccumStage(accum_user_cfg, make_empty_df, run_name="make_accum_user")),
        # ("blacklist_user", DataTransformStage(blacklist_user_cfg, process_data, run_name="blacklist_user")),
        # ("make_whitelist", SequentialStage(whitelist_by_antijoin_cfg, JoinTablesStage, join_tables, run_name="make_whitelist")),
    ])

    return stages, full_whitelist_pipeline_aws_debug.__name__


def test_recursive_filters():
    config_path = "/project/workspace/config/data/features/tree_filters_debug.yml"
    
    simple_or_cfg = load_config(config_path)["simple_or"]
    disj_of_conj_cfg = load_config(config_path)["disj_of_conj"]
    simple_or_w_aggregates_cfg = load_config(config_path)["simple_or_w_aggregates"]

    stages = OrderedDict([
        ("simple_or", DataTransformStage(simple_or_cfg, process_data, run_name="simple_or")),
        ("disj_of_conj", DataTransformStage(disj_of_conj_cfg, process_data, run_name="disj_of_conj")),
        ("simple_or_w_aggregates", DataTransformStage(simple_or_w_aggregates_cfg, process_data, run_name="simple_or_w_aggregates")),
    ])

    return stages, test_recursive_filters.__name__


def test(func):
    # assert len(sys.argv) == 2, "please provide path to stage yaml config as an argument"
    # preprocess_config_path = sys.argv[1]
    
    logger.debug("setting mlflow uri...")
    mlflow.set_tracking_uri("http://localhost:5000")
    logger.debug("setting mlflow exp...")
    mlflow.set_experiment(experiment_name="debug_filters")

    stages, run_name = func()

    with mlflow.start_run(run_name=run_name):
        logger.info(f"starting pipeline with stages: {list(stages.keys())}")
        for name, stage in stages.items():
            logger.info(f"running stage: {name}")
            stage.run()
        logger.info("transformed data successfully.")


if __name__ == "__main__":
    test(test_recursive_filters)
