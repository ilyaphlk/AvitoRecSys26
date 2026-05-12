import polars as pl
import os
import sys
from pathlib import Path
from loguru import logger
from datetime import datetime
from debug_constants import DEBUG_ARGV_MAKE_TRAIN, ARGV_MAKE_TRAIN_SEPARATE
from utils import load_config
from stage import BaseStage
import mlflow


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

def make_df(df, cfg):
    agg_frames = make_aggregations(df, cfg)
    filters = make_filters(cfg)

    collected_agg_frames = dict()
    for keys, agg_frame in agg_frames.items():
        valid_filters = {fname: f for fname, f in filters.items() if fname in agg_frame.collect_schema().names()}
        agg_frame = agg_frame.filter(*valid_filters.values())
        collected_agg_frames[keys] = agg_frame.collect()

    for keys, agg_frame in collected_agg_frames.items():
        df = df.join(agg_frame.lazy(), on=keys, how='semi')

    for keys, agg_frame in collected_agg_frames.items():
        df = df.join(agg_frame.lazy(), on=keys, how='inner')

    df = df.filter(*filters.values())

    return df

def process_data(df: pl.LazyFrame | list[pl.LazyFrame], cfg):
    if isinstance(df, list):
        return [make_df(elem, cfg) for elem in df]
    return make_df(df, cfg)

class DataTransformStage(BaseStage):
    def assert_args_in_cfg(self):
        assert "in_artifacts" in self.cfg
        assert "filename_in" in self.cfg["in_artifacts"]

        assert "kwargs" in self.cfg
        assert "cfg" in self.cfg["kwargs"]

        assert "out_artifacts" in self.cfg
        assert "filename_out" in self.cfg["out_artifacts"]

        assert (
            os.path.isfile(self.cfg["in_artifacts"]["filename_in"]) == os.path.isfile(self.cfg["out_artifacts"]["filename_out"])
            or not os.path.exists(self.cfg["out_artifacts"]["filename_out"])
        )

    def load_artifacts(self):
        path_in = self.cfg["in_artifacts"]["filename_in"]
        if os.path.isdir(path_in):
            res = []
            for part_filename in sorted(os.listdir(path_in)):
                logger.debug(f"scanning {part_filename} from {path_in}...")
                res.append(pl.scan_parquet(os.path.join(path_in, part_filename)))
            return {"df": res}
        else:
            logger.debug(f"scanning {path_in}...")
            return {"df": pl.scan_parquet(path_in)}
    
    def write_artifacts(self, df: pl.LazyFrame | list[pl.LazyFrame]):
        super().write_artifacts(df)
        path_in = self.cfg["in_artifacts"]["filename_in"]
        path_out = self.cfg["out_artifacts"]["filename_out"]
        if isinstance(df, list):
            part_filenames = sorted(os.listdir(path_in))
            for elem, part_filename in zip(df, part_filenames):
                logger.info(f"{'#'*20}\nProcessing {part_filename} from {path_in}...\n")
                elem.sink_parquet(
                    os.path.join(path_out, part_filename)
                )
        else:
            logger.info(f"{'#'*20}\nProcessing {path_in}...\n")
            df.sink_parquet(path_out)

    def parse_kwargs(self):
        return self.cfg["kwargs"]


def main():
    assert len(sys.argv) == 2, "please provide path to stage yaml config as an argument"
    preprocess_config_path = sys.argv[1]
    # preprocess_config_path = "/project/workspace/config/data/eval/unique_users_cnt_by_item_id.yml"

    preprocess_cfg = load_config(preprocess_config_path)["aggregate"]

    logger.info("starting pipeline...")

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("transform_data")

    with mlflow.start_run(run_name="data_transform_pipeline"):
        preproc_stage = DataTransformStage(preprocess_cfg, process_data)
        preproc_stage.run()
        logger.info("transformed data successfully.")


if __name__ == "__main__":
    main()
