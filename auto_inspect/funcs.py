import math
from datetime import datetime
import numpy as np
from typing import Tuple, Optional, List, Set, Dict
from pandas import Series, DataFrame
import ruamel.yaml as yaml


def inspect_dtype(col: Series, predefined_dtype=None) -> Tuple[str, bool]:
    dtype = 'int'
    all_missing = False
    all_samevalue = False

    if predefined_dtype:
        dtype = predefined_dtype
    else:
        for v in col:
            if isinstance(v, str):
                dtype = 'str'
                break

            if isinstance(v, datetime):
                dtype = 'time'
                break

            if not np.isnan(v) and math.ceil(v) > v:
                dtype = 'float'
                break

    col2 = col.dropna()

    if dtype == 'int':
        col2 = [int(v) for v in col2]
        if all([(np.isnan(v) | (v == 0)) for v in col2]):
            all_missing = True

    if dtype == 'str':
        if len(set(col)) == 1:
            all_samevalue = True
    elif dtype == 'int' or dtype == 'float':
        col2 = np.array([float(v) for v in col2])
        if len(set(col2)) == 1:
            all_samevalue = True

    return dtype, all_missing | all_samevalue


def inspect_categorical(col: Series, dtype: str, min_distinct=5) -> Optional[Set]:
    if dtype == 'str':
        feature_candidates = set(col)
        if np.nan in feature_candidates:
            feature_candidates.remove(np.nan)
        return feature_candidates
    elif dtype == 'int':
        col2 = np.array(col)
        col2 = col2[col2 != 0]
        feature_candidates = set(col2)

        if 2 < len(feature_candidates) < min_distinct:
            return feature_candidates
        elif (len(feature_candidates) == 2) and (0 in feature_candidates):
            return feature_candidates

    return None


def inspect_buckets(col: Series) -> List:
    col2 = np.array(col)
    buckets = np.percentile(col2, range(10, 100, 10))
    buckets = sorted(list(set(buckets)))
    return buckets


def inspect_standardscale(col: Series, min_diff=1000) -> Tuple[str, float, float]:
    col2 = np.array([float(v) for v in col])
    col2 = col2[~(col2 == 0)]

    if max(col2) - min(col2) > min_diff:
        col2 = np.log1p(col2)
        log_op = 'ln'
    else:
        log_op = 'direct'

    return log_op, np.mean(col2), np.std(col2)


def inspect_column(col: Series, bucketize_int=True, categorical_min_distinct=5,
                   standardscale_min_diff=1000, predefined_dtype=None) -> Dict[str, object]:
    dtype, useless = inspect_dtype(col, predefined_dtype)

    if not useless:
        kinds = inspect_categorical(col, dtype, categorical_min_distinct)

        if kinds:
            return {'btype': 'categorical', 'dtype': dtype, 'kinds': kinds}
        else:
            if bucketize_int and dtype == 'int':
                buckets = inspect_buckets(col)
                if (len(buckets) == 2 and 0 in set(buckets)) or (len(buckets) > 2):
                    return {'btype': 'bucketizing', 'dtype': dtype, 'buckets': buckets}
            elif dtype == 'int' or dtype == 'float':
                standardscale = inspect_standardscale(col, standardscale_min_diff)
                return {'btype': 'realvalued', 'dtype': dtype, 'standardscale': standardscale}
            elif dtype == 'time':
                return {'btype': 'time', 'dtype': dtype}

    return {'btype': 'useless', 'dtype': dtype}


def inspect_dataframe(df: DataFrame, skip_cols=[], bucketize_int=True, categorical_min_distinct=5,
                      standardscale_min_diff=1000, predefined_dtypes={}, skip_vals=[]) -> Dict[str, object]:
    outp = {}
    skip_cols = set(skip_cols)
    for c in df.columns:
        if c not in skip_cols:
            if c in predefined_dtypes:
                predefined_dtype = predefined_dtypes[c]
            else:
                predefined_dtype = None

            col = df[c]
            for sv in skip_vals:
                col = col.replace(to_replace=sv, value=np.nan)
            col = col.dropna()

            outp[c] = inspect_column(col, bucketize_int, categorical_min_distinct, standardscale_min_diff,
                                     predefined_dtype=predefined_dtype)
    return outp


def save_inspected(inspected: Dict[str, object], fname: str, skip_time=False) -> None:
    to_save = []

    for k, v in inspected.items():
        if v['btype'] == 'useless':
            continue
        elif skip_time and v['btype'] == 'time':
            continue
        else:
            dtype = v['dtype']

            if dtype == 'str':
                fill_missing = 'unknown'
            elif dtype == 'time':
                fill_missing = '-'
            else:
                fill_missing = -99998

            operations = [{'fill_missing': fill_missing}]

            if 'kinds' in v:
                if dtype == 'str':
                    operations.append({'kinds': [str(i) for i in v['kinds']]})
                elif dtype == 'int':
                    operations.append({'kinds': [int(i) for i in v['kinds']]})
            elif 'buckets' in v:
                operations.append({'buckets': [int(i) for i in v['buckets']]})
            elif 'standardscale' in v:
                if v['standardscale'][0] == 'ln':
                    operations.append({'log': 'ln'})
                operations.append({'mean': float(v['standardscale'][1]), 'std': float(v['standardscale'][2])})

            to_save.append({'attribute':
                                {'name': k, 'btype': v['btype'], 'dtype': v['dtype'], 'operations': operations}
                            })

    with open(fname, 'w') as f:
        yaml.round_trip_dump(to_save, f, default_flow_style=False)
