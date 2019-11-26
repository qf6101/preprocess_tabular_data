import ruamel.yaml as yaml
from typing import List, Optional
from pandas import DataFrame, Series
from sklearn.preprocessing import LabelBinarizer, KBinsDiscretizer, OrdinalEncoder, StandardScaler
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn_pandas import DataFrameMapper
from sklearn2pmml import sklearn2pmml
from xgboost.sklearn import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn import metrics
import numpy as np
from sklearn2pmml.decoration import ContinuousDomain, CategoricalDomain


def load_preprocess_conf(fname: str) -> List[object]:
    with open(fname, 'r') as f:
        loaded = yaml.safe_load(f)
        loaded = [attr['attribute'] for attr in loaded]
        loaded = {attr['name']: attr for attr in loaded}
    return loaded


def preprocess_dataframe(df: DataFrame, inspected: str, supervised: str = None, skip_time: bool = False,
                         y_column: Optional[str] = None, onehot_enc=True, normalize=True):
    inspected = load_preprocess_conf(inspected)
    if supervised:
        supervised = load_preprocess_conf[supervised]
    else:
        supervised = {}

    attr_preprocs = []

    for c in df.columns:
        if y_column and c == y_column:
            continue

        if c in supervised:
            attr_info = supervised[c]
        elif c in inspected:
            attr_info = inspected[c]
        else:
            continue

        if skip_time and attr_info['btype'] == 'time':
            continue
        else:
            missing_replace = attr_info['operations'][0]['fill_missing']

            if attr_info['btype'] == 'categorical':
                if onehot_enc:
                    attr_preprocs.append(([c], [CategoricalDomain(with_data=False, with_statistics=False,
                                                                  invalid_value_treatment='as_missing',
                                                                  missing_value_treatment='as_value',
                                                                  missing_values=['-', '/'],
                                                                  missing_value_replacement=missing_replace),
                                                SimpleImputer(strategy='constant', fill_value=missing_replace),
                                                LabelBinarizer()]))
                else:
                    attr_preprocs.append(([c], [CategoricalDomain(with_data=False, with_statistics=False,
                                                                  invalid_value_treatment='as_missing',
                                                                  missing_values=['-', '/'],
                                                                  missing_value_treatment='as_value',
                                                                  missing_value_replacement=missing_replace),
                                                SimpleImputer(strategy='constant', fill_value=missing_replace),
                                                OrdinalEncoder()]))
            elif attr_info['btype'] == 'realvalued':
                if normalize:
                    attr_preprocs.append(([c], [ContinuousDomain(with_data=False, with_statistics=False,
                                                                 invalid_value_treatment='as_missing',
                                                                 missing_values=['-', '/'],
                                                                 missing_value_treatment='as_value',
                                                                 missing_value_replacement=missing_replace),
                                                SimpleImputer(strategy='constant', fill_value=missing_replace),
                                                StandardScaler()]))
                else:
                    attr_preprocs.append(([c], [ContinuousDomain(with_data=False, with_statistics=False,
                                                                 invalid_value_treatment='as_missing',
                                                                 missing_values=['-', '/'],
                                                                 missing_value_treatment='as_value',
                                                                 missing_value_replacement=missing_replace),
                                                SimpleImputer(strategy='constant', fill_value=missing_replace)]))
            elif attr_info['btype'] == 'bucketizing':
                attr_preprocs.append(([c], [ContinuousDomain(with_data=False, with_statistics=False,
                                                             invalid_value_treatment='as_missing',
                                                             missing_values=['-', '/'],
                                                             missing_value_treatment='as_value',
                                                             missing_value_replacement=missing_replace),
                                            SimpleImputer(strategy='median'),
                                            KBinsDiscretizer(n_bins=11, encode='ordinal', strategy='quantile')]))

    return attr_preprocs


def train_model(X: DataFrame, y: Series, attr_preprocs, pmml_fname, n_estimators):
    pipeline = PMMLPipeline([
        ("attribute_preprocessor", DataFrameMapper(attr_preprocs)),
        ('classifier', XGBClassifier(n_gpus=0, objective="binary:logistic", n_jobs=30, max_depth=2,
                                     n_estimators=n_estimators, colsample_bytree=0.5, colsample_bylevel=0.5,
                                     colsample_bynode=0.5, subsample=0.5, reg_alpha=0.8, reg_lambda=2, missing=-99998))
    ])

    pipeline.fit(X, y)
    prob = pipeline.predict_proba(X)
    fpr, tpr, thresholds = metrics.roc_curve(y, prob[:, 1], pos_label=1)
    auc = metrics.auc(fpr, tpr)
    ks = np.max(tpr - fpr)

    sklearn2pmml(pipeline, pmml_fname, with_repr=True)

    return pipeline, auc, ks


def preprocess_X(X: DataFrame, attr_preprocs):
    dfm = DataFrameMapper(attr_preprocs).fit(X)
    outp_X = dfm.transform(X)
    return dfm, outp_X
