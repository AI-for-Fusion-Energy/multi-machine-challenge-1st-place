import os
import glob
import h5py

import numpy as np
import pandas as pd

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import EfficientFCParameters

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression

# -----------------------------------------------------------------------------

cmod_data_folder = os.path.join('.', 'cmod')

# -----------------------------------------------------------------------------

def get_shot_id(fname):
    return int(os.path.splitext(os.path.split(fname)[-1])[0])

def get_files(folder):
    path = os.path.join(folder, '**', '*.hdf5')
    files = glob.glob(path, recursive=True)
    if len(files) == 0:
        print('No files in', path)
        print('Please check your data folders.')
        exit()
    files = sorted(files, key=get_shot_id)
    return files

def get_data(files):
    data = dict()
    labels = dict()
    for fname in files:
        shot_id = get_shot_id(fname)
        data[shot_id] = dict()
        print('Reading:', fname)
        f = h5py.File(fname, 'r')
        for tag in f['data']:
            array = np.fabs(f['data'][tag][:])
            array = array[~np.isnan(array)]
            array = array[~np.isinf(array)]
            if len(array) > 0:
                data[shot_id][tag] = array
        labels[shot_id] = float(f['meta']['IsDisrupt'][()])
        f.close()
    return data, labels

def get_series(data, labels, tag):
    X = []
    y = []
    for shot_id in data:
        if tag not in data[shot_id]:
            continue
        n = len(data[shot_id][tag])
        X.append(pd.DataFrame({'shot': [shot_id]*n,
                               'time': np.arange(n),
                               'data': data[shot_id][tag]}))
        y.append(pd.Series({shot_id: labels[shot_id]}))
    X = pd.concat(X, axis=0)
    y = pd.concat(y, axis=0)
    return X, y

def get_features(X):
    df = extract_features(X,
                          default_fc_parameters=EfficientFCParameters(),
                          column_id='shot',
                          column_sort='time',
                          disable_progressbar=True,
                          impute_function=impute)
    return df

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    folder = cmod_data_folder
    files = get_files(folder)
    data, labels = get_data(files)

    train_ids = [shot_id for shot_id in labels if not np.isnan(labels[shot_id])]
    test_ids = [shot_id for shot_id in labels if np.isnan(labels[shot_id])]

    tags = sorted(set([tag for shot_id in data for tag in data[shot_id]]))

    coefficients = dict()
    predictions = dict()

    for tag in tags:
        print('Processing:', tag)
        X, y = get_series(data, labels, tag)
        X = get_features(X)

        scaler = StandardScaler()
        scaler.fit(X)
        X = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)

        select = VarianceThreshold(1e-6)
        select.fit(X)
        columns = select.get_feature_names_out()
        X = pd.DataFrame(select.transform(X), index=X.index, columns=columns)

        if X.empty:
            continue

        fit_ids = sorted(set(X.index) & set(train_ids))
        pred_ids = sorted(set(X.index) & set(test_ids))

        if len(fit_ids) == 0:
            continue
        if len(pred_ids) == 0:
            continue

        X_fit = X.loc[fit_ids]
        y_fit = y.loc[fit_ids]

        if y_fit.nunique() < 2:
            continue

        classifier = LogisticRegression(penalty=None, max_iter=1000)
        classifier.fit(X_fit, y_fit)

        coefficients[tag] = dict(zip(X.columns, classifier.coef_[0]))

        X_pred = X.loc[pred_ids]
        y_pred = classifier.predict_proba(X_pred)[:,1]

        predictions[tag] = dict(zip(pred_ids, y_pred))

    coefficients = pd.DataFrame.from_dict(coefficients).sort_index()
    predictions = pd.DataFrame.from_dict(predictions).sort_index()

    predictions['IsDisrupt'] = [labels[shot_id] for shot_id in predictions.index]

    fname = 'coefficients.csv'
    print('Writing:', fname)
    coefficients.to_csv(fname)

    fname = 'predictions.csv'
    print('Writing:', fname)
    predictions.to_csv(fname)
