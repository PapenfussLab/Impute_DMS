import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, pearsonr


def impute_missing_value(data, categ_feat, numer_feat):
    """Impute the missing values in the input data.

    Parameters
    ----------
    data: pd.DataFrame
        Data with missing values.
    categ_feat: list or None
        List of categorical features which will be imputed by most frequent value. None means no
        categorical features need to be imputed.
    numer_feat: list or None
        List of numerical features which will be imputed by mean value. None means no categorical
        features need to be imputed.

    Returns
    -------
    clean_data: pd.DataFrame
        Data with imputed missing values.
    """
    clean_data = data.copy().reset_index(drop=True)

    if numer_feat is not None:
        imputer = SimpleImputer(strategy="mean")
        impute_result = pd.DataFrame(
            imputer.fit_transform(clean_data[numer_feat]), columns=numer_feat
        )
        clean_data[numer_feat] = impute_result

    if categ_feat is not None:
        mode_values = clean_data[categ_feat].mode().iloc[0]
        clean_data[categ_feat] = clean_data[categ_feat].fillna(mode_values)
    clean_data.index = data.index
    return clean_data


def encode_categorical_feature(data, encode_col, retain_col=None):
    """Encode categorical features into one hot sparse matrix.

    The function will change the input columns to a one hot sparse DataFrame whose columns
    are the possible values of those features and 1.0 will be assigned to the rows with
    corresponding value otherwise being 0.

    Parameters
    ----------
    data: pd.DataFrame
        Data whose columns are all categorical features to be transformed.
    encode_col: list
        List of categorical features which will be encoded with one hot encoder.
    retain_col: list, optional (default=None)
        If it is not None, the according columns in data will be retained, keeping both the encoded
        and original feature values.

    Returns
    -------
    encoded_data: pd.DataFrame
    encoded_feat_col: list
        A list of encoded categorical feature names.

    Example
    -------
    >>> encode_categorical_feature(pd.DataFrame({'one':['a', 'b'], 'two': ['A', 'B']}))
           one_a  one_b  two_A  two_B
    0    1.0    0.0    1.0    0.0
    1    0.0    1.0    0.0    1.0
    """
    work_data = data.copy().reset_index(drop=True)

    encoder = OneHotEncoder(sparse=False)
    encoded_result = pd.DataFrame(encoder.fit_transform(work_data[encode_col]))
    encoded_feat_col = encoded_result.columns = list(
        encoder.get_feature_names(encode_col)
    )

    encoded_data = pd.concat(
        [work_data.drop(encode_col, axis=1), encoded_result], axis=1
    )
    encoded_data.index = data.index

    if retain_col is not None:
        encoded_data[retain_col] = data[retain_col]
    return encoded_data, encoded_feat_col


def create_dms_score_matrix(long_dms, wt_score, aas='ACDEFGHIKLMNPQRSTVWY'):
    """Transform long-format DMS data to a wide-format score matrix with assigned wildtype scores.

    Return a score_matrix whose rows being distinct positions tested in each DMS and columns 
    being variant type amino acids, with DMS socres in each cell. Wild-type socres are also 
    filled in corresponding cells.

    Parameters
    ----------
    long_dms: pd.DataFrame
        Long-format DMS data with at least 'dms_id', 'position', 'aa1', 'aa2' & 'score' columns.
    wt_score: int or float
        Assigned DMS scores for wildtype (or synonymous variants).
    aas: str
        Order of amino acids.

    Returns
    -------
    score_matrix: pd.DataFrame
    """
    wt_data = long_dms[['dms_id', 'position', 'aa1']].copy().drop_duplicates()
    wt_data['score'] = wt_score
    wt_data['aa2'] = wt_data['aa1']
    wt_dms = pd.concat([long_dms, wt_data], ignore_index=True)
    score_mat = wt_dms.pivot_table(columns='aa2', index=['dms_id', 'position'], values='score')
    for aa in aas:
        if aa not in score_mat.columns:
            score_mat[aa] = np.nan
    return score_mat


def concat_dms_score_matrix(input_mat, extra_wt):
    # Add residues only available in the testing data. This happens when the data is exremely sparse.
    extra_wt['score'] = 1
    extra_wt = extra_wt.pivot_table(index=['dms_id', 'position'], columns='aa1', values='score')
    output_mat = pd.concat([input_mat, extra_wt], ignore_index=False)
    return output_mat


def evaluate_tr_te_result(tr_pred, te_pred):
    eval_dict = dict()
    eval_dict["Train Spearman's ρ"] = spearmanr(tr_pred['pred_score'], tr_pred['score'])[0]
    eval_dict["Test Spearman's ρ"] = spearmanr(te_pred['pred_score'], te_pred['score'])[0]
    eval_dict["Train Pearson's ρ"] = pearsonr(tr_pred['pred_score'], tr_pred['score'])[0]
    eval_dict["Test Pearson's ρ"] = pearsonr(te_pred['pred_score'], te_pred['score'])[0]
    eval_dict['Train RMSE'] = np.sqrt(mean_squared_error(tr_pred['pred_score'], tr_pred['score']))
    eval_dict['Test RMSE'] = np.sqrt(mean_squared_error(te_pred['pred_score'], te_pred['score']))
    return eval_dict
