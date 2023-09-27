from sklearn.linear_model import Lasso
import pandas as pd


def impute_matrix_with_colrow_mean(input_matrix):
    # Preprocess feature values by replacing NA with column mean, then row mean (when a whole column is missing).
    output_matrix = input_matrix.copy()
    output_matrix = output_matrix.fillna(output_matrix.mean())
    if output_matrix.isna().any().any():
        # Equivalent to pd.DataFrame.fillna(axis=1) which is not implemented yet in this version.
        output_matrix = output_matrix.T.fillna(output_matrix.T.mean()).T
    return output_matrix


def linear_collaborative_filtering_on_rows_lasso(sparse_mat, feat_cols, alpha, seed, coef_file=None):
    full_mat = []
    coef_dict = dict()
    for pred_col in sparse_mat.columns:
        # Add a constant column to fit intercept and avoid error when len(feat_cols) == 1.
        cfeat_cols = [col for col in feat_cols if col != pred_col] + ['intercept']
        full_feat_mat = sparse_mat.copy()
        full_feat_mat['intercept'] = 1
        full_feat_mat[cfeat_cols] = impute_matrix_with_colrow_mean(full_feat_mat[cfeat_cols])

        estimator = Lasso(fit_intercept=False, random_state=seed, alpha=alpha, max_iter=1e4)
        # For rows with non-missing `pred_col`, use as training data.
        tr_mat = full_feat_mat[full_feat_mat[pred_col].notna()]
        if len(tr_mat) != 0:  
            estimator.fit(tr_mat[cfeat_cols], tr_mat[pred_col])
        else:
            # If all missing, i.e. no training label is available, build a row-mean regressor.
            row_mean = full_feat_mat[[col for col in cfeat_cols if col != 'intercept']].mean(axis=1) 
            estimator.fit(full_feat_mat[cfeat_cols], row_mean)

        # Return a full matrix predicted by the estimator, including predicted training data.
        imp_pred = full_feat_mat[[pred_col]].copy()
        imp_pred[pred_col] = estimator.predict(full_feat_mat[cfeat_cols])
        full_mat.append(imp_pred)
        
        # Collect coef.
        coef_dict[pred_col] = dict(zip(cfeat_cols, estimator.coef_))
    full_mat = pd.concat(full_mat, axis=1)
    if coef_file is not None:
        pd.DataFrame(coef_dict).to_csv(coef_file)
    return full_mat
