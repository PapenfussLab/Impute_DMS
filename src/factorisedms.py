import tensorflow as tf
import numpy as np
import pandas as pd


def _build_wals_model(indices, scores, n_rows, n_cols, seed, wals_params):
    """Create the WALSModel and input, row and col factor tensors.

    Parameters
    ----------
    indices: list or np.array
        A 2-D list or array of shape: N*2, which specifies the coordinates of the elements
        in the original matrix. For example, [[1, 3], [2, 4]], specifies that the elements
        with indices of [1, 3] and [2, 4].
    scores: list or np.array
        A 1-D list or array, which specifies the values of the elements given by indices in
        the original matrix.

    Returns
    -------
    input_tensor: tf.SparseTensor
        Tensor holding the input matrix.
    row_factor: tf.SparseTensor
    col_factor: tf.SparseTensor
    model: tf.contrib.factorization.WALSModel
    """
    with tf.Graph().as_default():
        # <<<<<====
        tf.set_random_seed(seed)

        # Transform the matrix to tf.SparseTensor.
        input_tensor = tf.SparseTensor(indices=indices, values=scores.astype(np.float32),
                                       dense_shape=[n_rows, n_cols])
        model = tf.contrib.factorization.WALSModel(n_rows, n_cols, **wals_params)

        # Retrieve the initial row and column factors.
        row_factor = model.row_factors[0]
        col_factor = model.col_factors[0]

    return input_tensor, row_factor, col_factor, model


def _simple_train(model, input_tensor, seed, num_updates):
    """Helper function to train model on input for num_iter.

    Parameters
    ----------
    model: tf.contrib.factorization.WALSModel
    input_tensor: tf.SparseTensor
        Tensor holding the input matrix.

    Returns
    -------
    sess: tensorflow session
        For evaluating results.
    """
    sess = tf.Session(graph=input_tensor.graph)

    with input_tensor.graph.as_default():
        # <<<<<====
        tf.set_random_seed(seed)

        row_update_op = model.update_row_factors(sp_input=input_tensor)[1]
        col_update_op = model.update_col_factors(sp_input=input_tensor)[1]

        sess.run(model.initialize_op)
        sess.run(model.worker_init)
        for i in range(num_updates):
            sess.run(model.row_update_prep_gramian_op)
            sess.run(model.initialize_row_update_op)
            sess.run(row_update_op)
            sess.run(model.col_update_prep_gramian_op)
            sess.run(model.initialize_col_update_op)
            sess.run(col_update_op)

    return sess


def _train_factorisation(indices, scores, n_rows, n_cols, seed, num_updates, **wals_param):
    """Instantiate WALS model and use "simple_train" to factorize the matrix.

    Parameters
    ----------
    indices: list or np.array
        A 2-D list or array of shape: N*2, which specifies the coordinates of the elements
        in the original matrix. For example, [[1, 3], [2, 4]], specifies that the elements
        with indices of [1, 3] and [2, 4].
    scores: list or np.array
        A 1-D list or array, which specifies the values of the elements given by indices in
        the original matrix.
    """
    input_tensor, row_factor, col_factor, model = _build_wals_model(indices, scores, n_rows, n_cols, seed, 
                                                                    wals_param)
    session = _simple_train(model, input_tensor, seed, num_updates)

    # evaluate output factor matrices
    output_row = row_factor.eval(session=session)
    # The column factor seems to be transposed. output_col should be a f*m (f*20) matrix.
    output_col = col_factor.eval(session=session).T
    session.close()
    return output_row, output_col


def _transform_df_matrix_to_full_coordinates(input_df):
    coor_df = input_df.copy()
    coor_df = coor_df.reset_index(drop=True)
    coor_df.columns = np.arange(len(coor_df.columns))
    coor_df.index.name = 'row_coor'
    
    long_df = coor_df.reset_index().melt(id_vars='row_coor', value_name='score', var_name='col_coor')
    long_df = long_df.dropna()
    indices = long_df[['row_coor', 'col_coor']].values
    scores = long_df['score'].values
    return indices, scores


def impute_by_matrix_factorisation(input_mat, seed, num_updates, **wals_param):
    """
    References
    ----------
    https://github.com/GoogleCloudPlatform/tensorflow-recommendation-wals/tree/master/wals_ml_engine
    """
    n_rows = len(input_mat)
    n_cols = len(input_mat.columns)
    indices, scores = _transform_df_matrix_to_full_coordinates(input_mat)
    output_row, output_col = _train_factorisation(indices, scores, n_rows, n_cols, seed, num_updates, **wals_param)
    full_mat = pd.DataFrame(data=np.matmul(output_row, output_col), columns=input_mat.columns, 
                            index=input_mat.index)
    return full_mat
