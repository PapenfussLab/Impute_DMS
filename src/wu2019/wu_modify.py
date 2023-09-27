import numpy as np
from datetime import datetime
from wu2019.projects.imputation.python import imputation
from wu2019.projects.ml.python import alm_fun


def local_create_imputation_instance(proj_path, arguments, imputation_web_session_log):
    # version: 2023030901 (milton)
    """
    Differences from published `create_imputation_instance`:
    1. Reset value for floor_flag and reverse_flag to 0, to keep the same as the online default setting.
    2. Reset `valueorderby` to blosum100, to get closer to onlie result.
    3. Changed path for local usage and add parameters to identify project path.
    4. Added email information but muted code for mailing which is unnecessary.
    5. Disabled set_to_disk which will cause error.
    """
    # <<<<====
    python_path = '../src/wu2019/projects/ml/python/'
    project_path = proj_path
    humandb_path = '../src/wu2019/database/humandb/'

    #email server related parametes
    server_address = 'smtp-relay.gmail.com'
    server_port = 587
    login_user = 'noreply@varianteffect.org'
    login_password = 'WoolyMammothInThePorcellanShop'
    from_address = 'noreply@varianteffect.org'
    subject = 'No Reply'
    # <<<<====
    
    protein_id = arguments['proteinid']
    session_id = arguments.get('sessionid','')
    email_address = arguments.get('email_address','')      
        
    dms_landscape_file = session_id + '.txt'
    dms_fasta_file = session_id + '.fasta'            
    regression_cutoff = float(arguments.get('regression_cutoff','-inf'))
    data_cutoff = float(arguments.get('data_cutoff','-inf'))        
    auto_regression_cutoff = int(arguments['if_auto_cutoff'])
    data_cutoff_flag = int(arguments['if_data_cutoff'])
    normalized_flag = 1 - int(arguments['if_normalization'])   
    regularization_flag = int(arguments['if_regularization'])
    rawprocessed_flag = 1 - int(arguments['if_rawprocessing'])         
    proper_count = int(arguments.get('proper_count',8))  
    synstop_cutoff = float(arguments.get('synstop_cutoff','-inf'))
    stop_exclusion = arguments.get('stop_exclusion','0')  
    
    #alm_project class parameters
    project_params = {}

    
    project_params['project_name'] = 'imputation'
    project_params['project_path'] = project_path
    project_params['humandb_path'] = humandb_path
    project_params['log'] = imputation_web_session_log 
    project_params['verbose'] = 1

        
    #the reason the following parameters don't belong to data class is we may want to create multiple data instance in one project instance 
    project_params['data_names'] = [] 
    project_params['train_data'] = []        
    project_params['test_data'] = []
    project_params['target_data'] = []
    project_params['extra_train_data'] = []
    project_params['use_extra_train_data'] = []
    project_params['input_data_type'] = []
     
    project_params['run_data_names'] = [session_id]
    project_params['run_estimator_name'] = 'xgb_r'
    project_params['run_estimator_scorename'] = 'rmse'
    project_params['grid_search_on'] = 0
    
    project_params['modes'] = None
    project_params['train_features'] = None
    project_params['train_features_name'] = None
    project_params['start_features'] = None
    project_params['start_features_name'] = None  
    project_params['compare_features'] = None
    project_params['compare_features_name'] = None
    project_params['compare_features_name_forplot'] = None
    project_params['feature_compare_direction'] = 0 
    project_params['compare_methods'] = None 
    
    project_params['plot_columns'] = [0, 1]
    project_params['plot_vmin'] = 0.5
    project_params['plot_vmax'] = 1
    project_params['fig_w'] = 20
    project_params['fig_h'] = 5
        
    #alm_data class parameters
    data_params = {}
    data_params['path'] = project_path 
    data_params['log'] = imputation_web_session_log  
    data_params['verbose'] = 1
    
    data_params['name'] = None 
    data_params['target_data_original_df'] = None
    data_params['train_data_original_df'] = None
    data_params['test_data_original_df'] = None                
    data_params['extra_train_data_original_df'] = None 
    data_params['use_extra_train_data'] =  None
    data_params['predicted_target_df'] = None

    data_params['independent_testset'] = 0

    data_params['test_split_method'] = 0
    data_params['test_split_folds'] = 1
    data_params['test_split_ratio'] = 0
    data_params['cv_split_method'] = 2
    data_params['cv_split_folds'] = 1
    data_params['cv_split_ratio'] = 0.1
    data_params['validation_from_testset'] = False
    data_params['percent_min_feature'] = 1
    
    data_params['dependent_variable'] = 'fitness'
    data_params['filter_target'] = 0
    data_params['filter_test'] = 0
    data_params['filter_train'] = 0
    data_params['filter_validation'] = 0
    data_params['prediction_bootstrapping'] = 0
    data_params['bootstrapping_num'] = 3
    
    data_params['if_gradient'] = auto_regression_cutoff
    data_params['if_engineer'] = 0
    data_params['load_from_disk'] = 0
    data_params['save_to_disk'] = 0  # <<<<====
    data_params['cur_test_split_fold'] = 0
    data_params['cur_gradient_key'] = 'no_gradient'
    data_params['innerloop_cv_fit_once'] = 0

    data_params['onehot_features'] = []
    data_params['cv_fitonce'] = 0
        
    #alm_ml class parameters
    ml_params = {}
    ml_params['log'] = imputation_web_session_log 
    ml_params['verbose'] = 1
    
    ml_params['run_grid_search'] = 0
    ml_params['fs_start_features'] = []
    ml_params['fs_T'] = 0.001
    ml_params['fs_alpha'] = 0.8
    ml_params['fs_K'] = 100
    ml_params['fs_epsilon'] = 0.00001 
    
    #es init parameters for es_ml class
    es_params = {}
    es_params['ml_type'] = 'regression'
    es_params['single_feature_as_prediction'] = 1
    es_params['estimator'] = None
    es_params['name'] = None
    es_params['gs_range'] = None
    es_params['score_name'] = None
    es_params['score_direction'] = None
    es_params['feature_importance_name'] = None
    es_params['round_digits'] = 4 
    es_params['if_feature_engineer'] = 1
    es_params['feature_engineer'] = None
     
    #data preprocess and update data_params                  
    imputation_params = {} 
    imputation_params['log'] = imputation_web_session_log 
    imputation_params['verbose'] = 1
    imputation_params['project_path'] = project_path
    imputation_params['humandb_path'] = humandb_path
    
    imputation_params['project_params'] = project_params
    imputation_params['data_params'] = data_params
    imputation_params['ml_params'] = ml_params
    imputation_params['es_params'] = es_params
       
    #imputation class: parameters for data preprocessing
    imputation_params['run_data_preprocess'] = 1 
    imputation_params['dms_landscape_files'] = [dms_landscape_file]
    imputation_params['dms_fasta_files'] = [dms_fasta_file]
    imputation_params['dms_protein_ids'] = [protein_id]
    imputation_params['data_names'] = [session_id]
    imputation_params['remediation'] = [0]
    
    if normalized_flag == 1:
        imputation_params['synstop_cutoffs'] = [float("-inf")]
        imputation_params['stop_exclusion'] = ["0"]
    else:
        imputation_params['synstop_cutoffs'] = [synstop_cutoff]
        imputation_params['stop_exclusion'] = [stop_exclusion]
        
    if data_cutoff_flag == 1:
        imputation_params['quality_cutoffs'] = [data_cutoff]
    else:
        imputation_params['quality_cutoffs'] = [float("-inf")]
    
    if auto_regression_cutoff == 1:
        imputation_params['regression_quality_cutoffs'] = [float("-inf")]
    else:
        imputation_params['regression_quality_cutoffs'] = [regression_cutoff]
                    
    imputation_params['proper_num_replicates'] = [proper_count]
    imputation_params['raw_processed'] = [rawprocessed_flag]
    imputation_params['normalized_flags'] = [normalized_flag]
    imputation_params['regularization_flags'] = [regularization_flag]
    imputation_params['reverse_flags'] = [0]  # <<<<====
    imputation_params['floor_flags'] = [0]  # <<<<====
    imputation_params['combine_flags'] = [0]

    imputation_params['pre_process'] = 1
    imputation_params['combine_dms'] = 0
        
    #imputation class: parameters for feature engineering
    imputation_params['k_range'] = range(3, 4)
    imputation_params['use_funsums'] = ['funsum_fitness_mean']
    imputation_params['use_funsums_name'] = ['fs']  
    imputation_params['value_orderby'] = ['blosum100']    # <<<<====
    imputation_params['value_orderby_name'] = ['bs']    # <<<<====
    imputation_params['centrality_names'] = ['mean', 'se', 'count']
    imputation_params['dependent_variable'] = 'fitness'
    imputation_params['add_funsum_onfly'] = 0 

    #imputation class: Jochen's R script related parameters
    imputation_params['if_runR'] = 0
    imputation_params['R_command'] = '/Library/Frameworks/R.framework/Versions/3.3/Resources/Rscript'
    imputation_params['R_wd'] = '/Users/joewu/Google_Drive/Business/AlphaMe/Source_Code/R/R_GI/Jochen/dmspipeline'
    imputation_params['R_script_path'] = '/Users/joewu/Google_Drive/Business/AlphaMe/Source_Code/R/R_GI/Jochen/dmspipeline/bin/simpleImpute.R'
    imputation_params['R_bend'] = '0'   

    #imputation class: email related parameters
    imputation_params['email_server_address'] = server_address
    imputation_params['email_server_port'] = server_port
    imputation_params['email_login_user'] = login_user
    imputation_params['email_login_password'] = login_password
    imputation_params['email_from_address'] = from_address
    #imputation_params['email_msg_content'] = create_email_msg(session_id)
    #imputation_params['email_error_content'] = create_email_error_msg(email_address,session_id)
    #imputation_params['email_notification_content'] = create_email_notification_msg(session_id)
        
    im_proj = imputation.imputation(imputation_params)    
    return (im_proj)


def foo_alm_project_run(uniprot_id, session_id, proj_path, imputation_web_session_log):
    # version: 2023030901
    """
    `if_auto_cutoff` to false is the default with default `regression_cutoff`; these are related to _auto training variant quality cutoff_ (inspected); and also means that quality score lower than 0 will be discarded
    `if_rawprocessing` to false may mean the input is not raw data
    `if_normalization` to false may mean it is normalized and do not need data rescaling (non-syn related) (inspected)
    `if_data_cutoff` to false may mean do not filter low quality variants which is default (inspected)
    `if_regularization` to false is the default (inspected)
    """
    arguments = {'proteinid': uniprot_id, 'sessionid': session_id, 'if_auto_cutoff': 0, 'regression_cutoff':0,
                 'if_rawprocessing': 0, 'if_normalization': 0, 'if_data_cutoff': 0,  'data_cutoff': 0, 
                 'if_regularization': 0, 'proper_count': 8, 'stop_exclusion': 0, 'synstop_cutoff': '-inf'}
    imp_r = local_create_imputation_instance(proj_path, arguments, imputation_web_session_log)

    data_name = session_id
    alm_fun.show_msg (imp_r.log,imp_r.verbose,'**Class: [imputation] Fun: [imputation_run] .... starts @' + str(datetime.now()))
    features = []
    for k in imp_r.k_range:
        for orderby_name in imp_r.value_orderby_name:
            cur_col_feature_names = [x + '_' + str(k) + '_' + orderby_name for x in imp_r.centrality_names]
            features += cur_col_feature_names  
    features = features + ['polyphen_score', 'provean_score', 'blosum100']
    imp_r.project.train_features = features
    imp_r.project.run_data_names = [data_name]
    imp_r.project.modes = ['target_prediction']
    fitness_predicted = imp_r.project.run(refresh_data = 1)['target_prediction'][data_name]  
    dms_gene_df = imp_r.project.data[data_name].target_data_original_df.copy()
    dms_gene_df['fitness_imputed'] = np.nan
    dms_gene_df.loc[imp_r.project.data[data_name].target_data_df.index, 'fitness_imputed'] = fitness_predicted
    return dms_gene_df


def foo_wu2019_wrapping_func_23031601(tr_data_ori, session_id):
    proj_path = '../result/wu2019/'
    uniprot_id = tr_data_ori['uniprot_id'].iloc[0]
    
    tr_data = tr_data_ori.copy()[['aa1', 'aa2', 'u_pos', 'score']]
    tr_data = tr_data.rename(columns={'aa1': 'aa_ref', 'aa2': 'aa_alt', 'u_pos': 'aa_pos', 
                                      'score': 'fitness_input'})
    tr_data['quality_score'] = 1 # Must > 0
    tr_data['num_replicates'] = 1
    tr_data['fitness_input_sd'] = 0
    tr_data.to_csv(f"{proj_path}upload/{session_id}.txt", sep='\t', index=None) 
    with open(f"../data/uniprot_fasta/{uniprot_id}.fasta",  'r') as file:
        fasta = file.read()
        with open(f"{proj_path}upload/{session_id}.fasta", 'w+') as new_f:
            new_f.write(fasta)
            
    imputation_web_session_log = open(f'{proj_path}log.txt', 'a+')
    imp_result = foo_alm_project_run(uniprot_id, session_id, proj_path, imputation_web_session_log)
    imputation_web_session_log.close()
        
    imp_result = imp_result[['aa_pos', 'aa_alt', 'fitness_imputed']]
    imp_result.columns = ['u_pos', 'aa2', 'pred_score']
    return imp_result
