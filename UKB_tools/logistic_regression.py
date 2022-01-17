import pandas as pd
import numpy as np

def do_impute_data(input_df: pd.core.frame.DataFrame, meta_data_dict_list: list, skip_columns: list = None, type: int = 0, round_impute: bool = False) -> pd.core.frame.DataFrame:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.impute import SimpleImputer
    
    new_input_df = input_df.copy()
    skipped_columns_df = pd.DataFrame()
    
    if skip_columns != None:
        for skip_column in skip_columns:
            new_input_df = new_input_df.drop([skip_column], axis = 1)
            skipped_columns_df[skip_column] = input_df[skip_column]
    
    for column in new_input_df:
        for meta_data_dict in meta_data_dict_list:
            if column.startswith(meta_data_dict['column']):
                if meta_data_dict['value_type'].startswith("Categorical"):
                    new_input_df[column] = new_input_df[column].replace(np.nan, 0)
                        
    if type == 0:
        imp = SimpleImputer(missing_values=np.nan, strategy = 'mean')
        imputated_input_np = imp.fit_transform(new_input_df)
    elif type == 1:
        imp = IterativeImputer(max_iter = 10)
        imputated_input_np = imp.fit_transform(new_input_df)

    if skip_columns != None:
        skipped_columns_np = skipped_columns_df.to_numpy()
        imputated_input_np = np.concatenate((skipped_columns_np, imputated_input_np), axis = 1)
    
    imputated_input_df = pd.DataFrame(imputated_input_np, columns = input_df.columns.values.tolist())

    if round_impute == True:
        for column in imputated_input_df.copy():
            for meta_data_dict in meta_data_dict_list:
               if column.startswith(meta_data_dict['column']):
                    if meta_data_dict['value_type'].startswith("Categorical"):
                        imputated_input_df[column] = imputated_input_df[column].round(0)

    return imputated_input_df

def do_encode_labels(y: pd.core.frame.DataFrame, encoding_scheme: dict = None):
    from sklearn.preprocessing import LabelEncoder
    
    if encoding_scheme != None:
        for key in encoding_scheme:
            y = np.where((y == encoding_scheme[key]), key, y)
        
        y_transform = pd.Series(y).astype(int)
        
    else:
        le = LabelEncoder()
        y_transform = le.fit_transform(y)
    
        encoding_scheme = dict(zip(y_transform, y))

    return y_transform, encoding_scheme

def do_scale_data(df: pd.core.frame.DataFrame, skip_columns: list = None, target_column: str = None) -> pd.core.frame.DataFrame:
    from sklearn.preprocessing import StandardScaler
    
    copy_df = df.copy()
    
    if skip_columns != None:
        for col in skip_columns:
            copy_df = copy_df.drop(col, 1)
            
    if target_column != None:
        copy_df = copy_df.drop(target_column, 1)
    
    scaler = StandardScaler()
    scaler.fit(copy_df)
    copy_df = scaler.transform(copy_df)
    
    copy_df = pd.DataFrame(copy_df, columns = df.columns)
    
    return copy_df

def do_create_polynominal_features(input_df: pd.core.frame.DataFrame, target_columns: list, use_only_interaction: bool = False) -> pd.core.frame.DataFrame:
    from sklearn.preprocessing import PolynomialFeatures
    
    selection_df = input_df[target_columns]
    input_df = input_df.drop(columns = target_columns)
    
    number_of_targets = len(target_columns)   
    new_target_columns = target_columns.copy()
    for i in range(number_of_targets):
        for j in range(number_of_targets):
            if i != j and j > i:
                new_target_columns.append(new_target_columns[i] + "_" + new_target_columns[j])
    
    poly = PolynomialFeatures(interaction_only = True, include_bias = False)
    poly_features = poly.fit_transform(selection_df)
    poly_features_df = pd.DataFrame(poly_features, columns = new_target_columns)
    output_df = pd.concat([input_df, poly_features_df], axis = 1, join = 'outer')
    
    if use_only_interaction == True:
        output_df = output_df.drop(columns = target_columns)
    
    return output_df
    
def create_training_test_group(input_df: pd.core.frame.DataFrame, skip_columns: list, target_column: str, encode_labels: dict = None) -> [pd.core.frame.DataFrame, dict]:
    from sklearn.model_selection import train_test_split
    
    new_input_df = input_df.copy()
    
    for skip_column in skip_columns:
        new_input_df = new_input_df.drop([skip_column], axis = 1)
    
    y = new_input_df[target_column]
    x = new_input_df.drop([target_column], axis = 1)
    
    y, encoding_scheme = do_encode_labels(y, encode_labels)
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, stratify = y, test_size = 0.2, random_state = None)
    
    return X_train, X_test, y_train, y_test, encoding_scheme

def bootstrap_selection(dataframe_list: list) -> pd.core.frame.DataFrame:
    import random
    
    output_df_list = []
    
    for dataframe in dataframe_list:
        output_df_list.append(pd.DataFrame(columns = dataframe.columns.tolist()))
    
    for i in range(dataframe_list[0].shape[0]):
        randint = random.randint(0, dataframe_list[0].shape[0] - 1)
        
        j = 0
        for output_dataframe in output_df_list.copy():
            output_df_list[j] = output_dataframe.append(dataframe_list[j].iloc[randint], ignore_index = True)
            j += 1
        
    return output_df_list

def under_over_sampler(X: pd.core.frame.DataFrame, y: pd.core.frame.DataFrame, sample_type: int, ratio: float) -> pd.core.frame.DataFrame:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    
    if sample_type == 0:
        rus = RandomUnderSampler(sampling_strategy = ratio)
        X_res, y_res = rus.fit_resample(X, y)
    elif sample_type == 1:
        sm = SMOTE(sampling_strategy = ratio)
        X_res, y_res = sm.fit_resample(X, y)

    return X_res, y_res

def get_model(model_type: str) -> object:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.experimental import enable_hist_gradient_boosting
    from sklearn.ensemble import HistGradientBoostingClassifier
    
    if model_type == "logistic_regression":
        model = LogisticRegression(penalty = 'l1', solver = 'liblinear', n_jobs = -1)
    elif model_type == "svc":
        model = SVC(kernel = 'linear', max_iter = 1000, cache_size = 2000, probability = True)
    elif model_type == "clf":
        model = RandomForestClassifier(random_state = None,  n_jobs = -1)
    elif model_type == "mlp":
        model = MLPClassifier(max_iter = 10000)
    elif model_type == "gbc":
        model = HistGradientBoostingClassifier()
    elif model_type == "abc":
        model = AdaBoostClassifier()
        
    return model

def adjust_thresholds(y_test: np.array, y_test_proba: np.array, scorer: str) -> dict:
    from sklearn import metrics
    
    if scorer == "aupcr":
        scorer = "f1"
    
    adj_thresholds_dict = {}
    
    for t in [t * 0.01 for t in range(1, 101, 1)]:
        y_test_proba_adj = [1 if y >= t else 0 for y in y_test_proba]
                
        if scorer == "f1":
            score_adj = metrics.f1_score(y_test, y_test_proba_adj)
        elif scorer == "fbeta0.5":
            score_adj = metrics.fbeta_score(y_test, y_test_proba_adj, beta = 0.5)
        
        adj_thresholds_dict[t] = {'y_test_proba_adj': y_test_proba_adj, 'score_adj': score_adj}
        
    best_score_adj = 0
    for key in adj_thresholds_dict:
        if adj_thresholds_dict[key]['score_adj'] > best_score_adj:
            best_score_adj = adj_thresholds_dict[key]['score_adj']
            best_y_test_proba_adj =  adj_thresholds_dict[key]['y_test_proba_adj']
            best_threshold = key
        
    return adj_thresholds_dict, best_score_adj, best_y_test_proba_adj, best_threshold
        

def do_train_test_model(X_train: pd.core.frame.DataFrame, X_test: pd.core.frame.DataFrame, y_train: pd.core.frame.DataFrame, y_test: pd.core.frame.DataFrame, model_type: str, model: object, parameter_search: dict = None, pos_label_encode: int = 1, scorer_type: str = "f1") -> dict:
    import matplotlib.pyplot as plt
    from collections import Counter
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn import metrics
            
    if parameter_search != None:
        model = parameter_search['best_estimator']
    
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_train_pred_probs = model.predict_proba(X_train)[:,1]
    
    y_test_pred = model.predict(X_test)
    y_test_pred_probs = model.predict_proba(X_test)[:,1]
    
    adj_thresholds_dict, best_score_adj, best_y_test_proba_adj, best_threshold = adjust_thresholds(y_test, y_test_pred_probs, scorer_type)
    
    train_fpr, train_tpr, train_thresholds = metrics.roc_curve(y_train, y_train_pred_probs, pos_label = pos_label_encode)
    test_fpr, test_tpr, test_thresholds = metrics.roc_curve(y_test, y_test_pred_probs, pos_label = pos_label_encode)
        
    cf_test_tn, cf_test_fp, cf_test_fn, cf_test_tp = metrics.confusion_matrix(y_test, y_test_pred, labels = [0, 1]).ravel()
    cf_test = ["tn: " + str(cf_test_tn), "fp: " + str(cf_test_fp), "fn: " + str(cf_test_fn), "tp: " + str(cf_test_tp)]
    
    cf_test_tn, cf_test_fp, cf_test_fn, cf_test_tp = metrics.confusion_matrix(y_test, best_y_test_proba_adj, labels = [0, 1]).ravel()
    cf_test_adj = ["tn: " + str(cf_test_tn), "fp: " + str(cf_test_fp), "fn: " + str(cf_test_fn), "tp: " + str(cf_test_tp)]
    
    recall_precision, recall, _ = metrics.precision_recall_curve(y_test, y_test_pred)
    recall_disp = metrics.PrecisionRecallDisplay(precision = recall_precision, recall = recall)
    recall_plot = recall_disp.plot()
    recall_plot = plt.gcf()
    
    precision_values, recall_values, pr_re_thresholds = metrics.precision_recall_curve(y_test, y_test_pred)
        
    model_dict = {
        'model': model,
        'classes': model.classes_,
        'distribution': {'train': Counter(y_train), 'test': Counter(y_test)},
        'train': {'true_values': y_train, 'matrix': metrics.confusion_matrix(y_train, y_train_pred), 'accuracy': metrics.accuracy_score(y_train, y_train_pred), 'f1': metrics.f1_score(y_train, y_train_pred), 'roc_auc': metrics.roc_auc_score(y_train, y_train_pred_probs), 'prediction_probabilities': y_train_pred_probs},
        'test': {'true_values': y_test, 'matrix': cf_test, 'accuracy': metrics.accuracy_score(y_test, y_test_pred), 'f1': metrics.f1_score(y_test, y_test_pred), 'recall': metrics.recall_score(y_test, y_test_pred), 'aupcr': metrics.average_precision_score(y_test, y_test_pred_probs), 'precision': metrics.precision_score(y_test, y_test_pred), 'roc_auc': metrics.roc_auc_score(y_test, y_test_pred_probs), 'fbeta0.5': metrics.fbeta_score(y_test, y_test_pred, beta = 0.5), 'prediction_probabilities': y_test_pred_probs},
        'predicted': y_test_pred,
        'classification_report': metrics.classification_report(y_test, y_test_pred, digits = 6),
        'roc_curve_train': {'train_fpr': train_fpr, 'train_tpr': train_tpr, 'train_thresholds': train_thresholds},
        'roc_curve_test': {'test_fpr': test_fpr, 'test_tpr': test_tpr, 'test_thresholds': test_thresholds},
        'recall_plot': recall_plot,
        'precission_recall_plot_values': {'precision_values': precision_values, 'recall_values': recall_values, 'pr_re_thresholds': pr_re_thresholds},
        'adjusted_thresholds': {'adj_thresholds_dict': adj_thresholds_dict, 'best_score_adj': best_score_adj, 'best_y_test_proba_adj': best_y_test_proba_adj, 'best_threshold': best_threshold, 'cf_matrix_adj': cf_test_adj}
        }
    
    if model_type == "logistic_regression" or model_type == "svc":
        model_dict['intercept'] = model.intercept_
        model_dict['coef_df'] = pd.DataFrame({'Column': X_train.columns, 'coef': model.coef_[0].tolist()})
    
    if model_type == "clf" or model_type == "abc":
        model_dict['coef_df'] = pd.DataFrame({'Column': X_train.columns, 'coef': model.feature_importances_.tolist()})

    if model_type == "mlp" or model_type == "gbc":
        model_score = model.score(X_test, y_test)
        diff_score = []
        for iter_column in X_test.columns:
            X_test_shuffle = X_test.copy()
            iter_column_values = X_test_shuffle[iter_column].copy()
            iter_column_values_arr = np.array(iter_column_values.values.tolist())
            np.random.shuffle(iter_column_values_arr)
            iter_column_values = pd.Series(iter_column_values_arr)
            X_test_shuffle[iter_column] = iter_column_values
            diff_score.append(model_score - model.score(X_test_shuffle, y_test))
        
        model_dict['coef_df'] = pd.DataFrame({'Column': X_train.columns, 'coef': diff_score})
        
    if parameter_search != None:
        model_dict['best_params'] = parameter_search['best_params']
        
    plt.close('all')

    return model_dict

def do_parameter_search(X_train: pd.core.frame.DataFrame, y_train: pd.core.frame.DataFrame, model_type: str, model: object, scorer_type: str = "f1") -> dict:
    from sklearn.experimental import enable_halving_search_cv
    from sklearn.model_selection import HalvingGridSearchCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import metrics
    from sklearn.svm import SVC
    import itertools
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import RandomizedSearchCV
    
    if model_type == "logistic_regression":
        param_grid = {'solver': ['saga'], 'penalty': ['elasticnet'], 'l1_ratio': [x*0.01 for x in range(1, 101, 1)], 'class_weight': ['balanced']}
    elif model_type == "svc":
        param_grid = {'C': [x*0.01 for x in range(1, 101, 1)], 'kernel': ['linear'], 'class_weight': ['balanced']}
    elif model_type == "clf":
        param_grid = {'n_estimators': [10, 50, 100, 200, 500, 1000], 'max_depth': [2, 5, 10, 50, 100, 200, 500, 1000, None], 'min_samples_split': [2, 5, 10, 50, 100], 'min_samples_leaf': [2, 5, 10, 50, 100], 'class_weight': ['balanced', 'balanced_subsample']}
    elif model_type == "mlp":
        param_grid = {'hidden_layer_sizes': [x for x in itertools.product((10, 20, 30, 40, 50, 100), repeat = 3)], 'alpha': [0.0001, 0.001, 0.1]} # [0.0001, 0.001, 0.01]
    elif model_type == "gbc":
        param_grid = {'max_depth': [2, 5, 10, 50, 100, 200, 500, 1000, None], 'min_samples_leaf': [2, 5, 10, 50, 100], 'l2_regularization': [0, 0.1, 1, 10, 100]}
    elif model_type == "abc":
        param_grid = {'n_estimators': [5, 10, 50, 100, 200, 500, 700, 1000, 1500, 2000]}

    if scorer_type == "f1":
        scorer = metrics.make_scorer(metrics.f1_score)
    elif scorer_type == "aupcr":
        scorer = metrics.make_scorer(metrics.average_precision_score)
    elif scorer_type == "fbeta0.5":
        scorer = metrics.make_scorer(metrics.fbeta_score, beta = 0.5)

    sh = GridSearchCV(model, param_grid, n_jobs = -1, scoring = scorer).fit(X_train, y_train)
    
    best_estimator_dict = {
        'best_estimator': sh.best_estimator_,
        'best_params': sh.best_params_,
        'cv_results': sh.cv_results_
    }
        
    return best_estimator_dict

def calculate_shap_values(model: object, X_test: pd.core.frame.DataFrame, model_type: str, create_shap_fig: list = None, replace_column_names: list = None) -> dict:
    import matplotlib.pyplot as plt
    import shap
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    
    old_columns = X_test.columns
    
    if replace_column_names != None:
        for col in X_test.copy():
            new_colname = ""
            
            for meta_data_dict in replace_column_names:
                if col.startswith(meta_data_dict['column']):
                    new_colname = meta_data_dict['description']
                    
                    if len(col.split("_")) > 1:
                        new_colname = new_colname + "_" + col.split("_")[1]
                        
                    X_test.rename(columns = {col: new_colname}, inplace = True)
    
    shap_dict = {}
    
    if model_type == "clf":
        explainer = shap.TreeExplainer(model)
    else:
        X_test_summary = shap.kmeans(X_test, 10)
        explainer = shap.KernelExplainer(model.predict, X_test_summary)
        
    shap_values = np.array(explainer.shap_values(X_test))
    
    if create_shap_fig != None:
        if model_type == "clf":
            for key in create_shap_fig[1]:
                if create_shap_fig[1][key] == create_shap_fig[0]:
                    shap_values = shap_values[key]
                    expected_value = explainer.expected_value[key]
        else:
            expected_value = explainer.expected_value
            
        shap_dict['shap_values'] = shap_values
        shap_feature_importance = shap_values.mean(0)
        shap_feature_importance_df = pd.DataFrame(list(zip(old_columns, shap_feature_importance)), columns = ['col_name', 'feature_importance'])
        shap_dict['shap_feature_importance_df'] = shap_feature_importance_df
    
        summary_plot = plt.figure()
        shap.summary_plot(shap_values, X_test, show = False, max_display = 15)
        shap_dict['summary_plot'] = summary_plot
                
        class ShapObject:
            def __init__(self, base_values, data, values, feature_names):
                self.base_values = base_values # Single value
                self.data = data # Raw feature values for 1 row of data
                self.values = values # SHAP values for the same row of data
                self.feature_names = feature_names # Column names
                
        shap_object = ShapObject(base_values = expected_value,
                                 values = shap_values[0],
                                 feature_names = X_test.columns,
                                 data = X_test.iloc[0,:])
                
        waterfall_plot = plt.figure()
        shap.plots._waterfall.waterfall_legacy(expected_value, shap_values[0], feature_names = X_test.columns, show = False)
        shap_dict['waterfall_plot'] = waterfall_plot
        
        force_plot = shap.force_plot(expected_value, shap_values[0], X_test.iloc[[0]], matplotlib = True, show = False)
        shap_dict['force_plot'] = force_plot
        
        decision_plot = plt.figure()
        shap.decision_plot(expected_value, shap_values[0], X_test.iloc[[0]], show = False)
        shap_dict['decision_plot'] = decision_plot
                
                
        plt.close('all')

    return shap_dict

def multiple_do_train_test_model(input_df: pd.core.frame.DataFrame, sample_type: str, n_times: int, skip_columns: list, target_column: str, encode_labels: dict = None, model_type: str = "logistic_regression", meta_data_dict_list: list = None, impute_data: list = [False], sample_data: bool = False, parameter_search: bool = False, create_polynominal_features: list = False, scorer_type: str = "f1") -> [list, pd.core.frame.DataFrame]:
    import statistics
    import json
    from UKB_tools.support import printProgressBar
    
    predicted_sab_list = []
    sab_list = []
    models_list = []
    values_dict = {'train_auc': [], 'train_auc_mean': [], 'train_auc_std': [], 'test_auc': [], 'test_auc_mean': [], 'test_auc_std': [], 'test_f1': [], 'test_f1_mean': [], 'test_f1_std': [], 'coef': {}, 'coef_mean': {}, 'coef_std': {}, 'shap_values_feature_importance': {}, 'shap_values_feature_importance_mean': {}, 'shap_values_feature_importance_std': {}}
     
    for i in range(n_times):
        printProgressBar(i + 1, n_times, prefix = 'Progress:', suffix = 'Complete', length = 50)
            
        model = get_model(model_type)
        
        X_train, X_test, y_train, y_test, encoding_scheme = create_training_test_group(input_df, skip_columns, target_column, encode_labels)
        
        if impute_data[0] == True:
            X_train = do_impute_data(X_train, meta_data_dict_list, None, 0, impute_data[1])   
            X_test = do_impute_data(X_test, meta_data_dict_list, None, 0, impute_data[1])  
        
        if create_polynominal_features != False:
            X_train = do_create_polynominal_features(X_train, create_polynominal_features, False)
            X_test = do_create_polynominal_features(X_test, create_polynominal_features, False)
            
        if sample_data == True:
            X_train, y_train = under_over_sampler(X_train, y_train, 0, 0.5)
                    
        if sample_type == "bootstrap":
            y_train = y_train.to_frame()
            y_test = y_test.to_frame()
            
            bootstrap_list = bootstrap_selection([X_train, y_train])
            X_train = bootstrap_list[0]
            y_train = bootstrap_list[1].squeeze().astype(int)
            
            bootstrap_list = bootstrap_selection([X_test, y_test])
            X_test = bootstrap_list[0]
            y_test = bootstrap_list[1].squeeze().astype(int)
            
        else:
            pass
        
        X_train = do_scale_data(X_train, None, None)
        X_test = do_scale_data(X_test, None, None)
                
        if parameter_search == True:
            parameter_dict = do_parameter_search(X_train, y_train, model_type, model, scorer_type)
        else:
            parameter_dict = None
        
        model_dict = do_train_test_model(X_train, X_test, y_train, y_test, model_type, model, parameter_dict, 1, scorer_type)
        
        model_dict['encoding_scheme'] = encoding_scheme
        
        shap_dict = calculate_shap_values(model_dict['model'], X_test, model_type, ["sab", model_dict['encoding_scheme']], meta_data_dict_list)
        model_dict['shap'] = shap_dict
        
        models_list.append(model_dict)
        
        values_dict['train_auc'].append(model_dict['train']['roc_auc'])
        values_dict['test_auc'].append(model_dict['test']['roc_auc'])
        values_dict['test_f1'].append(model_dict['test']['f1'])
        
        values_dict['train_auc_mean'].append(statistics.mean(values_dict['train_auc']))
        values_dict['test_auc_mean'].append(statistics.mean(values_dict['test_auc']))
        values_dict['test_f1_mean'].append(statistics.mean(values_dict['test_f1']))
        
        values_dict['train_auc_std'].append(np.std(values_dict['train_auc']))
        values_dict['test_auc_std'].append(np.std(values_dict['test_auc']))
        values_dict['test_f1_std'].append(np.std(values_dict['test_f1']))
        
        for index, row in model_dict['coef_df'].iterrows():
            if i > 0:
                values_dict['coef'][row['Column']].append(row['coef'])
                values_dict['coef_mean'][row['Column']].append(statistics.mean(values_dict['coef'][row['Column']]))
                values_dict['coef_std'][row['Column']].append(np.std(values_dict['coef'][row['Column']]))
            else:
                values_dict['coef'][row['Column']] = [row['coef']]
                values_dict['coef_mean'][row['Column']] = [row['coef']]
                values_dict['coef_std'][row['Column']] = [np.std(values_dict['coef'][row['Column']])]
                
        for index, row in shap_dict['shap_feature_importance_df'].iterrows():
            if i > 0:
                values_dict['shap_values_feature_importance'][row['col_name']].append(row['feature_importance'])
                values_dict['shap_values_feature_importance_mean'][row['col_name']].append(statistics.mean(values_dict['shap_values_feature_importance'][row['col_name']]))
                values_dict['shap_values_feature_importance_std'][row['col_name']].append(np.std(values_dict['shap_values_feature_importance'][row['col_name']]))
            else:
                values_dict['shap_values_feature_importance'][row['col_name']] = [row['feature_importance']]
                values_dict['shap_values_feature_importance_mean'][row['col_name']] = [row['feature_importance']]
                values_dict['shap_values_feature_importance_std'][row['col_name']] = [np.std(values_dict['shap_values_feature_importance'][row['col_name']])]
                
        qi = 0
        qj = 0
        for prediction in model_dict['predicted']:
            if prediction == 1:
                qi += 1
        for sab_case in y_test:
            if sab_case == 1:
                qj += 1
        predicted_sab_list.append(qi)
        sab_list.append(qj)
    
    out_file = open('preprocessing2/json_test.json','w+')
    json.dump(values_dict, out_file)
    
    coef_df = pd.DataFrame.from_dict(values_dict['coef'])
    coef_df_mean = pd.DataFrame.from_dict(values_dict['coef_mean'])
    try:
        coef_df_std = pd.DataFrame.from_dict(values_dict['coef_std'])
    except:
        print(values_dict['coef_std'])
    auc_df = pd.DataFrame(list(zip(values_dict['train_auc'], values_dict['train_auc_mean'], values_dict['train_auc_std'], values_dict['test_auc'], values_dict['test_auc_mean'], values_dict['test_auc_std'], values_dict['test_f1'], values_dict['test_f1_mean'], values_dict['test_f1_std'])), columns = ['train_auc', 'train_auc_mean', 'train_auc_std', 'test_auc', 'test_auc_mean', 'test_auc_std', 'test_f1', 'test_f1_mean', 'test_f1_std'])
    
    shap_values_feature_importance_df = pd.DataFrame.from_dict(values_dict['shap_values_feature_importance'])
    shap_values_feature_importance_df_mean = pd.DataFrame.from_dict(values_dict['shap_values_feature_importance_mean'])
    shap_values_feature_importance_df_std = pd.DataFrame.from_dict(values_dict['shap_values_feature_importance_std'])
    
    
    output_list = [coef_df, coef_df_mean, coef_df_std, auc_df, sab_list, shap_values_feature_importance_df, shap_values_feature_importance_df_mean, shap_values_feature_importance_df_std]
    
    return models_list, output_list
                
        
            
            