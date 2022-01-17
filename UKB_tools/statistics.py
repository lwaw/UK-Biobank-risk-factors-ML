import pandas as pd
import numpy as np
import json
from scipy import stats
import itertools

id = 0

def get_new_id() -> int:
    global id
    
    new_id = id
    id += 1
    
    return new_id

#check if column is ordinal by checking data coding id's
def isin_ordinal_data_codings(meta_data: list, column: str) -> set:
    ordinal_data_codings = {'100348', '100356', '100004', '100262', '100398', '100429', '100428', '100360', '100417', '100301', '100570', '100511', '100006', '100539', '100007', '100017', '100334', '100343', '100011', '100479', '100335', '101310', '100327', '5004', '22', '100014', '100318', '100478', '100402', '100377', '100317', '100337', '100346', '23', '100341', '100394', '100293', '100536', '100635', '15', '100508', '100639', '100636', '3637', '100435', '100401', '100637', '100432', '100012', '5006', '100003', '100484', '100431', '100294', '100336', '100501', '100499', '504', '534', '532', '537', '494', '506'}
    column_data_coding = None
    
    for meta_data_dict in meta_data:
        if meta_data_dict['column'] == column:
            try:
                column_data_coding = meta_data_dict['data_coding']['data_coding_id']
                
                if column_data_coding in ordinal_data_codings:
                    return True
            except:
                pass
            
    return False

#add statistic to dictionary
def add_to_statistics_dict(statistics_dict_list: list, column: str, statistic_name: str, statistic_values: dict, add_when_dict_exists: int = 0, group: list = None) -> list:
    new_statistics_dict_list = []
    added_statistic = False
    
    for statistics_dict in statistics_dict_list:
        if statistics_dict['column'] == column:
            statistics_dict[statistic_name] = statistic_values
            added_statistic = True
        new_statistics_dict_list.append(statistics_dict)
        
    if added_statistic == False and add_when_dict_exists == 0:
        new_statistics_dict_list.append({'id': get_new_id(),'column': column, statistic_name: statistic_values, 'group': group})
        
    return new_statistics_dict_list

#remove nan from dataframe
def remove_nan(subset_data_df: pd.core.frame.DataFrame, column: str) -> pd.core.frame.DataFrame:
    subset_data_df = subset_data_df.replace(r'^\s*$', np.nan, regex=True)
    subset_data_df.dropna(subset=[column], inplace=True)
    
    return subset_data_df

#get all unique groups from group column in dataframe
def get_group(subset_data_df: pd.core.frame.DataFrame, column: str) -> list:
    if 'group' in subset_data_df:
        group = subset_data_df.group.unique().tolist()
    else:
        group = None
        
    return group

#adds the number of cases per group to dictionaries
def get_number_of_cases(subset_data_df: pd.core.frame.DataFrame, statistics_dict_list: list = None, max_number_of_columns: int = None) -> list:
    if statistics_dict_list == None:
        statistics_dict_list = []
        
    for column in subset_data_df:
        if column != "eid" and column != "group": 
            if len(subset_data_df.columns) <= max_number_of_columns or max_number_of_columns == None:
                removed_nan = remove_nan(subset_data_df, column)
                group = get_group(removed_nan, column)
                
                statistic_values = {}
                if 'group' in removed_nan:
                    number_of_cases = removed_nan[[column, 'group']].groupby('group').count()
                    
                    for index, row in number_of_cases.iterrows():
                        statistic_values[index] = row[column]
                else:
                    number_of_cases = removed_nan[column].count()
                    statistic_values['count'] = number_of_cases
            
            else:
                removed_nan = remove_nan(subset_data_df, column)
                group = get_group(removed_nan, column)
                statistic_values = {}
                statistic_values['count'] = 0
            
            statistics_dict_list = add_to_statistics_dict(statistics_dict_list, column, "number_of_cases", statistic_values, 0, group)
    
    return statistics_dict_list

#adds means to dictonaries
def get_mean(subset_data_df: pd.core.frame.DataFrame, statistics_dict_list: list = None, remove_values: list = None) -> list:
    from UKB_tools.preprocessing import filter_coding_type
    
    if statistics_dict_list == None:
        statistics_dict_list = []
        
    for column in subset_data_df:
        if column != "eid" and column != "group":           
            removed_nan = remove_nan(subset_data_df, column)
            removed_nan[column] = pd.to_numeric(removed_nan[column])
            if remove_values != None:
                removed_nan = filter_coding_type(removed_nan, remove_values, column, False)
            group = get_group(removed_nan, column)
            
            statistic_values = {}
            if 'group' in removed_nan:
                means = removed_nan[[column, 'group']].groupby('group').mean()
                
                statistic_values_diff_list = []
                for index, row in means.iterrows():
                    statistic_values[index] = row[column]
                    statistic_values_diff_list.append(row[column])
                    
                statistic_values_diff_array = np.array(statistic_values_diff_list)
                a, b = np.meshgrid(statistic_values_diff_array, statistic_values_diff_array)
                statistic_values_diff_array = (b - a)
                statistic_values_diff_dict = dict(enumerate(dict(enumerate(row)) for row in statistic_values_diff_array))
                
                statistics_dict_list = add_to_statistics_dict(statistics_dict_list, column, "mean_diff", statistic_values_diff_dict, 0, group)
            else:
                means = removed_nan[[column]].mean()
                
                statistic_values['mean'] = means
            
            statistics_dict_list = add_to_statistics_dict(statistics_dict_list, column, "mean", statistic_values, 0, group)
    
    return statistics_dict_list

#test means on significance
def get_mean_difference_significance(subset_data_df: pd.core.frame.DataFrame, statistics_dict_list: list = None, report_empty_tests: float = None, remove_values: list = None) -> list:
    from UKB_tools.preprocessing import filter_coding_type
    
    if statistics_dict_list == None:
        statistics_dict_list = []
    
    for column in subset_data_df:
        if column != "eid" and column != "group":           
            removed_nan = remove_nan(subset_data_df, column)
            removed_nan[column] = pd.to_numeric(removed_nan[column])
            if remove_values != None:
                removed_nan = filter_coding_type(removed_nan, remove_values, column, False)
            group = get_group(removed_nan, column)
            
            statistic_values = {}
            if 'group' in removed_nan:
                p_normal_list = []
                elements_list = []
                
                for gr in group:
                    elements = removed_nan.loc[removed_nan['group'] == gr][column].to_numpy().astype(np.float)
                    elements_list.append(elements)
                    
                    if elements.size >= 8:
                        k2, p = stats.normaltest(removed_nan.loc[removed_nan['group'] == gr][column].to_numpy().astype(np.float))
                        p_normal_list.append(p)
                
                if len(group) == 2:
                    if len(p_normal_list) == len(group) and all(i >= 0.05 for i in p_normal_list):
                        s, p = stats.ttest_ind(elements_list[0], elements_list[1], equal_var = False)
                        statistic_values = {'type': "welch_t_test", 's': s, 'p': p}
                    else:
                        if all(len(i) >= 20 for i in elements_list):
                            if np.size(np.unique(elements_list[0])) > 1 and np.size(np.unique(elements_list[1])) > 1:
                                s, p = stats.mannwhitneyu(elements_list[0], elements_list[1])
                                statistic_values = {'type': "mann_whitney_u", 's': s, 'p': p}
                            else:
                                statistic_values = {'type': "none_not_unique", 's': report_empty_tests, 'p': report_empty_tests}
                        else:
                            statistic_values = {'type': "none_not_enough_observations", 's': report_empty_tests, 'p': report_empty_tests}
                else:#anova???
                    statistic_values = {'type': "anova", 's': report_empty_tests, 'p': report_empty_tests}
                
                statistics_dict_list = add_to_statistics_dict(statistics_dict_list, column, "mean_diff_significance", statistic_values, 0, group)
                
    return statistics_dict_list
                            
#replaces category numbers for the data codings of the meta data
def set_data_coding(subset_data_df: pd.core.frame.DataFrame, meta_data: list, column: int) -> pd.core.frame.DataFrame:
    for meta_data_dict in meta_data:
        if column == meta_data_dict['column']:
            try:
                subset_data_df = subset_data_df.replace({column: meta_data_dict['data_coding']['data_coding_types']})
            except:
                pass
            
    return subset_data_df

#calculate chi2
def get_chi2(subset_data_df: pd.core.frame.DataFrame, meta_data: list, statistics_dict_list: list = None, use_ordinal_data = True,  remove_values: list = None) -> list:
    from UKB_tools.preprocessing import filter_coding_type

    if statistics_dict_list == None:
        statistics_dict_list = []
        
    for column in subset_data_df:        
        if column != "eid" and column != "group":
            removed_nan = remove_nan(subset_data_df, column)
            group = get_group(removed_nan, column)
            
            if remove_values != None:
                removed_nan = filter_coding_type(removed_nan, remove_values, column, True)
            
            if use_ordinal_data == False:
                isin_ordinal_data = isin_ordinal_data_codings(meta_data, column)
                if isin_ordinal_data == True:
                    get_statistic = False
                else:
                    get_statistic = True
            else:
                get_statistic = True
            
            if 'group' in removed_nan and get_statistic == True and len(removed_nan) > 0:
                contigency = pd.crosstab(removed_nan['group'], removed_nan[column])
                c, p, dof, expected = stats.chi2_contingency(contigency)
                statistic_values = {'c': c, 'p': p, 'dof': dof, 'expected': expected.tolist()}
                
                statistics_dict_list = add_to_statistics_dict(statistics_dict_list, column, "chi2", statistic_values, 0, group)
                
    return statistics_dict_list

#calculate kruskal wallis
def get_kruskal_wallis(subset_data_df: pd.core.frame.DataFrame, meta_data: list, statistics_dict_list: list = None, use_nominal_data = True,  remove_values: list = None) -> list:
    from UKB_tools.preprocessing import filter_coding_type
    
    if statistics_dict_list == None:
        statistics_dict_list = []
        
    for column in subset_data_df:
        if column != "eid" and column != "group":
            removed_nan = remove_nan(subset_data_df, column)
            group = get_group(removed_nan, column)

            if remove_values != None:
                removed_nan = filter_coding_type(removed_nan, remove_values, column, True)
            
            if use_nominal_data == False:
                isin_ordinal_data = isin_ordinal_data_codings(meta_data, column)
                if isin_ordinal_data == True:
                    get_statistic = True
                else:
                    get_statistic = False
            else:
                get_statistic = True
            
            if 'group' in removed_nan and get_statistic == True and len(removed_nan) > 0:
                groups_list = []
                
                for gro in group:
                    groups_list.append(removed_nan.loc[removed_nan['group'] == gro][column])
                
                try:
                    field_path = "(exec('from scipy import stats') or stats.kruskal("
                    
                    i = 0
                    for groups_l in groups_list:
                        if i > 0:
                            field_path = field_path + ","
                            
                        field_path = field_path + "groups_list[" + str(i) + "]"
                        i += 1
                    field_path = field_path + "))"
                       
                    locals_dict = locals().copy()
                    s, p = eval(field_path, locals_dict)
                    
                    statistic_values = {'s': s, 'p': p}
                    statistics_dict_list = add_to_statistics_dict(statistics_dict_list, column, "kruskal_wallis", statistic_values, 0, group)
                except:
                    pass
                
    return statistics_dict_list

#calculate fisher exact
def get_fisher_exact(subset_data_df: pd.core.frame.DataFrame, meta_data: list, statistics_dict_list: list = None,  remove_values: list = None) -> list:
    from UKB_tools.preprocessing import filter_coding_type
    
    if statistics_dict_list == None:
        statistics_dict_list = []
        
    for column in subset_data_df:
        if column != "eid" and column != "group":  
            removed_nan = remove_nan(subset_data_df, column)
            group = get_group(removed_nan, column)

            if remove_values != None:
                removed_nan = filter_coding_type(removed_nan, remove_values, column, True)
            
            if len(group) == 2 and 'group' in removed_nan:
                data_coding_types = removed_nan[column].unique()
                data_coding_types_comb = list(itertools.combinations(data_coding_types, 2))
                
                statistic_values = {}
                for combination in data_coding_types_comb:
                    removed_nan_combination = removed_nan.loc[(removed_nan[column] == combination[0]) | (removed_nan[column] == combination[1])]
                    contigency = pd.crosstab(removed_nan_combination['group'], removed_nan_combination[column])
                    
                    if contigency.size == 4:
                        o, p = stats.fisher_exact(contigency)
                        contigency_table = contigency.to_dict()
                        
                        number_of_cases = {}
                        for gro in group:
                            number_of_cases[gro] = len(removed_nan_combination.loc[(removed_nan_combination['group'] == gro)])
                    else:
                        o = 999
                        p = 999
                        number_of_cases = {}
                        contigency_table = {}
                        
                    statistic_values[combination] = {'o': o, 'p': p, 'number_of_cases': number_of_cases, 'contigency_table': contigency_table}
                
                statistics_dict_list = add_to_statistics_dict(statistics_dict_list, column, "fisher_exact", statistic_values, 0, group)
                
    return statistics_dict_list

#calculate bonferroni corrected p value
def get_bonferroni(statistics_dict_list: list, p_value_crit: float, p_value_location: str) -> list:
    from statsmodels.stats.multitest import multipletests
    
    new_statistics_dict_list = []
    number_of_tests = len(statistics_dict_list)
    
    p_values = []
    
    for statistics_dict in statistics_dict_list:
        p_values.append(statistics_dict[p_value_location])
        
        if statistics_dict[p_value_location] <= p_value_crit / number_of_tests:
            statistics_dict['bonferroni_corrected_p_value'] = "pass"
        else:
            statistics_dict['bonferroni_corrected_p_value'] = "fail"
            
        new_statistics_dict_list.append(statistics_dict)
    
    holm = multipletests(p_values, 0.1, "holm")
    i = 0
    for statistics_dict in statistics_dict_list:
        if holm[0][i] == True:
            statistics_dict['holm_p'] = "pass"
            statistics_dict['holm_p2'] = holm[1][i]
        else:
            statistics_dict['holm_p'] = "fail"
            statistics_dict['holm_p2'] = holm[1][i]
        i += 1
    
    return new_statistics_dict_list

#remove dictionaries that do not have not enough cases
def filter_minimum_number_of_cases(statistics_dict_list: list, minimum_number_of_cases: int) -> list:
    new_statistics_dict_list = []
    
    for statistics_dict in statistics_dict_list:
        skip = False
        for number_of_cases_index in statistics_dict['number_of_cases']:
            if statistics_dict['number_of_cases'][number_of_cases_index] < minimum_number_of_cases:
                skip = True
        if skip == True:
            continue
        
        new_statistics_dict_list.append(statistics_dict)
        
    return new_statistics_dict_list

def filter_statistic_columns(statistics_dict_list: list, filter_type: int, argument: list, df: pd.core.frame.DataFrame = pd.DataFrame()) -> list:
    #filter_type: 0 [statistic_type, statistic_value], statistic [columns]; 1, columns list
    new_statistics_dict_list = []
    new_df = df[['eid', 'diagnosis_group']]
    
    if filter_type == 0:
        for statistics_dict in statistics_dict_list:
            if statistics_dict[argument[0]] <= argument[1]:
                new_statistics_dict_list.append(statistics_dict)
                cols =  df.columns[df.columns.str.startswith(statistics_dict['column'])].tolist()
                
                for col in cols:
                    new_df[col] = df[col]
                    
    elif filter_type == 1:
        for column in argument:

            new_statistics_dict_list.append({'column': column})
            cols =  df.columns[df.columns.str.startswith(column)].tolist()
            
            for col in cols: 
                new_df[col] = df[col]
    
    if df.empty:
        return new_statistics_dict_list
    else:
        return new_statistics_dict_list, new_df

#Convert a statistic to separate dictionaries
def convert_statistic_to_columns(statistics_dict_list: list, statistic: str, new_statistic_name: str) -> list:
    new_statistics_dict_list = []
    
    for statistic_dict in statistics_dict_list:
        old_column_name = statistic_dict['column']
        
        if statistic in statistic_dict:            
            for statistic_sub_dict in statistic_dict[statistic]:
                statistic_values = {}
                new_statistic_dict = statistic_dict.copy()
                
                if not isinstance(statistic_sub_dict, str):
                    dict_name = '_'.join(statistic_sub_dict)
                else:
                    dict_name = statistic_sub_dict
                
                new_column_name =str(old_column_name + "_" + dict_name)
                new_statistic_dict['column'] = new_column_name
                new_statistic_dict['id'] = get_new_id()
                
                new_statistics_dict_list.append(new_statistic_dict)
                
                statistic_values = statistic_dict[statistic][statistic_sub_dict]
                new_statistics_dict_list = add_to_statistics_dict(new_statistics_dict_list, new_column_name, new_statistic_name, statistic_values, 1, new_statistic_dict['group'])
                
        else:
            new_statistics_dict_list.append(statistic_dict.copy())
                       
    return new_statistics_dict_list

#sort statistics dictionaries list
def sort_statistic_list(statistics_dict_list: list, statistic: str, descending: bool = False, export_sorting_statistic: str = None) -> list:
    sorting_dict_list = []
    new_statistics_dict_list = []
    
    for statistics_dict in statistics_dict_list:        
        if statistic == "mean":
            sort_value = sort_mean(statistics_dict)
        elif statistic == "chi2":
            sort_value = sort_chi2(statistics_dict)
        elif statistic == "mean_diff_significance":
            sort_value = sort_mean_difference_significance(statistics_dict)
        elif statistic == "converted_fisher_exact":
            sort_value = sort_converted_fisher_exact(statistics_dict)
        elif statistic == "kruskal_wallis":
            sort_value = sort_kruskal_wallis(statistics_dict)
        else:
            sort_value = sort_else(statistics_dict, statistic)
        
        if sort_value != None:
            sorting_dict_list.append({'id': statistics_dict['id'], 'value': sort_value})
            
    if len(sorting_dict_list) != 0:  
        sorting_dict_list = sorted(sorting_dict_list, key=lambda k: k['value'], reverse = descending) 
    
    for sorting_dict in sorting_dict_list:
        for statistics_dict in statistics_dict_list:
            if sorting_dict['id'] == statistics_dict['id']:
                if export_sorting_statistic != None:
                    statistics_dict[export_sorting_statistic] = sorting_dict['value']
                    
                new_statistics_dict_list.append(statistics_dict)
    
    return new_statistics_dict_list

def sort_mean(statistics_dict: dict) -> float:
    largest_diff = 0
    
    try:
        for row in statistics_dict['mean_diff']:
            for key in statistics_dict['mean_diff'][row]:
                diff = statistics_dict['mean_diff'][row][key]
                if diff > largest_diff:
                    largest_diff = diff
                    
        return largest_diff
    except:
        return None
    
def sort_chi2(statistics_dict: dict) -> float:
    try:
        return statistics_dict['chi2']['p']
    except:
        return None
    
def sort_kruskal_wallis(statistics_dict: dict) -> float:
    try:
        return statistics_dict['kruskal_wallis']['p']
    except:
        return None
    
def sort_mean_difference_significance(statistics_dict: dict) -> float:
    try:
        return statistics_dict['mean_diff_significance']['p']
    except:
        return None
    
def sort_converted_fisher_exact(statistics_dict: dict) -> float:
    try:
        return statistics_dict['converted_fisher_exact']['p']
    except:
        return None
        
def sort_else(statistics_dict: dict, statistic: str) -> float:
    try:
        return statistics_dict[statistic]
    except:
        return None

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

#save selected data from dictionaries list as tsv
def export_statistics_dict_list(statistics_dict_list: list, save_path: str, save_type: str = "json", fields: list = None):
    header = True
    f = open(save_path, "w+")
    f.close()
    
    for statistics_dict in statistics_dict_list:
        if save_type == "json":
            print_object = json.dumps(statistics_dict, cls = NpEncoder) 
        elif save_type == "tab_separated":
            print_object = ""
            
            if header == True:
                for field in fields:
                    print_object = print_object + str(field) + "\t"
                    header = False
                print_object = print_object + "\n"
                
            for field in fields:
                field_path = "statistics_dict"
                
                if "\*" in field:
                    field_split = field.split("\*")
                    
                    for sub_field in field_split:
                        field_path = field_path + "['" + sub_field + "']"
                else:
                    field_path = field_path + "['" + field + "']"
                
                locals_dict = locals().copy()
                r = eval(field_path, locals_dict)
                print_object = print_object + str(r) + "\t"
        
        f = open(save_path, "a")
        f.write(print_object)
        f.write("\n")
        f.close()
        
