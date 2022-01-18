import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
import json
import random
import sys
import pickle

import UKB_tools.preprocessing as pre
import UKB_tools.statistics as sta
import UKB_tools.support as sup
import UKB_tools.logistic_regression as log
#import UKB_tools.user_defined_functions as use

begin_time = datetime.datetime.now()

test = False
hpc = False
create_subset_data = False
keep_columns = True
read_all_selected_cases_df = False
sex_female = False
save = 3
if save == 1:
    save_path = "preprocessing/"
elif save == 2:
    save_path = "preprocessing2/"
elif save == 3:
    save_path = "preprocessing3/"
elif save == 4:
    save_path = "preprocessing4/"
elif save == 5:
    save_path = "preprocessing5/"
elif save == 6:
    save_path = "preprocessing6/"

#binary categorical columns:
binary_categorical_columns = ["4631-0.0", "3606-0.0", "31-0.0", "22001-0.0", "4204-0.0", "1990-0.0", "2188-0.0", "22132-0.0", "22128-0.0", "2207-0.0", "4598-0.0", "2814-0.0", "3799-0.0", "22135-0.0", "1797-0.0", "2492-0.0", "1980-0.0"]

#aneurysm cols
#keep_columns_list = ["2178-0.0", "904-0.0", "4653-0.0", "137-0.0", "100009-0.0", "2060-0.0", "1960-0.0", "100001-0.0", "1249-0.0", "6164-0.0", "20116-0.0", "6154-0.0", "6152-0.0", "30150-0.0", "874-0.0", "5463-0.0", "1873-0.0", "4548-0.0", "2453-0.0", "23100-0.0", "1568-0.0", "1727-0.0", "6177-0.0", "30090-0.0", "2070-0.0", "3606-0.0", "1528-0.0", "2277-0.0", "6150-0.0", "100017-0.0", "30080-0.0", "100019-0.0", "2335-0.0", "21001-0.0", "2754-0.0", "2020-0.0", "2080-0.0", "100014-0.0", "23104-0.0", "20018-0.0", "41235-0.0", "100015-0.0", "2473-0.0", "981-0.0", "699-0.0", "30160-0.0", "100025-0.0", "100011-0.0", "1090-0.0", "1299-0.0"]

#control cols
#keep_columns_list = ["23113-0.0", "23114-0.0", "23118-0.0", "23117-0.0", "1239-0.0", "23105-0.0", "23101-0.0", "23102-0.0", "21022-0.0", "23126-0.0", "23125-0.0", "23129-0.0", "23130-0.0", "23121-0.0", "23122-0.0", "31-0.0", "3064-0.0", "20116-0.0", "3063-0.0", "30010-0.0", "47-0.0", "21002-0.0", "46-0.0", "23098-0.0", "50-0.0", "3062-0.0", "20015-0.0", "41214-0.0", "6148-0.0", "23106-0.0", "23107-0.0", "48-0.0", "23109-0.0", "23108-0.0", "23110-0.0", "6032-0.0", "30270-0.0", "1100-0.0", "3088-0.0", "1070-0.0", "30020-0.0", "6017-0.0", "6024-0.0", "100009-0.0", "738-0.0", "4537-0.0", "30040-0.0", "20181-0.0", "2492-0.0", "6014-0.0", "30530-0.0", "6138-0.0", "3143-0.0", "30050-0.0", "6154-0.0", "41235-0.0", "6150-0.0", "23112-0.0", "20127-0.0", "2744-0.0", "20111-0.0", "137-0.0", "100001-0.0", "23116-0.0", "845-0.0", "4598-0.0", "2814-0.0", "22604-0.0", "42007-0.0", "2207-0.0", "22001-0.0", "30250-0.0", "22128-0.0", "22132-0.0", "21001-0.0", "709-0.0", "20150-0.0", "1031-0.0", "4631-0.0", "3606-0.0", "2090-0.0", "20153-0.0", "4204-0.0", "20151-0.0", "30260-0.0", "1990-0.0", "24024-0.0", "100920-0.0", "2415-0.0", "2794-0.0", "23104-0.0", "30510-0.0", "4196-0.0", "136-0.0", "100150-0.0", "4198-0.0", "49-0.0", "924-0.0", "1558-0.0", "1001-0.0", "5540-0.0", "2188-0.0", "20090-0.0"]

#new list 10.000
#keep_columns_list = ["20511-0.0", "20513-0.0", "103010-0.0", "22132-0.0", "22136-0.0", "23105-0.0", "23102-0.0", "30010-0.0", "3064-0.0", "31-0.0", "21002-0.0", "100920-0.0", "20510-0.0", "47-0.0", "20015-0.0", "20494-0.0", "3062-0.0", "20116-0.0", "23106-0.0", "20094-0.0", "50-0.0", "1100-0.0", "6141-0.0", "46-0.0", "30270-0.0", "41214-0.0", "30020-0.0", "20488-0.0", "48-0.0", "103100-0.0", "30040-0.0", "6154-0.0", "6148-0.0", "103030-0.0", "1070-0.0", "20490-0.0", "30050-0.0", "20524-0.0", "30260-0.0", "5540-0.0", "100280-0.0", "40013-0.0", "6017-0.0", "136-0.0", "6032-0.0", "103180-0.0", "20518-0.0", "21001-0.0", "20497-0.0", "30250-0.0", "6024-0.0", "42009-0.0", "20491-0.0", "3143-0.0", "30530-0.0", "3799-0.0", "20487-0.0", "20043-0.0", "5182-0.0", "20117-0.0", "2824-0.0", "3148-0.0", "21022-0.0", "137-0.0", "104240-0.0", "20458-0.0", "22135-0.0", "6142-0.0", "1558-0.0", "4198-0.0", "738-0.0", "10711-0.0", "20525-0.0", "6150-0.0", "41235-0.0", "6014-0.0", "20016-0.0", "4204-0.0", "20466-0.0", "1031-0.0", "10115-0.0", "20051-0.0", "20495-0.0", "6146-0.0", "100009-0.0", "20042-0.0", "100001-0.0", "20531-0.0", "104190-0.0", "3088-0.0", "1797-0.0", "709-0.0", "6138-0.0", "30510-0.0", "20048-0.0", "41219-0.0", "845-0.0", "10723-0.0", "20127-0.0", "2492-0.0", "1980-0.0", "2188-0.0", "30000-0.0", "100150-0.0", "3581-0.0", "4537-0.0", "1438-0.0", "22128-0.0", "4100-0.0", "20439-0.0", "2149-0.0", "22604-0.0", "5890-0.0", "1990-0.0", "30300-0.0", "104080-0.0", "4119-0.0", "20193-0.0", "20047-0.0", "699-0.0", "1080-0.0", "4631-0.0", "4080-0.0", "4598-0.0", "20112-0.0", "20044-0.0", "1727-0.0", "4196-0.0", "30140-0.0"]

#new list 500.000
keep_columns_list = ["103100-0.0", "30535-0.0", "22138-0.0", "22131-0.0", "90016-0.0", "22141-0.0", "10711-0.0", "20041-0.0", "103030-0.0", "22136-0.0", "22132-0.0", "23105-0.0", "23102-0.0", "3064-0.0", "20046-0.0", "30010-0.0", "20116-0.0", "31-0.0", "20045-0.0", "3062-0.0", "20015-0.0", "23106-0.0", "22139-0.0", "50-0.0", "41214-0.0", "100920-0.0", "1100-0.0", "6141-0.0", "100350-0.0", "6154-0.0", "30270-0.0", "22135-0.0", "30020-0.0", "1070-0.0", "6148-0.0", "30040-0.0", "6017-0.0", "136-0.0", "6032-0.0", "21001-0.0", "22140-0.0", "30260-0.0", "30050-0.0", "738-0.0", "5540-0.0", "3088-0.0", "20117-0.0", "3799-0.0", "20484-0.0", "30530-0.0", "20016-0.0", "104240-0.0", "21022-0.0", "137-0.0", "41235-0.0", "20439-0.0", "6142-0.0", "6014-0.0", "6138-0.0", "20094-0.0", "100009-0.0", "1558-0.0", "6146-0.0", "30250-0.0", "845-0.0", "6150-0.0", "1031-0.0", "709-0.0", "4198-0.0", "4537-0.0", "1980-0.0", "20127-0.0", "2188-0.0", "104190-0.0", "100480-0.0", "20454-0.0", "3148-0.0", "100001-0.0", "22604-0.0", "103180-0.0", "2824-0.0", "1080-0.0", "30510-0.0", "10776-0.0", "41219-0.0", "924-0.0", "4119-0.0", "20524-0.0", "103010-0.0", "699-0.0", "24024-0.0", "20191-0.0", "4598-0.0", "30000-0.0", "1438-0.0", "100280-0.0", "100150-0.0", "5890-0.0", "680-0.0", "2149-0.0", "1990-0.0", "1200-0.0", "104080-0.0", "10115-0.0", "2492-0.0", "1727-0.0", "100850-0.0", "1835-0.0", "1538-0.0", "20022-0.0", "1797-0.0", "20018-0.0"]

#small list 500.000
#keep_columns_list = ["103100-0.0", "22138-0.0", "22131-0.0", "22141-0.0", "103030-0.0", "22136-0.0", "22132-0.0", "23105-0.0", "23102-0.0", "3064-0.0", "30010-0.0", "20116-0.0", "31-0.0", "21022-0.0", "3062-0.0", "20015-0.0", "23106-0.0", "22139-0.0", "50-0.0", "41214-0.0", "100920-0.0", "1100-0.0", "6141-0.0"]

if sex_female == True:
    female_only_cols = ["2674-0.0", "2684-0.0", "2694-0.0", "2704-0.0", "2714-0.0", "2724-0.0", "2734-0.0", "2744-0.0", "2754-0.0", "2764-0.0", "2774-0.0", "2784-0.0", "2794-0.0", "2804-0.0", "2814-0.0", "2824-0.0", "2834-0.0", "2844-0.0", "3140-0.0", "3536-0.0", "3546-0.0", "3581-0.0", "3591-0.0", "3700-0.0", "3710-0.0", "3720-0.0", "3829-0.0", "3839-0.0", "3849-0.0", "3872-0.0", "3882-0.0", "4041-0.0", "6153-0.0", "10132-0.0", "10844-0.0", "41219-0.0", "41220-0.0", "41221-0.0", "41222-0.0", "41223-0.0", "41224-0.0", "41225-0.0", "41226-0.0", "41227-0.0", "41228-0.0", "21026-0.0", "21050-0.0"]
else:
    female_only_cols = None

keep_columns_list = pre.remove_columns_dictionary(keep_columns_list, save_path + "remove_columns.txt", female_only_cols)

#search_for_diagnosis = {"aneurysm": ["I671"], "sab": ["I600","I601","I602","I603","I604","I605","I606","I607","I608","I609"]} #I671 = aneurysm
search_for_diagnosis = {"sab": ["I600","I601","I602","I603","I604","I605","I606","I607","I608","I609"]} #I671 = aneurysm
#search_for_diagnosis = {"aneurysm": ["I671"], "sab": ["I600","I601","I602","I603","I604","I605","I606","I607","I608","I609"], 'DM': ['E100', 'E101', 'E102', 'E103', 'E104', 'E105', 'E106', 'E107', 'E108', 'E109', 'E110', 'E111', 'E112', 'E113', 'E114', 'E115', 'E116', 'E117', 'E118', 'E119', 'E120', 'E121', 'E122', 'E123', 'E124', 'E125', 'E126', 'E127', 'E128', 'E129', 'E130', 'E131', 'E132', 'E133', 'E134', 'E135', 'E136', 'E137', 'E138', 'E139', 'E140', 'E141', 'E142', 'E143', 'E144', 'E145', 'E146', 'E147', 'E148', 'E149'], 'Hypertension':  ['I10', 'I150', 'I151', 'I152', 'I158', 'I159'], 'Hypercholesterolemia': ['E780']}

if hpc == True:
    basket_location = "../../../hpc/hers_en/shared/Aneurysm/UKBB/basket_"
else:
    basket_location = "basket_"

if test == True:
    data_1 = "Subset_data"
    data_2 = "Subset_data_2"
    data_list = [data_1, data_2]
else:
    data_1 = "2008454"
    data_2 = "10036"
    data_3 = "2012163"
    data_list = [data_1, data_2, data_3]
    
#read-in icd10 columns
print("step 2")
df_icd10 = pre.read_basket(basket_location + data_1 + ".csv", pre.select_column_names(["41270-0.0", "41280-0.0"], pre.get_column_names(pre.read_basket(basket_location + data_1 + ".csv", nrows = 1)), 1))

#create dictionary with diagnosis cases + diagnosis dates and add some other columns from basket_10036, add control cases control cases and combine aneurysm sab
print("step 3")
#selected_cases_dict_list = pre.select_column_data(pre.read_basket(basket_location + data_2 + ".csv", columns = ["eid", "31-0.0", "34-0.0", "52-0.0", "53-0.0"]), {"sex": "31-0.0", "birth_year": "34-0.0", "birth_month": "52-0.0", "date_attending_assesment_centre": "53-0.0"}, pre.combine_diagnoses(pre.select_control_cases(df_icd10, pre.select_icd_10_diagnoses(df_icd10, search_for_diagnosis, 1, 1), [1, "combined_diagnoses"], 1), ["sab", "aneurysm"]))
selected_cases_dict_list = pre.select_column_data(pre.read_basket(basket_location + data_2 + ".csv", columns = ["eid", "31-0.0", "34-0.0", "52-0.0", "53-0.0"]), {"sex": "31-0.0", "birth_year": "34-0.0", "birth_month": "52-0.0", "date_attending_assesment_centre": "53-0.0"}, pre.select_control_cases(df_icd10, pre.select_icd_10_diagnoses(df_icd10, search_for_diagnosis, 1, 1), [3, "100000"], 1))
#selected_cases_dict_list = pre.select_column_data(pre.read_basket(basket_location + data_2 + ".csv", columns = ["eid", "31-0.0", "34-0.0", "52-0.0", "53-0.0"]), {"sex": "31-0.0", "birth_year": "34-0.0", "birth_month": "52-0.0", "date_attending_assesment_centre": "53-0.0"}, pre.select_control_cases(df_icd10, pre.select_icd_10_diagnoses(df_icd10, search_for_diagnosis, 1, 1), [4, ""], 1))

if sex_female == True:
    print("step 3.2")
    selected_cases_dict_list = pre.filter_selected_cases(selected_cases_dict_list, 'sex', "0")

#vita selection
#selected_cases_dict_list = pre.select_column_data(pre.read_basket(basket_location + data_2 + ".csv", columns = ["eid", "31-0.0", "34-0.0", "52-0.0"]), {"sex": "31-0.0", "birth_year": "34-0.0", "birth_month": "52-0.0"}, pre.select_control_cases(df_icd10, pre.select_icd_10_diagnoses(df_icd10, search_for_diagnosis, 1, 1), [0, ""], 1))
#pre.export_selected_cases_dict_list(selected_cases_dict_list, save_path + "export_eids.csv", "comma_separated", ["eid", "diagnoses\*aneurysm", "diagnoses\*sab", "diagnoses\*DM", "diagnoses\*Hypertension", "diagnoses\*Hypercholesterolemia", "missing"])
#sys.exit()

print("step 3.3")
pre.export_selected_cases_dict_list(selected_cases_dict_list, save_path + "export_selected_eids.csv", "comma_separated", ["eid", "diagnoses\*aneurysm", "diagnoses\*sab", "diagnoses\*control", "diagnoses\*combined_diagnoses", "diagnoses\*DM", "diagnoses\*Hypertension", "missing"])

#filter list of cases for occurence of selected diagnoses
print("step 4")
#selected_cases_dict_list, filtered_list = pre.filter_diagnoses(selected_cases_dict_list, [{"combined_diagnoses": "*"}, {"control": "*"}])
selected_cases_dict_list, filtered_list = pre.filter_diagnoses(selected_cases_dict_list, [{"control": "*"}, {"sab": "*"}])

#filter list of cases for occurence of selected diagnoses and visiting assesment center
print("step 4.2")
selected_cases_dict_list = pre.filter_case_date(selected_cases_dict_list, "date_attending_assesment_centre")

#create subset of basket based on filtered data
print("step 5")
if create_subset_data == True:
    for file in data_list:
        #save subsetted data dictionary to csv file
        subset_data = pre.subset_data(basket_location + file + ".csv",  selected_cases_dict_list, save_path + "basket_" + file + "_subset.csv", None, 0, 0)

filter_columns = pre.get_filter_columns(save_path + "remove_columns.txt", female_only_cols)

print("step 6")
#create group for sab and aneurysm
selected_cases_dict_list = pre.create_diagnoses_group(selected_cases_dict_list)

all_selected_cases_df = pd.DataFrame(columns=['eid', 'diagnosis_group'])
for selected_cases_dict in selected_cases_dict_list:
    all_selected_cases_df = all_selected_cases_df.append({'eid': selected_cases_dict['eid'], 'diagnosis_group': selected_cases_dict['diagnosis_group']}, ignore_index=True)
    
statistics_dict_list = []
j = 0
for file in data_list:
    j += 1
    print("File " + str(j) + " out of " + str(len(data_list)))
    meta_data_dict_list = pre.open_meta_data(save_path + "basket_" + file + "_meta_data.txt")
    meta_data_dict_list = pre.filter_instance(meta_data_dict_list, "0.0")
    i = 0
    length = len(meta_data_dict_list)
    sup.printProgressBar(i, length, prefix = 'Progress:', suffix = 'Complete', length = 50)
    
    #sort selected cases eid to match the csv file
    sorted_selected_cases_dict_list = pre.sort_selected_cases_dict_list(save_path + "basket_" + file + "_subset.csv", selected_cases_dict_list)
    
    for meta_data_dict in meta_data_dict_list:
        i += 1
        sup.printProgressBar(i, length, prefix = 'Progress:', suffix = 'Complete', length = 50)
        
        loop_selected_cases_dict_list = sorted_selected_cases_dict_list.copy()
        
        column = meta_data_dict['column']
                
        #print(column)
        value_type = meta_data_dict['value_type']
        
        if keep_columns == True:
            if column not in keep_columns_list:
                continue
        else:
            if column in filter_columns:
                continue
        
        #filter diagnosis later than column debut
        begin_time = datetime.datetime.now()
        loop_selected_cases_dict_list = pre.filter_column_date(loop_selected_cases_dict_list, column, meta_data_dict_list, 0)
                
        #combine selected cases and columns in dataframe and create group column ##pass function as argument user_defined_grouping to create manual grouping
        selected_cases_df = pre.get_grouped_data(save_path + "basket_" + file + "_subset.csv", [column], loop_selected_cases_dict_list, ["diagnosis_group"], 1)
        
        if value_type.startswith("Continuous"):      
            if keep_columns == False:
                #create statistics dictionary per column and calculate mean per group
                begin_time = datetime.datetime.now()
                statistics_dict_list = sta.get_mean(selected_cases_df, statistics_dict_list, None)
                statistics_dict_list = sta.get_mean_difference_significance(selected_cases_df, statistics_dict_list, 999, None)
            
            if read_all_selected_cases_df == False:
                all_selected_cases_df = pre.merge_dataframes(all_selected_cases_df, selected_cases_df, [column], 'eid', False, None)
                
        if value_type.startswith("Integer"):  
            if keep_columns == False:
                begin_time = datetime.datetime.now()
                statistics_dict_list = sta.get_mean(selected_cases_df, statistics_dict_list, [-1, -3, -4, -10])
                statistics_dict_list = sta.get_mean_difference_significance(selected_cases_df, statistics_dict_list, 999, [-1, -3, -4, -10])
            
            if read_all_selected_cases_df == False:
                all_selected_cases_df = pre.merge_dataframes(all_selected_cases_df, selected_cases_df, [column], 'eid', False, [-1, -3, -4, -10])
            
        if value_type.startswith("Categorical"): 
            if keep_columns == False:
                
                statistics_dict_list = sta.get_chi2(selected_cases_df, [meta_data_dict], statistics_dict_list, False, [-1, -3, -818, -121])
                
                statistics_dict_list = sta.get_kruskal_wallis(selected_cases_df, [meta_data_dict], statistics_dict_list, False, [-1, -3, -818, -121])
                
                statistics_dict_list = sta.add_to_statistics_dict(statistics_dict_list, column, "data_coding_types", {'data_coding_types': meta_data_dict['data_coding']['data_coding_types']}, 1, None)
                
            if read_all_selected_cases_df == False:
                if column not in binary_categorical_columns:
                    if sta.isin_ordinal_data_codings([meta_data_dict], column):
                        binary_selected_cases_df = pre.make_categories_binary(selected_cases_df, [meta_data_dict], column, [-1, -3, -818, -121], True)
                        all_selected_cases_df = pre.merge_dataframes(all_selected_cases_df, binary_selected_cases_df, [column], 'eid', True, [-1, -3, -818, -121])
                    else:
                        binary_selected_cases_df, added_columns = pre.create_dummies(selected_cases_df, [meta_data_dict], column, [-1, -3, -818, -121], False)
                        if len(added_columns) <= 100:
                            all_selected_cases_df = pre.merge_dataframes(all_selected_cases_df, binary_selected_cases_df, added_columns, 'eid', True, [-1, -3, -818, -121])
                else:
                    all_selected_cases_df = pre.merge_dataframes(all_selected_cases_df, selected_cases_df, [column], 'eid', True, [-1, -3, -818, -121])
            
        if keep_columns == False:
            #add other data to dictionaries
            statistics_dict_list = sta.add_to_statistics_dict(statistics_dict_list, column, "description", {'description': meta_data_dict['description']}, 1, None)
            statistics_dict_list = sta.add_to_statistics_dict(statistics_dict_list, column, "value_type", {'value_type': meta_data_dict['value_type']}, 1, None)

if keep_columns == False:
    #filter minimum number of cases and sort largest mean difference
    print("step 7")
    mean_statistics_dict_list = sta.sort_statistic_list(statistics_dict_list, "mean_diff_significance", False, "export_p")
    chi_statistics_dict_list = sta.sort_statistic_list(statistics_dict_list, "chi2", False, "export_p")
    kruskal_wallis_statistics_dict_list = sta.sort_statistic_list(statistics_dict_list, "kruskal_wallis", False, "export_p")

    #remove columns that are not adding anything and do bonferroni correction
    print("step 8")
    mean_statistics_dict_list = sta.get_bonferroni(pre.remove_columns_dictionary(mean_statistics_dict_list, save_path + "remove_columns.txt", female_only_cols), 0.1, "export_p")
    chi_statistics_dict_list = sta.get_bonferroni(pre.remove_columns_dictionary(chi_statistics_dict_list, save_path + "remove_columns.txt", female_only_cols), 0.1, "export_p")
    kruskal_wallis_statistics_dict_list = sta.get_bonferroni(pre.remove_columns_dictionary(kruskal_wallis_statistics_dict_list, save_path + "remove_columns.txt", female_only_cols), 0.1, "export_p")
    
    #export
    print("step 9")  
    sta.export_statistics_dict_list(mean_statistics_dict_list, save_path + "statistics_dict_list_mean.tsv", "tab_separated", ["column", "description", "mean_diff_significance", "mean", "mean_diff", "bonferroni_corrected_p_value"])
    sta.export_statistics_dict_list(chi_statistics_dict_list, save_path + "statistics_dict_list_chi2.tsv", "tab_separated", ["column", "description", "chi2\*c", "chi2\*p", "data_coding_types\*data_coding_types", "bonferroni_corrected_p_value"])
    sta.export_statistics_dict_list(kruskal_wallis_statistics_dict_list, save_path + "statistics_dict_list_kruskal_wallis.tsv", "tab_separated", ["column", "description", "kruskal_wallis\*s", "kruskal_wallis\*p", "data_coding_types\*data_coding_types", "bonferroni_corrected_p_value"])
    
    #add all statistics and calculate bonferroni
    print("step 10")
    total_statistics_list = mean_statistics_dict_list + chi_statistics_dict_list + kruskal_wallis_statistics_dict_list
 
    #sort and select p < 0.1/number_of_features then sort
    total_statistics_list = sta.sort_statistic_list(total_statistics_list, "export_p", False, None)
    sta.export_statistics_dict_list(total_statistics_list, save_path + "total_bonferroni_corrected_p_values.tsv", "tab_separated", ["column", "description", "value_type", "bonferroni_corrected_p_value", "export_p", "holm_p", "holm_p2"])
    
filtered_dict_list = []
for total_statistics_dict in statistics_dict_list:
    if total_statistics_dict['column'] in filter_columns:
        filtered_dict_list.append(total_statistics_dict)
sta.export_statistics_dict_list(filtered_dict_list, save_path + "filtered_columns.tsv", "tab_separated", ["column", "description"])

#make dataframe containing all cases filtered by debut date and significant columns
print("step 11")

if read_all_selected_cases_df == False:
    if keep_columns == True:
        total_statistics_list, all_selected_cases_df = sta.filter_statistic_columns([], 1, keep_columns_list, all_selected_cases_df)
    else:
        total_statistics_list, all_selected_cases_df = sta.filter_statistic_columns(total_statistics_list, 0, ["export_p", 0.05], all_selected_cases_df)
    
    #clean up dataframe (remove _l,_r, remove empty cases, change column names), remove columns < 10% of data available, make figure and export all cases df to csv    
    all_selected_cases_df = pre.clean_dataframe(all_selected_cases_df)
    
    all_selected_cases_df.to_csv(save_path + "all_selected_cases_df.csv", sep = ',')
else:
    all_selected_cases_df = pd.read_csv(save_path + "all_selected_cases_df.csv")  

#do imputation and select train test group
print("step 12")

print("step 13")
meta_data_dict_list = []
for file in data_list:
    meta_data_dict_list_part = pre.open_meta_data(save_path + "basket_" + file + "_meta_data.txt")
    meta_data_dict_list = meta_data_dict_list + meta_data_dict_list_part

#[impute, round_impute], sample, search, polynominal
#scorer_type f1 aupcr fbeta0.5
#models_list, model_stats_df_list = log.multiple_do_train_test_model(all_selected_cases_df, "bootstrap", 10, ['eid'], "diagnosis_group", {0: "control", 1: "sab"}, "logistic_regression", meta_data_dict_list, [True, False], True, True, False, "aupcr") # ["30530-0.0", "30510-0.0"]
#models_list, model_stats_df_list = log.multiple_do_train_test_model(all_selected_cases_df, "bootstrap", 10, ['eid'], "diagnosis_group", {0: "control", 1: "sab"}, "svc", meta_data_dict_list, [True, False], True, True, False, "aupcr")
models_list, model_stats_df_list = log.multiple_do_train_test_model(all_selected_cases_df, "bootstrap", 10, ['eid'], "diagnosis_group", {0: "control", 1: "sab"}, "clf", meta_data_dict_list, [True, False], True, True, False, "aupcr")
#models_list, model_stats_df_list = log.multiple_do_train_test_model(all_selected_cases_df, "bootstrap", 10, ['eid'], "diagnosis_group", {0: "control", 1: "sab"}, "mlp", meta_data_dict_list, [True, False], True, True, False, "aupcr")
#models_list, model_stats_df_list = log.multiple_do_train_test_model(all_selected_cases_df, "bootstrap", 10, ['eid'], "diagnosis_group", {0: "control", 1: "sab"}, "gbc", meta_data_dict_list, [True, False], True, True, False, "aupcr")
#models_list, model_stats_df_list = log.multiple_do_train_test_model(all_selected_cases_df, "bootstrap", 10, ['eid'], "diagnosis_group", {0: "control", 1: "sab"}, "abc", meta_data_dict_list, [True, False], True, True, False, "aupcr")

print("step 14")
scorers_dict = {'accuracy': [], 'f1': [], 'recall': [], 'aupcr': [], 'precision': [], 'roc_auc': [], 'fbeta0.5': []}
for model_dict in models_list:
    scorers_dict['accuracy'].append(model_dict['test']['accuracy'])
    scorers_dict['f1'].append(model_dict['test']['f1'])
    scorers_dict['recall'].append(model_dict['test']['recall'])
    scorers_dict['aupcr'].append(model_dict['test']['aupcr'])
    scorers_dict['precision'].append(model_dict['test']['precision'])
    scorers_dict['roc_auc'].append(model_dict['test']['roc_auc'])
    scorers_dict['fbeta0.5'].append(model_dict['test']['fbeta0.5'])
pd.DataFrame(scorers_dict).to_csv(save_path + "scorers.csv", index = False)

#if models_list[0]['best_params']:
if 'best_params' in models_list[0]:
    with open(save_path + "best_params.txt", 'w') as f:
        for model_dict in models_list:
            f.write(str(model_dict['best_params']) + "\n")
            
if 'cv_results' in models_list[0]:
    with open(save_path + "cv_results.txt", 'w') as f:
        for model_dict in models_list:
            f.write(str(model_dict['cv_results']) + "\n")
            
with open(save_path + "confusion_matrix.txt", 'w') as f:
    for model_dict in models_list:
        f.write(str(model_dict['test']['matrix']) + "\n")
        
with open(save_path + "roc_curve_test_values.txt", 'w') as f:
    for model_dict in models_list:
        f.write(str(model_dict['roc_curve_test']) + "\n")
        
with open(save_path + "precission_recall_test_values.txt", 'w') as f:
    for model_dict in models_list:
        f.write(str(model_dict['precission_recall_plot_values']) + "\n")
        
with open(save_path + "true_number_of_sabs.txt", 'w') as f:
    f.write(str(model_stats_df_list[4]))
    
with open(save_path + "confusion_matrix_adj.txt", 'w') as f:
    for model_dict in models_list:
        f.write(str(model_dict['adjusted_thresholds']['cf_matrix_adj']) + "\n")
        
with open(save_path + "adjust_thresholds.json", 'w') as fp:
    for model_dict in models_list:
        json.dump(model_dict['adjusted_thresholds'], fp)
    
with open(save_path + "y_probs.json", 'w') as fp:
    for model_dict in models_list:
        y_probs_dict = {}

        y_probs_dict['y_true'] = model_dict['test']['true_values'].tolist()
        y_probs_dict['y_predicted'] = model_dict['predicted'].tolist()
        y_probs_dict['probabilities'] = model_dict['test']['prediction_probabilities'].tolist()
        
        json.dump(y_probs_dict, fp)

count = 0
for model_dict in models_list:
    try:
        model_dict['shap']['summary_plot'].savefig(save_path + "shap_plots/summary_plot/summary_plot_" + str(count) + ".png", bbox_inches='tight', dpi=600)
    except:
        print("err summary plot")
    try:
        model_dict['shap']['waterfall_plot'].savefig(save_path + "shap_plots/waterfall_plot/waterfall_plot_" + str(count) + ".png", bbox_inches='tight', dpi=600)
    except:
        print("err waterfall plot")
    try:
        model_dict['shap']['force_plot'].savefig(save_path + "shap_plots/force_plot/force_plot_" + str(count) + ".png", bbox_inches='tight', dpi=600)
    except:
        print("err force plot")
    try:
        model_dict['shap']['decision_plot'].savefig(save_path + "shap_plots/decision_plot/decision_plot_" + str(count) + ".png", bbox_inches='tight', dpi=600)
    except:
        print("err decision plot")
    try:
        model_dict['recall_plot'].savefig(save_path + "shap_plots/recall_plot/recall_plot" + str(count) + ".png", bbox_inches='tight', dpi=600)
    except:
        print("err recall plot")
    
    roc_values_df = pd.DataFrame({'False positive rate': model_dict['roc_curve_test']['test_fpr'], 'True positive rate': model_dict['roc_curve_test']['test_tpr']})
    sns.lineplot(data = roc_values_df, x = 'False positive rate', y = 'True positive rate', label='ROC').figure
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.legend()
    plt.savefig(save_path + "shap_plots/roc_test_plot/roc_test_plot_" + str(count) + ".png", bbox_inches='tight', dpi=600)
    plt.close('all')
    
    count += 1

last_row = model_stats_df_list[1].tail(1)
coef_df = last_row.T
coef_df = coef_df.reset_index()
coef_df.columns = ["Column", "coef"]

last_row = model_stats_df_list[2].tail(1)
coef_std_df = last_row.T
coef_std_df = coef_std_df.reset_index()
coef_std_df.columns = ["Column", "std"]

coef_df = coef_df.merge(coef_std_df, how='left', on='Column')

###
last_row = model_stats_df_list[6].tail(1)
shap_coef_df = last_row.T
shap_coef_df = shap_coef_df.reset_index()
shap_coef_df.columns = ["Column", "coef"]

last_row = model_stats_df_list[7].tail(1)
shap_coef_std_df = last_row.T
shap_coef_std_df = shap_coef_std_df.reset_index()
shap_coef_std_df.columns = ["Column", "std"]

shap_coef_df = shap_coef_df.merge(shap_coef_std_df, how='left', on='Column')

for file in data_list:
    meta_data_dict_list = pre.open_meta_data(save_path + "basket_" + file + "_meta_data.txt")
    
    for meta_data_dict in meta_data_dict_list:
        coef_df.loc[coef_df['Column'].str.startswith(meta_data_dict['column']), 'Description'] = meta_data_dict['description']
        shap_coef_df.loc[shap_coef_df['Column'].str.startswith(meta_data_dict['column']), 'Description'] = meta_data_dict['description']
        
        if meta_data_dict['value_type'].startswith("Categorical"):
            try:
                coef_df.loc[coef_df['Column'].str.startswith(meta_data_dict['column']), 'data_coding'] = str(meta_data_dict['data_coding']['data_coding_types'])
                shap_coef_df.loc[shap_coef_df['Column'].str.startswith(meta_data_dict['column']), 'data_coding'] = str(meta_data_dict['data_coding']['data_coding_types'])
            except:
                print(meta_data_dict['column'])

coef_df.to_csv(save_path + "coef_df_summary.csv", index = False)
shap_coef_df.to_csv(save_path + "shap_coef_df_summary.csv", index = False)

model_stats_df_list[0].to_csv(save_path + "coef_df.csv", index = False)
model_stats_df_list[1].to_csv(save_path + "coef_mean_df.csv", index = False)
model_stats_df_list[2].to_csv(save_path + "coef_std_df.csv", index = False)
model_stats_df_list[3].to_csv(save_path + "auc_df.csv", index = False)
model_stats_df_list[5].to_csv(save_path + "shap_values_feature_importance_df.csv", index = False)

print(models_list[-1]['encoding_scheme'])

print("best score matrix: " + str(models_list[0]['adjusted_thresholds']['cf_matrix_adj']))
print("best score adjusted: " + str(models_list[0]['adjusted_thresholds']['best_score_adj']))
print("threshold" + str(models_list[0]['adjusted_thresholds']['best_threshold']))

print("Runtime: " + str(datetime.datetime.now() - begin_time))
