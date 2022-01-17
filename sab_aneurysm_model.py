import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import os
import json
import random
import sys

#from UKB_tools.test import test
#from UKB_tools import *
import UKB_tools.test as t
import UKB_tools.preprocessing as pre
import UKB_tools.statistics as sta
import UKB_tools.support as sup
import UKB_tools.logistic_regression as log
#import UKB_tools.user_defined_functions as use

begin_time = datetime.datetime.now()
p1 = t.Person("John", 36)
print(p1.name)
p1.test("test")

test = False
hpc = False
create_subset_data = True
replace_aneurysm_control_group = False
keep_columns = True
save = 3
if save == 1:
    save_path = "preprocessing/"
elif save == 2:
    save_path = "preprocessing2/"
elif save == 3:
    save_path = "preprocessing3/"
    
keep_columns_list = ["2178-0.0", "904-0.0", "4653-0.0", "137-0.0", "100009-0.0", "2060-0.0", "1960-0.0", "100001-0.0", "1249-0.0", "6164-0.0", "20116-0.0", "6154-0.0", "6152-0.0", "30150-0.0", "874-0.0", "5463-0.0", "1873-0.0", "4548-0.0", "2453-0.0", "23100-0.0", "1568-0.0", "1727-0.0", "6177-0.0", "30090-0.0", "2070-0.0", "3606-0.0", "1528-0.0", "2277-0.0", "6150-0.0", "100017-0.0", "30080-0.0", "100019-0.0", "2335-0.0", "21001-0.0", "2754-0.0", "2020-0.0", "2080-0.0", "100014-0.0", "23104-0.0", "20018-0.0", "41235-0.0", "100015-0.0", "2473-0.0", "981-0.0", "699-0.0", "30160-0.0", "100025-0.0", "100011-0.0", "1090-0.0", "1299-0.0"]

#search_for_diagnosis = {"aneurysm": ["I671"], "sab": ["I600","I601","I602","I603","I604","I605","I606","I607","I608","I609"]} #I671 = aneurysm
#search_for_diagnosis = {"aneurysm": ["I671"]} #I671 = aneurysm
#search_for_diagnosis = {"sab": ["I600","I601","I602","I603","I604","I605","I606","I607","I608","I609"]} #I671 = aneurysm
#search_for_diagnosis = {"aneurysm": ["I671"], "sab": ["I600","I601","I602","I603","I604","I605","I606","I607","I608","I609"], 'DM': ['E100', 'E101', 'E102', 'E103', 'E104', 'E105', 'E106', 'E107', 'E108', 'E109', 'E110', 'E111', 'E112', 'E113', 'E114', 'E115', 'E116', 'E117', 'E118', 'E119', 'E120', 'E121', 'E122', 'E123', 'E124', 'E125', 'E126', 'E127', 'E128', 'E129', 'E130', 'E131', 'E132', 'E133', 'E134', 'E135', 'E136', 'E137', 'E138', 'E139', 'E140', 'E141', 'E142', 'E143', 'E144', 'E145', 'E146', 'E147', 'E148', 'E149'], 'Hypertension': ['I10', 'I150', 'I151', 'I152', 'I158', 'I159']}
search_for_diagnosis = {"aneurysm": ["I671"], "sab": ["I600","I601","I602","I603","I604","I605","I606","I607","I608","I609"], 'DM': ['E100', 'E101', 'E102', 'E103', 'E104', 'E105', 'E106', 'E107', 'E108', 'E109', 'E110', 'E111', 'E112', 'E113', 'E114', 'E115', 'E116', 'E117', 'E118', 'E119', 'E120', 'E121', 'E122', 'E123', 'E124', 'E125', 'E126', 'E127', 'E128', 'E129', 'E130', 'E131', 'E132', 'E133', 'E134', 'E135', 'E136', 'E137', 'E138', 'E139', 'E140', 'E141', 'E142', 'E143', 'E144', 'E145', 'E146', 'E147', 'E148', 'E149'], 'Hypertension':  ['I10', 'I150', 'I151', 'I152', 'I158', 'I159'], 'Hypercholesterolemia': ['E780']}

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

#create text file with column names
#for file in data_list:
#    print("step 1")
#    pre.get_column_names(pre.read_basket(basket_location + file + ".csv", nrows = 1), save_path + "basket_" + file + "columns.txt", 1)
#    os.system("python3 UKBiobank_scraper.py " + save_path + "basket_" + file + "_columns.txt" + " " + save_path + "basket_" + file + "_meta_data.txt")

#read-in icd10 columns
print("step 2")
df_icd10 = pre.read_basket(basket_location + data_1 + ".csv", pre.select_column_names(["41270-0.0", "41280-0.0"], pre.get_column_names(pre.read_basket(basket_location + data_1 + ".csv", nrows = 1)), 1))

#create dictionary with diagnosis cases + diagnosis dates and add some other columns from basket_10036, replace aneurysm with control cases
print("step 3")
#if replace_aneurysm_control_group == False:
#    selected_cases_dict_list = pre.select_column_data(pre.read_basket(basket_location + data_2 + ".csv", columns = ["eid", "31-0.0", "34-0.0", "52-0.0", "53-0.0"]), {"sex": "31-0.0", "birth_year": "34-0.0", "birth_month": "52-0.0", "date_attending_assesment_centre": "53-0.0"}, pre.select_icd_10_diagnoses(df_icd10, search_for_diagnosis, 1, 1))
#else:
#    selected_cases_dict_list = pre.select_column_data(pre.read_basket(basket_location + data_2 + ".csv", columns = ["eid", "31-0.0", "34-0.0", "52-0.0", "53-0.0"]), {"sex": "31-0.0", "birth_year": "34-0.0", "birth_month": "52-0.0", "date_attending_assesment_centre": "53-0.0"}, pre.select_control_cases(df_icd10, pre.select_icd_10_diagnoses(df_icd10, search_for_diagnosis, 1, 1), [1, "aneurysm"], 1))

#pre.export_selected_cases_dict_list(selected_cases_dict_list, save_path + "export_selected_eids.csv", "comma_separated", ["eid", "diagnoses\*aneurysm", "diagnoses\*sab", "diagnoses\*control", "diagnoses\*DM", "diagnoses\*Hypertension", "missing"])

#vita selection. Select control group + export
print("step 3.2")
selected_cases_dict_list = pre.select_column_data(pre.read_basket(basket_location + data_2 + ".csv", columns = ["eid", "31-0.0", "34-0.0", "52-0.0"]), {"sex": "31-0.0", "birth_year": "34-0.0", "birth_month": "52-0.0"}, pre.select_control_cases(df_icd10, pre.select_icd_10_diagnoses(df_icd10, search_for_diagnosis, 1, 1), [0, ""], 1))
pre.export_selected_cases_dict_list(selected_cases_dict_list, save_path + "export_eids.csv", "comma_separated", ["eid", "diagnoses\*aneurysm", "diagnoses\*sab", "diagnoses\*DM", "diagnoses\*Hypertension", "diagnoses\*Hypercholesterolemia", "missing"])

#filter list of cases for occurence of selected diagnoses
print("step 4")
if replace_aneurysm_control_group == False:
    selected_cases_dict_list, filtered_list = pre.filter_diagnoses(selected_cases_dict_list, [{"sab": "*"}, {"aneurysm": "*"}])
else:
    selected_cases_dict_list, filtered_list = pre.filter_diagnoses(selected_cases_dict_list, [{"sab": "*"}, {"control": "*"}])

#print(pre.filter_diagnoses(selected_cases_dict_list, [{"test": 1}]) )
#print(pre.filter_diagnoses(selected_cases_dict_list, [{"aneurysm": 1, "sab": 1}]) )

#filter list of cases for occurence of selected diagnoses and visiting assesment center
print("step 4.2")
selected_cases_dict_list = pre.filter_case_date(selected_cases_dict_list, "date_attending_assesment_centre")

ane_c = 0
sab_c = 0
cases_list_ = []
filtered_cases_unique = 0

f = open("preprocessing/test_filtered_cases.txt", "w+")
for filtered_list_dict in filtered_list:
    diagnoses_list = []
    for diagnosis in filtered_list_dict['diagnoses']:
        for diagnosis_key in diagnosis:
            diagnoses_list.append(diagnosis_key)
    if len(set(diagnoses_list)) == 1:
        filtered_cases_unique += 1
    
    f.write(str(filtered_list_dict) + "\n")
f.close()
print("filtered_cases_unique:"+str(filtered_cases_unique))

f = open("preprocessing/test_cases.txt", "w+")
for llist in selected_cases_dict_list:
    cases_list_.append(llist['eid'])
    q_s = 0
    q_a = 0
    
    for diagnoses_llist_item in llist['diagnoses']:
        try:
            if q_a == 0:
                x_ty = diagnoses_llist_item['control']
                ane_c += 1
                q_a = 1
            #break
        except:
            pass
        try:
            if q_s == 0:
                x_ty = diagnoses_llist_item['sab']
                sab_c += 1
                q_s = 1
            #break
        except:
            pass
        
    f.write(str(llist) + "\n")
f.close()
print("ane:"+str(ane_c))
print("sab:"+str(sab_c))
print("length3:"+str(len(selected_cases_dict_list)))
print("length:"+str(len(cases_list_)))
print("length2:"+str(len(set(cases_list_))))
import collections
print([item for item, count in collections.Counter(cases_list_).items() if count > 1])
for llist in selected_cases_dict_list:
    if llist['eid'] in set(cases_list_):
        #print(llist)
        pass

#create subset of basket based on filtered data
print("step 5")
if create_subset_data == True:
    for file in data_list:
        #subset_data = pre.subset_data(basket_location + data_1 + ".csv", selected_cases_dict_list, filter_coumn_names = ['eid', '30000-0.0'], output_csv = save_path + "subset_data.csv", return_output = 2)
        #subset_data = pre.subset_data(data_file_path,  selected_cases_dict_list, output_csv = save_path + "subset_data.csv", return_output = 1)
    
        #save subsetted data dictionary to csv file
        subset_data = pre.subset_data(basket_location + file + ".csv",  selected_cases_dict_list, save_path + "basket_" + file + "_subset.csv", None, 0)

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
    
    for meta_data_dict in meta_data_dict_list:
        i += 1
        sup.printProgressBar(i, length, prefix = 'Progress:', suffix = 'Complete', length = 50)
        
        loop_selected_cases_dict_list = selected_cases_dict_list
        
        column = meta_data_dict['column']
        value_type = meta_data_dict['value_type']
        
        #filter diagnosis later than column debut
        loop_selected_cases_dict_list = pre.filter_column_date(loop_selected_cases_dict_list, column, meta_data_dict_list, 0)
        
        #create group for sab and aneurysm
        #loop_selected_cases_dict_list = pre.create_diagnoses_group(loop_selected_cases_dict_list)

        #combine selected cases and columns in dataframe and create group column ##pass function as argument user_defined_grouping to create manual grouping
        selected_cases_df = pre.get_grouped_data(save_path + "basket_" + file + "_subset.csv", [column], loop_selected_cases_dict_list, ["diagnosis_group"], 1)
        #selected_cases_df = pre.get_grouped_data(save_path + "basket_" + file + "_subset.csv", [column], loop_selected_cases_dict_list, ["diagnosis_group", "sex"], 1)
                
        if value_type.startswith("Continuous"):            
            #create statistics dictionary per column and calculate mean per group
            statistics_dict_list = sta.get_mean(selected_cases_df, statistics_dict_list, None)
            statistics_dict_list = sta.get_mean_difference_significance(selected_cases_df, statistics_dict_list, 999, None)
            
            all_selected_cases_df = pre.merge_dataframes(all_selected_cases_df, selected_cases_df, [column], 'eid', False, None)
                
        if value_type.startswith("Integer"):  
            statistics_dict_list = sta.get_mean(selected_cases_df, statistics_dict_list, [-1, -3, -4, -10])
            statistics_dict_list = sta.get_mean_difference_significance(selected_cases_df, statistics_dict_list, 999, [-1, -3, -4, -10])
            
            all_selected_cases_df = pre.merge_dataframes(all_selected_cases_df, selected_cases_df, [column], 'eid', False, [-1, -3, -4, -10])
            
        if value_type.startswith("Categorical"): 
            #pre.get_meta_data([meta_data_dict], "data_coding", column)[0]
            statistics_dict_list = sta.get_chi2(selected_cases_df, [meta_data_dict], statistics_dict_list, False, [-1, -3])
            statistics_dict_list = sta.get_kruskal_wallis(selected_cases_df, [meta_data_dict], statistics_dict_list, False, [-1, -3])
            #statistics_dict_list = sta.get_fisher_exact(selected_cases_df, [meta_data_dict], statistics_dict_list)
            statistics_dict_list = sta.add_to_statistics_dict(statistics_dict_list, column, "data_coding_types", {'data_coding_types': meta_data_dict['data_coding']['data_coding_types']}, 1, None)
            
            if sta.isin_ordinal_data_codings([meta_data_dict], column):
                binary_selected_cases_df = pre.make_categories_binary(selected_cases_df, [meta_data_dict], column, [-1, -3], True)
                all_selected_cases_df = pre.merge_dataframes(all_selected_cases_df, binary_selected_cases_df, [column], 'eid', True, [-1, -3])
            else:
                binary_selected_cases_df, added_columns = pre.create_dummies(selected_cases_df, [meta_data_dict], column, [-1, -3], False)
                all_selected_cases_df = pre.merge_dataframes(all_selected_cases_df, binary_selected_cases_df, added_columns, 'eid', True, [-1, -3])
        
        statistics_dict_list = sta.get_number_of_cases(selected_cases_df, statistics_dict_list)
        
        #add other data to dictionaries
        statistics_dict_list = sta.add_to_statistics_dict(statistics_dict_list, column, "description", {'description': meta_data_dict['description']}, 1, None)
        statistics_dict_list = sta.add_to_statistics_dict(statistics_dict_list, column, "value_type", {'value_type': meta_data_dict['value_type']}, 1, None)

#filter minimum number of cases and sort largest mean difference
print("step 7")
mean_statistics_dict_list = sta.sort_statistic_list(sta.filter_minimum_number_of_cases(statistics_dict_list, 10), "mean_diff_significance", False, "export_p")
chi_statistics_dict_list = sta.sort_statistic_list(sta.filter_minimum_number_of_cases(statistics_dict_list, 10), "chi2", False, "export_p")
kruskal_wallis_statistics_dict_list = sta.sort_statistic_list(sta.filter_minimum_number_of_cases(statistics_dict_list, 10), "kruskal_wallis", False, "export_p")
#converted_fisher_exact_statistics_dict_list = sta.sort_statistic_list(sta.filter_minimum_number_of_cases(sta.convert_statistic_to_columns(statistics_dict_list, "fisher_exact", "converted_fisher_exact"), 10), "converted_fisher_exact", False, True)

#remove columns that are not adding anything and do bonferroni correction
print("step 8")
mean_statistics_dict_list = sta.get_bonferroni(pre.remove_columns_dictionary(mean_statistics_dict_list, save_path + "remove_columns.txt"), 0.1, "export_p")
chi_statistics_dict_list = sta.get_bonferroni(pre.remove_columns_dictionary(chi_statistics_dict_list, save_path + "remove_columns.txt"), 0.1, "export_p")
kruskal_wallis_statistics_dict_list = sta.get_bonferroni(pre.remove_columns_dictionary(kruskal_wallis_statistics_dict_list, save_path + "remove_columns.txt"), 0.1, "export_p")

#export
print("step 9")
sta.export_statistics_dict_list(mean_statistics_dict_list, save_path + "statistics_dict_list_mean.tsv", "tab_separated", ["column", "description", "mean_diff_significance", "number_of_cases", "mean", "mean_diff", "bonferroni_corrected_p_value"])
sta.export_statistics_dict_list(chi_statistics_dict_list, save_path + "statistics_dict_list_chi2.tsv", "tab_separated", ["column", "description", "number_of_cases", "chi2\*c", "chi2\*p", "data_coding_types\*data_coding_types", "bonferroni_corrected_p_value"])
sta.export_statistics_dict_list(kruskal_wallis_statistics_dict_list, save_path + "statistics_dict_list_kruskal_wallis.tsv", "tab_separated", ["column", "description", "number_of_cases", "kruskal_wallis\*s", "kruskal_wallis\*p", "data_coding_types\*data_coding_types", "bonferroni_corrected_p_value"])
#sta.export_statistics_dict_list(converted_fisher_exact_statistics_dict_list, save_path + "statistics_dict_list_fisher_exact.tsv", "tab_separated", ["column", "description", "converted_fisher_exact\*number_of_cases", "converted_fisher_exact\*o", "converted_fisher_exact\*p", "converted_fisher_exact\*contigency_table", "data_coding_types\*data_coding_types", "bonferroni_corrected_p_value"])


#add all statistics and calculate bonferroni
print("step 10")
total_statistics_list = mean_statistics_dict_list + chi_statistics_dict_list + kruskal_wallis_statistics_dict_list
#print(len(total_statistics_list))
#remove_columns = ["41244-0.0", "41202-0.0", "41246-0.0", "40001-0.0", "41249-0.0", "41245-0.0", "40018-0.0", "41210-0.0", "41250-0.0", "19-0.0", "41230-0.0", "41200-0.0", "20003-0.0", "20013-0.0", "4095-0.0", "55-0.0", "4275-0.0", "41201-0.0", "41229-0.0", "20079-0.0", "20080-0.0", "40019-0.0", "4092-0.0", "22000-0.0", "2129-0.0", "20014-0.0", "41253-0.0", "4081-0.0", "4268-0.0", "4281-0.0", "40006-0.0", "4186-0.0", "20005-0.0", "41251-0.0", "111-0.0", "20152-0.0", "20012-0.0", "62-0.0", "21-0.0", "3081-0.0", "30004-0.0", "30014-0.0", "30024-0.0", "30034-0.0", "30044-0.0", "30054-0.0", "30064-0.0", "30074-0.0", "30084-0.0", "30094-0.0", "30104-0.0", "30114-0.0", "30124-0.0", "30134-0.0", "30144-0.0", "30154-0.0", "30164-0.0", "30174-0.0", "30184-0.0", "30194-0.0", "30204-0.0", "30214-0.0", "30224-0.0", "30234-0.0", "30244-0.0", "30254-0.0", "30264-0.0", "30274-0.0", "30284-0.0", "30294-0.0", "30304-0.0", "4924-0.0", "110005-0.0", "30503-0.0", "30513-0.0", "30523-0.0", "30533-0.0", "3082-0.0", "3140-0.0", "4093-0.0", "4096-0.0", "4287-0.0", "5182-0.0", "41231-0.0", "41247-0.0", "42013-0.0", "100260-0.0", "100460-0.0", "100890-0.0", "22004-0.0", "20082-0.0", "3137-0.0", "96-0.0", "20081-0.0", "20077-0.0", "6039-0.0", "20133-0.0", "189-0.0", "4288-0.0", "20075-0.0", "129-0.0", "130-0.0", "20074-0.0", "30001-0.0", "30011-0.0", "30021-0.0", "30031-0.0", "30041-0.0", "30051-0.0", "30061-0.0", "30071-0.0", "30081-0.0", "30091-0.0", "30101-0.0", "30111-0.0", "30121-0.0", "30131-0.0", "30141-0.0", "30151-0.0", "30161-0.0", "30171-0.0", "30181-0.0", "30191-0.0", "30201-0.0", "30211-0.0", "30221-0.0", "30231-0.0", "30241-0.0", "30251-0.0", "30261-0.0", "30271-0.0", "30281-0.0", "30291-0.0", "30301-0.0", "3659-0.0", "3700-0.0", "3786-0.0", "3809-0.0", "3872-0.0", "4250-0.0", "4253-0.0", "4254-0.0", "4255-0.0", "4256-0.0", "4260-0.0", "4283-0.0", "4285-0.0", "5985-0.0", "20083-0.0", "20129-0.0", "20130-0.0", "20248-0.0", "4259-0.0"]
#total_statistics_list = pre.remove_columns_dictionary(total_statistics_list, remove_columns)
#print(len(total_statistics_list))

#do bonferroni, sort and select p < 0.1/number_of_features then sort
#total_statistics_list = sta.get_bonferroni(total_statistics_list, "export_p")
total_statistics_list = sta.sort_statistic_list(total_statistics_list, "export_p", False, None)
sta.export_statistics_dict_list(total_statistics_list, save_path + "total_bonferroni_corrected_p_values.tsv", "tab_separated", ["column", "description", "value_type", "bonferroni_corrected_p_value", "export_p", "holm_p", "holm_p2"])

columns = []
with open(save_path + "remove_columns.txt") as f:
    for line in f:
        columns.append(line.strip())

filtered_dict_list = []
for total_statistics_dict in statistics_dict_list:
    if total_statistics_dict['column'] in columns:
        filtered_dict_list.append(total_statistics_dict)
sta.export_statistics_dict_list(filtered_dict_list, save_path + "filtered_columns.tsv", "tab_separated", ["column", "description"])

#make dataframe containing all cases filtered by debut date and significant columns
print("step 11")
if replace_aneurysm_control_group == False:
    total_statistics_list, all_selected_cases_df = sta.filter_statistic_columns(total_statistics_list, 0, ["export_p", 0.05], all_selected_cases_df)
else:
    keep_columns_list = ["2178-0.0", "904-0.0", "4653-0.0", "137-0.0", "100009-0.0", "2060-0.0", "1960-0.0", "100001-0.0", "1249-0.0", "6164-0.0", "20116-0.0", "6154-0.0", "6152-0.0", "30150-0.0", "874-0.0", "5463-0.0", "1873-0.0", "4548-0.0", "2453-0.0", "23100-0.0", "1568-0.0", "1727-0.0", "6177-0.0", "30090-0.0", "2070-0.0", "3606-0.0", "1528-0.0", "2277-0.0", "6150-0.0", "100017-0.0", "30080-0.0", "100019-0.0", "2335-0.0", "21001-0.0", "2754-0.0", "2020-0.0", "2080-0.0", "100014-0.0", "23104-0.0", "20018-0.0", "41235-0.0", "100015-0.0", "2473-0.0", "981-0.0", "699-0.0", "30160-0.0", "100025-0.0", "100011-0.0", "1090-0.0", "1299-0.0"]
    total_statistics_list, all_selected_cases_df = sta.filter_statistic_columns(total_statistics_list, 1, keep_columns_list, all_selected_cases_df)

#clean up dataframe, make figure and export all cases df to csv
all_selected_cases_df = pre.clean_dataframe(all_selected_cases_df)
#pre.missing_values_figure(all_selected_cases_df, save_path + "missing_cases.png")

all_selected_cases_df.to_csv(save_path + "all_selected_cases_df.csv", sep = ',')

#do imputation and select train test group
print("step 12")
all_selected_cases_df = log.imputate_data(all_selected_cases_df, ['eid', 'diagnosis_group'])
all_selected_cases_df.to_csv(save_path + "imputated_all_selected_cases_df.csv", sep=',')
#X_train, X_test, y_train, y_test = log.create_training_test_group(all_selected_cases_df, ['eid'], "diagnosis_group")
#model_dict = log.do_logistic_regression(X_train, X_test, y_train, y_test, False)

models_list, model_stats_df_list = log.multiple_do_logistic_regression(all_selected_cases_df, "bootstrap", 1000, ['eid'], "diagnosis_group")
model_stats_df_list[1].to_csv(save_path + "coef_df.csv")

last_row = model_stats_df_list[1].tail(1)
coef_df = last_row.T
coef_df = coef_df.reset_index()
coef_df.columns = ["Column", "coef"]

for file in data_list:
    meta_data_dict_list = pre.open_meta_data(save_path + "basket_" + file + "_meta_data.txt")
    
    for meta_data_dict in meta_data_dict_list:
        coef_df.loc[coef_df['Column'].str.startswith(meta_data_dict['column']), 'Description'] = meta_data_dict['description']
        
        if meta_data_dict['value_type'].startswith("Categorical"):
            try:
                coef_df.loc[coef_df['Column'].str.startswith(meta_data_dict['column']), 'data_coding'] = str(meta_data_dict['data_coding']['data_coding_types'])
            except:
                print(meta_data_dict['column'])
                
coef_df.to_csv(save_path + "coef_df_summary.csv", index = False)

model_stats_df_list[0].to_csv(save_path + "coef_df.csv", index = False)
model_stats_df_list[1].to_csv(save_path + "coef_mean_df.csv", index = False)
model_stats_df_list[2].to_csv(save_path + "auc_df.csv", index = False)

print(model_stats_df_list[1])
print(models_list[-1]['classes'])

print("Runtime: " + str(datetime.datetime.now() - begin_time))
#save_test