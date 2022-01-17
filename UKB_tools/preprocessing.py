import pandas as pd
from csv import reader
from itertools import compress
import csv
import json
import sys

#read in a dataframe from a csv file
def read_basket(file_path: str, columns: list = None, nrows: int = None) -> pd.core.frame.DataFrame:
    df = pd.read_csv(file_path, usecols = columns, nrows = nrows, dtype = str)
    
    return df

#open a meta data file, returns a list of dictionaries
def open_meta_data(file_path: str) -> list:
    instance_date_dict_list = []
    with open(file_path) as f:
        for jsonObj in f:
            instance_date_dict = json.loads(jsonObj)
            instance_date_dict_list.append(instance_date_dict)
    return instance_date_dict_list  

#get meta data of a column >>>> not used???
def get_meta_data(meta_data: list, meta_data_type: str, column: int) -> list:
    for meta_data_dict in meta_data:
        if column == meta_data_dict['column']:
            try:
                meta_data = [meta_data_dict[meta_data_type]]
            except:
                meta_data = [None]
            
    return meta_data

#remove instance from meta data dictionaries list
def filter_instance(meta_data_dict_list: list, instance: str) -> list:
    new_meta_data_dict_list = []
    
    for meta_data_dict in meta_data_dict_list:
        if meta_data_dict['column'].split("-")[1].startswith(instance):
            new_meta_data_dict_list.append(meta_data_dict)
    
    return new_meta_data_dict_list

#create a subset of a dataframe based on the selected cases dictionaries. Leave filter column names open to export all columns. Can also return the dataframe
def subset_data(input_scv: str, selected_cases_dict_list: list = None, output_csv: str = None, filter_coumn_names: list = None, return_output: int = 0, read_type = 0):
    read_column_name = True
    subset_data = []
    import datetime
    
    if output_csv != None:
        f = open(output_csv, "w")
        f.close
    
    selected_eid_list = []
    for selected_cases_dict in selected_cases_dict_list:
        selected_eid_list.append(selected_cases_dict['eid'])
        
    if read_type == 0:
        begin_time = datetime.datetime.now()
        with open(input_scv, 'r') as read_obj:
            csv_reader = reader(read_obj)
            
            for row in csv_reader:
                if read_column_name == True:
                    columns_names = row
                    eid_index_list = [column_name in ["eid"] for column_name in columns_names]
                                    
                    if filter_coumn_names != None:
                        filter_columns_list = [column_name in filter_coumn_names for column_name in columns_names]
                    else:
                        filter_columns_list = columns_names
                    
                    row = list(compress(row, filter_columns_list))
                    if output_csv != None:
                        with open(output_csv, 'a') as f:
                            writer = csv.writer(f)
                            writer.writerow(row)
                    
                    read_column_name = False
                    
                else:
                    eid = list(compress(row, eid_index_list))[0]
                    
                    if selected_cases_dict_list != None:
                        new_row = []

                        if eid in selected_eid_list:
                            new_row = row
                            
                        row = new_row
                    
                    if filter_coumn_names != None:
                        row = list(compress(row, filter_columns_list))
                        
                    if row:
                        if output_csv != None:
                            with open(output_csv, 'a') as f:
                                writer = csv.writer(f)
                                writer.writerow(row)
                        
                        if return_output != 0:
                            subset_data.append(row)
        print("Runtime_subset_0: " + str(datetime.datetime.now() - begin_time))
        begin_time = datetime.datetime.now()
                            
        begin_time = datetime.datetime.now()
        if return_output == 2:
            subset_data = pd.DataFrame.from_records(subset_data, columns = filter_coumn_names) 
        print("Runtime_subset_1: " + str(datetime.datetime.now() - begin_time))
        begin_time = datetime.datetime.now()
    
    
    elif read_type == 1:
        if filter_coumn_names != None:
            if "eid" not in filter_coumn_names:
                filter_coumn_names.insert(0, "eid")
        
        begin_time = datetime.datetime.now()
        subset_data = pd.read_csv(input_scv, usecols = filter_coumn_names, dtype = str)
        print("Runtime_subset_2: " + str(datetime.datetime.now() - begin_time))
        begin_time = datetime.datetime.now()
        subset_data = subset_data[subset_data['eid'].isin(selected_eid_list)]
        print("Runtime_subset_3: " + str(datetime.datetime.now() - begin_time))
        begin_time = datetime.datetime.now()
        
        if output_csv != None:
            subset_data.to_csv(output_csv)
        print("Runtime_subset_4: " + str(datetime.datetime.now() - begin_time))
    
    return subset_data

#get column names of dataframe, can save them to a file                  
def get_column_names(basket_df: pd.core.frame.DataFrame, output: str = None, omit_eid: int = 0) -> list:
    columns = list(basket_df.columns)
    
    if omit_eid == 1:
        columns.remove("eid")
    
    if output != None:
        f = open(output, "w+")
        for col in columns:
            f.write(col + "\n")
        f.close()
        
    return columns

#select all column instance names that belong to a column e.g. 1-0, 1-1, 1-2...
def select_column_names(select_column: list, columns: list, add_eid: int = 0) -> list:
    select_columns = []
    for select in select_column:
        select = select.split("-")[0]
        
        for column in columns:
            if column.startswith(select):
                select_columns.append(column)
    
    if add_eid == 1:
        select_columns.append("eid")
        
    return select_columns

#Select all occurences of inserted icd10 diagnoses in dataframe en add to dictionaries. Can also add diagnoses dates
def select_icd_10_diagnoses(df: pd.core.frame.DataFrame, select_diagnoses_list: dict, add_diagnoses_date: int = 0, add_diagnoses_type: int = 0) -> list:
    from UKB_tools.support import printProgressBar
    
    total_list = []
    
    for key in select_diagnoses_list:
        total_list.extend(select_diagnoses_list[key])
        
    diagnoses_df = df[df.isin(total_list).any(axis=1)]
    
    i = 0
    length = len(df)
    printProgressBar(i, length, prefix = 'Progress:', suffix = 'Complete', length = 50)
    
    selected_cases_dict_list = []
    for index, row in diagnoses_df.iterrows():
        diagnoses_row_dict_list = []
        eid = row['eid']
        i += 1
        
        printProgressBar(i, length, prefix = 'Progress:', suffix = 'Complete', length = 50)
        
        if add_diagnoses_date == 1 or add_diagnoses_type == 1:
            for series_index, value in row.items():
                for key in select_diagnoses_list:
                    if value in select_diagnoses_list[key] and series_index.startswith("41270-") :
                        diagnosis_date = ""
                        
                        if add_diagnoses_date == 1:
                            column = series_index
                            column_sub = column.split("-")[1]
                            dates_column = "41280-" + column_sub
                            diagnosis_date = pd.to_datetime(row[dates_column])
                        
                        diagnoses_row_dict_list.append({key: diagnosis_date})

        selected_cases_dict_list.append({"eid": eid, "diagnoses": diagnoses_row_dict_list})
                                
    return selected_cases_dict_list

#Select control cases based on selected diagnosis selection
def select_control_cases(df: pd.core.frame.DataFrame, selected_cases_dict_list: list = [], control_type: list = [0, ""], add_control_diagnosis: int = 0) -> list:
    #control_type [type, argument]: [0, include all eids that are not in the selected cases dict] [1, replace a diagnosis with a control group] [2, combine given diagnoses and add controls] [3, add number of control, add 2x* two times as many controls as in the input, use 2xdiagnosis for a multiplication compared to diagnosis ] [4, add all other than diagnosis to control]
    import random
    from UKB_tools.support import printProgressBar
    
    new_selected_cases_dict_list = []
    eid_list = df['eid'].values.tolist()
    cases_eids = []
    i = 0
    length = int
    
    def select_control_cases_add_control_diagnose():
        if add_control_diagnosis == 0:
            new_selected_cases_dict_list.append(new_dict)
        else:
            new_dict['diagnoses'] = [{'control': pd.to_datetime("2100-01-01")}]
            new_selected_cases_dict_list.append(new_dict)
    
    def select_control_cases_progressbar(length_item, init = False):
        global i
        global length
        
        if init == True:
            i = 0
            length = len(length_item)
            printProgressBar(i, length, prefix = 'Progress:', suffix = 'Complete', length = 50)
        else:
            printProgressBar(i, length, prefix = 'Progress:', suffix = 'Complete', length = 50)
            i += 1
        
    for selected_cases_dict in selected_cases_dict_list:
        cases_eids.append(selected_cases_dict['eid'])
        
    if control_type[0] == 0:    
        select_control_cases_progressbar(eid_list, init = True)
        
        for eid in eid_list:
            select_control_cases_progressbar(eid_list, init = False)
                              
            if eid not in cases_eids:
                new_dict = {'eid': eid, 'control': "yes"} #possibly add the eid of the case to which control is coupled in 'control':
                select_control_cases_add_control_diagnose()
            else:
                for selected_cases_dict in selected_cases_dict_list:
                    if selected_cases_dict['eid'] == eid:
                        new_selected_cases_dict_list.append(selected_cases_dict)
                        
    elif control_type[0] == 1:
        select_control_cases_progressbar(selected_cases_dict_list, init = True)
        
        for eid in eid_list.copy():
            if eid in cases_eids:
                eid_list.remove(eid)
    
        for selected_cases_dict in selected_cases_dict_list:
            diagnosis_list = []
            select_control_cases_progressbar(selected_cases_dict_list, init = False)
            
            for diagnoses in selected_cases_dict['diagnoses']:
                for diagnosis in diagnoses:
                    diagnosis_list.append(diagnosis)
                                         
            if control_type[1] in diagnosis_list:
                control_eid = eid_list[random.randint(0, len(eid_list) - 1)]
                new_dict = {'eid': control_eid, 'control': "yes"}
                eid_list.remove(new_dict['eid'])
                cases_eids.append(new_dict['eid'])
                select_control_cases_add_control_diagnose()
            else:
                new_selected_cases_dict_list.append(selected_cases_dict)
            
    elif control_type[0] == 3:
        select_control_cases_progressbar(range(int(control_type[1].split("x")[0])), init = True)
        
        for eid in eid_list.copy():
            if eid in cases_eids:
                eid_list.remove(eid)
                
        for selected_cases_dict in selected_cases_dict_list:
            new_selected_cases_dict_list.append(selected_cases_dict)
            
            diagnosis_list = []
            
        if "x" in control_type[1]:
            if control_type[1].split("x")[1] == "*":
                number_of_controls = int(control_type[1].replace("x", "").replace("*", "")) * len(selected_cases_dict_list)
            else:
                number_of_cases = 0
                search_diagnosis = control_type[1].split("x")[1]
                
                for selected_cases_dict in selected_cases_dict_list:
                    diagnosis_list = []
                    
                    for diagnoses in selected_cases_dict['diagnoses']:
                        for diagnosis in diagnoses:
                            diagnosis_list.append(diagnosis)
                            
                    if search_diagnosis in diagnosis_list:
                        number_of_cases += 1
                        
                number_of_controls = int(control_type[1].replace("x", "").replace("*", "")) * number_of_cases
                
        else:
            number_of_controls = int(control_type[1])
            
        for j in range(number_of_controls):
            select_control_cases_progressbar(range(number_of_controls), init = False)
            
            control_eid = eid_list[random.randint(0, len(eid_list) - 1)]
            new_dict = {'eid': control_eid, 'control': "yes"}
            eid_list.remove(new_dict['eid'])
            cases_eids.append(new_dict['eid'])
            select_control_cases_add_control_diagnose()
            
    elif control_type[0] == 4:
        for eid in eid_list.copy():
            if eid in cases_eids:
                eid_list.remove(eid)
        
        for selected_cases_dict in selected_cases_dict_list:
            new_selected_cases_dict_list.append(selected_cases_dict)
        
        select_control_cases_progressbar(range(len(eid_list)), init = True)
        
        for j in range(len(eid_list)):
            select_control_cases_progressbar(range(len(eid_list)), init = False)
            
            control_eid = eid_list[j]
            new_dict = {'eid': control_eid, 'control': "yes"}
            cases_eids.append(new_dict['eid'])
            select_control_cases_add_control_diagnose()
                
    return new_selected_cases_dict_list

def combine_diagnoses(selected_cases_dict_list: list = [], combine_diagnoses: list = []) -> list:
    new_selected_cases_dict_list = []
    
    for selected_cases_dict in selected_cases_dict_list:
        
        for comb_diagnosis in combine_diagnoses:
            for diagnoses in selected_cases_dict['diagnoses']:
                if comb_diagnosis in diagnoses:
                    diagnoses["combined_diagnoses"] = diagnoses.pop(comb_diagnosis)
                elif comb_diagnosis == "*":
                    for diagnosis in diagnoses.copy():
                        diagnoses["combined_diagnoses"] = diagnoses.pop(diagnosis)
                    
        new_selected_cases_dict_list.append(selected_cases_dict)
        
    return new_selected_cases_dict_list

#export selected cases dict list to a file
def export_selected_cases_dict_list(selected_cases_dict_list: list, save_path: str, save_type: str = "json", fields: list = None):
    header = True
    f = open(save_path, "w+")
    f.close()
    
    for selected_case_dict in selected_cases_dict_list:
        if save_type == "json":
            pass
        elif save_type == "comma_separated":
            print_object = ""
            
            if header == True:
                for field in fields:
                    print_object = print_object + str(field) + ","
                    
                print_object = print_object + "\n"
                header = False
                
            for field in fields:             
                if "\*" in field:
                    diagnosis_date = None
                    field_split = field.split("\*")
                    
                    for diagnosis_dict in selected_case_dict['diagnoses']:
                        for diagnosis_key in diagnosis_dict:
                           if diagnosis_key == field_split[1]:
                               if diagnosis_date == None:
                                   diagnosis_date = diagnosis_dict[diagnosis_key]
                               elif diagnosis_date > diagnosis_dict[diagnosis_key]:
                                   diagnosis_date = diagnosis_dict[diagnosis_key]
                    
                    if diagnosis_date == pd.to_datetime("1900-01-01"):
                        diagnosis_date == None
                        
                    if diagnosis_date == None:
                        diagnosis_date = ""
                        
                    r = diagnosis_date

                else:
                    r = selected_case_dict[field]
                
                print_object = print_object + str(r) + ","
        
        f = open(save_path, "a")
        f.write(print_object)
        f.write("\n")
        f.close()
                  
#add data from specific columns to the selected cases dictionaries. Data is added as dictionaries
def select_column_data(df: pd.core.frame.DataFrame, select_columns: dict, selected_cases_dict_list: list = None) -> list:
    from UKB_tools.support import printProgressBar
    
    if selected_cases_dict_list == None:
        selected_cases_dict_list = []
        for eid in df['eid']:
            selected_cases_dict_list.append({"eid": eid})
    
    new_selected_cases_dict_list = []
    i = 0
    length = len(selected_cases_dict_list)
    printProgressBar(i, length, prefix = 'Progress:', suffix = 'Complete', length = 50)
    
    for dictionary in selected_cases_dict_list:
        i += 1
        printProgressBar(i, length, prefix = 'Progress:', suffix = 'Complete', length = 50)
        
        for key in select_columns:
            try:
                dictionary[key] = df.loc[(df["eid"] == dictionary['eid']), select_columns[key]].values[0]
                
                if not 'missing' in dictionary:
                    dictionary['missing'] = ""
            except:
                dictionary[key] = ""
                
                if not 'missing' in dictionary:
                    dictionary['missing'] = ""
                    dictionary["missing"] = str(dictionary['missing']) + str(key)
                else:
                    dictionary["missing"] = str(dictionary['missing']) + "_" + str(key)
                
        new_selected_cases_dict_list.append(dictionary)
     
    return(new_selected_cases_dict_list)

def filter_selected_cases(selected_cases_dict_list: list, filter_type: str, filter_value: int):
    new_selected_cases_dict_list = []
    
    for selected_case_dict in selected_cases_dict_list:
        if selected_case_dict[filter_type] == filter_value:
            new_selected_cases_dict_list.append(selected_case_dict)
            
    return new_selected_cases_dict_list

#filter from selected cases the combinations of diagnoses that are wanted in cases. Inserted as a list of dictionaries: [{"sab": "*"}, {"aneurysm": "*"}] for sab or aneurysm in one case, or [{"aneurysm": 1, "sab": 1}] for both 1 sab and 1 aneurysm in one case, * indicates any amount of diagnoses
def filter_diagnoses(selected_cases_dict_list: list, filter_diagnoses_list: list) -> list:
    new_selected_cases_dict_list = []
    filtered_list = []
    
    for case_dict in selected_cases_dict_list:
        diagnoses_list = []
        
        for diagnosis in case_dict['diagnoses']:
            for diagnosis_key in diagnosis:
                diagnoses_list.append(diagnosis_key)
        
        for filter_dict in filter_diagnoses_list:
            filter_case = False
            filter_list = []
            
            for filter_key in filter_dict:
                filter_list.append(filter_key)
                
                if diagnoses_list.count(filter_key) != filter_dict[filter_key] and filter_dict[filter_key] != "*":
                    filter_case = True
                               
            for diagnosis_un in set(diagnoses_list):
                if diagnosis_un not in filter_list:
                    filter_case = True
            
            if filter_case == False:
                new_selected_cases_dict_list.append(case_dict)
            else:
                filtered_list.append(case_dict)
    
    new_filtered_list = []
    for il in filtered_list:
        remove = False
        for el in new_selected_cases_dict_list:
            if il['eid'] == el['eid']:
                remove = True
        if remove == False:
            new_filtered_list.append(il)
    
    return new_selected_cases_dict_list, new_filtered_list

#remove cases based on coding type
def filter_coding_type(df: pd.core.frame.DataFrame, filter_coding_type_list: list, column: str, list_as_string: bool = False) -> pd.core.frame.DataFrame:
    if list_as_string == True:
        new_filter_coding_type_list = []
        
        for l in filter_coding_type_list:
            if not isinstance(l, str):
                new_filter_coding_type_list.append(str(l))
            else:
                new_filter_coding_type_list.append(l)
                
        filter_coding_type_list = new_filter_coding_type_list
        
    for filter_coding_type in filter_coding_type_list:
        df.drop(df[df[column] == filter_coding_type].index, inplace=True)
        
    return df

#filter diagnosis for date field
def filter_case_date(selected_cases_dict_list: list, date_field: str) -> list:
    new_selected_cases_dict_list = []
    
    for selected_case in selected_cases_dict_list:
        filter_date = pd.to_datetime(selected_case[date_field])
        diagnosis_date = pd.to_datetime("2100-01-01")
        
        for diagnosis_dict in selected_case['diagnoses']:
            for diagnosis_key in diagnosis_dict:
                if diagnosis_date > diagnosis_dict[diagnosis_key]:
                    diagnosis_date = diagnosis_dict[diagnosis_key]
                    
        if diagnosis_date > filter_date:
            new_selected_cases_dict_list.append(selected_case)
            
    return new_selected_cases_dict_list

#filters diagnoses dates. Type 0 filters cases diagnosed after debut date and type 2 filters cases diagnosed after version date
def filter_column_date(selected_cases_dict_list: list, column_name: str, column_meta_data: list, date_type: int = 0) -> list:
    new_selected_cases_dict_list = []
    
    if date_type == 0:
        date_type = "debut_date"
    elif date_type == 1:
        date_type = "version_date"
    
    for column_dict in column_meta_data:
        if column_dict['column'] == column_name:
            column_date = pd.to_datetime(column_dict['dates'][date_type])
    
    for selected_case in selected_cases_dict_list:
        diagnosis_date = pd.to_datetime("2100-01-01")
        
        for diagnosis_dict in selected_case['diagnoses']:
            for diagnosis_key in diagnosis_dict:
                if diagnosis_date > diagnosis_dict[diagnosis_key]:
                    diagnosis_date = diagnosis_dict[diagnosis_key]
        
        if diagnosis_date > column_date:
            new_selected_cases_dict_list.append(selected_case)
            
            
    return new_selected_cases_dict_list

#creates a diagnoses group for selected cases. Concatenates unique diagnoses
def create_diagnoses_group(selected_cases_dict_list: list) -> list:
    new_selected_cases_dict_list = []
    
    for selected_case_dict in selected_cases_dict_list:
        diagnoses_list = []
        for diagnosis_dict in selected_case_dict['diagnoses']:
            for key in diagnosis_dict:
                diagnoses_list.append(key)
        
        diagnosis_group = ""
        for unique_diagnosis in sorted(set(diagnoses_list)):
            diagnosis_group = diagnosis_group + unique_diagnosis
            
        selected_case_dict['diagnosis_group'] = diagnosis_group
        new_selected_cases_dict_list.append(selected_case_dict)
        
    return new_selected_cases_dict_list

def sort_selected_cases_dict_list(file_path: str, selected_cases_dict_list: list) -> list:
    new_sort_selected_cases_dict_list = []
    
    subset_data_df = subset_data(input_scv = file_path, selected_cases_dict_list = selected_cases_dict_list, filter_coumn_names = ['eid'], return_output = 2, read_type = 1)
    eid_list = subset_data_df['eid'].tolist()
        
    for eid in eid_list:
        i = 0
        
        for selected_case_dict in selected_cases_dict_list:
            if selected_case_dict['eid'] == eid:
                new_sort_selected_cases_dict_list.append(selected_case_dict)
                break
                
            i += 1

    return new_sort_selected_cases_dict_list

#adds groups to dataframe based on selected cases dictionaries
def get_grouped_data(file_path: str, columns: list, selected_cases_dict_list: list, group_list: list = None, omit_empty_column: int = 0, user_defined_grouping: callable = None) -> pd.core.frame.DataFrame:
    from UKB_tools.statistics import remove_nan
    import datetime
    begin_time = datetime.datetime.now()
    if "eid" not in columns:
        columns.insert(0, "eid")
    print("Runtime_grouped_0: " + str(datetime.datetime.now() - begin_time))
    begin_time = datetime.datetime.now()
        
    subset_data_df = subset_data(input_scv = file_path, selected_cases_dict_list = selected_cases_dict_list, filter_coumn_names = columns, return_output = 2, read_type = 1)
    print("Runtime_grouped_1: " + str(datetime.datetime.now() - begin_time))
    begin_time = datetime.datetime.now()
    
    if group_list != None:
        eid_list = subset_data_df['eid'].tolist()
        
        eid_group_list = []
                    
        i = 0 
        for selected_eid in eid_list:            
            eid_group_list.append(select_group(selected_cases_dict_list[i], group_list, user_defined_grouping))
            
            i += 1
                
        subset_data_df['group'] = eid_group_list
        
        print("Runtime_grouped_2: " + str(datetime.datetime.now() - begin_time))
        begin_time = datetime.datetime.now()
            
    if omit_empty_column == 1:
        for column in subset_data_df:
            if column != "eid" and column != "group":
                subset_data_df = remove_nan(subset_data_df, column)
                empty_cols = [col for col in subset_data_df.columns if subset_data_df[col].isnull().all()]
                subset_data_df.drop(empty_cols, axis = 1, inplace = True)
    print("Runtime_grouped_3: " + str(datetime.datetime.now() - begin_time))
    begin_time = datetime.datetime.now()
    
    return subset_data_df

#create the group names, can accept a user defined grouping function
def select_group(selected_case_dict: dict, group_list: list, user_defined_grouping: callable = None) -> str:
    if user_defined_grouping == None:
        selected_group = ""
        
        for group in group_list:
            selected_group = selected_group + str(selected_case_dict[group])
    else:
        selected_group = user_defined_grouping()
    
    return selected_group

def get_filter_columns(columns_path: str, skip_columns: list = None) -> list:
    filter_columns = []
    
    with open(columns_path) as f:
        for line in f:
            if skip_columns == None:
                filter_columns.append(line.strip())
            elif line.strip() not in skip_columns:
                filter_columns.append(line.strip())
            
    return filter_columns

#remove columns from list dictionary list
def remove_columns_dictionary(dictionary_list: list, columns_path: str, skip_columns: list = None) -> list:
    new_dictionary_list = []
    
    columns = get_filter_columns(columns_path, skip_columns)
    
    if type(dictionary_list[0]) == dict:
        for dictionary in dictionary_list:
            if not dictionary['column'] in columns:
                new_dictionary_list.append(dictionary)
    elif type(dictionary_list[0]) == str:
        for column in dictionary_list:
            if column not in columns:
                new_dictionary_list.append(column)
            
    return new_dictionary_list

#drop _l and _r columns and drop empty rows
#def clean_dataframe(input_df: pd.core.frame.DataFrame, replace_column_names: list = None) -> pd.core.frame.DataFrame:
def clean_dataframe(input_df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:    
    copy_input_df = input_df.copy()
    
    for col in input_df:
        new_colname = ""
        
        if col.endswith("_l"):
            colname_length = len(col.split("_"))
            step = 0
            
            while colname_length > 1:
                if step > 0:
                    new_colname = new_colname + "_"
                    
                new_colname = new_colname + col.split("_")[step]
                colname_length -= 1
                step += 1

            if len(new_colname) > 0:    
                copy_input_df.rename(columns = {col: new_colname}, inplace = True)

        elif col.endswith("_r"):
            copy_input_df.drop(columns = [col], axis = 0, inplace = True)
            
        else:
            pass
            
            if len(new_colname) > 0:
                copy_input_df.rename(columns = {col: new_colname}, inplace = True)
    
    rows_count = copy_input_df.count(axis=1).tolist()
    columns_count = copy_input_df.count(axis=0).tolist()
    
    i = 0
    for row_count in rows_count:
        if row_count == 0:
            copy_input_df = copy_input_df.drop(index = i)
        i += 1
    
    n_required_rows = len(copy_input_df) * 0.1
    i = 0
    for col in copy_input_df.columns:
        if columns_count[i] < n_required_rows:
            copy_input_df.drop(col, axis = 1, inplace = True)
        i += 1
                    
    return(copy_input_df)
    
def missing_values_figure(input_df: pd.core.frame.DataFrame, save_path: str):
    import missingno as msno
    
    msno.matrix(input_df, figsize=(10, 6)).figure.savefig(save_path)

#merge column of dataframe to another dataframes on column
def merge_dataframes(output_df: pd.core.frame.DataFrame, merge_df: pd.core.frame.DataFrame, column_list: list, on_column: str, categorical: bool = False, filter_coding_types: list = None) -> pd.core.frame.DataFrame:
    if len(merge_df) > 0:   
        for column in column_list:    
            if filter_coding_types != None:
                if categorical:
                    merge_df = merge_df.astype({column: 'category'})
                    merge_df = filter_coding_type(merge_df, filter_coding_types, column, True)
                    
                else:
                    merge_df[column] = pd.to_numeric(merge_df[column])
                    merge_df = filter_coding_type(merge_df, filter_coding_types, column, False)

            output_df = output_df.merge(merge_df[[on_column, column]] , on = (on_column), suffixes = ('_l', '_r'), how = "left")
    
    return output_df

#makes categorical dataframe column binary depending on data codings in meta data
def make_categories_binary(input_df: pd.core.frame.DataFrame, meta_data: list, column: str, remove_data_codings: list = [], use_ordinal_data: bool = True) -> pd.core.frame.DataFrame:
    import math
    from UKB_tools.statistics import isin_ordinal_data_codings
    
    if column in input_df:
        if (use_ordinal_data == True and isin_ordinal_data_codings(meta_data, column)) or use_ordinal_data == False:
            for meta_data_dict in meta_data:
                if meta_data_dict['column'] == column:
                    column_data_coding = meta_data_dict['data_coding']['data_coding_types']
                    
                    for remove_data_coding in remove_data_codings:
                        if remove_data_coding in column_data_coding:
                            del column_data_coding[remove_data_coding]
                            
                    data_codings_list = list(column_data_coding.keys())
                    data_codings_list = sorted([int(i) for i in data_codings_list])
                    split_index = math.ceil(len(data_codings_list) / 2) - 1
                    
                    for data_coding in data_codings_list:
                        if data_codings_list.index(data_coding) <= split_index:
                            input_df.loc[input_df[column] == str(data_coding), column] = "0"
                        else:
                            input_df.loc[input_df[column] == str(data_coding), column] = "1"
                                
    return input_df

def create_dummies(input_df: pd.core.frame.DataFrame, meta_data: list, column: str, remove_data_codings: list = [], use_ordinal_data: bool = False) -> [pd.core.frame.DataFrame, list]:
    from UKB_tools.statistics import isin_ordinal_data_codings
    import numpy as np
    
    new_columns_list = []
    
    if column in input_df:
        if (use_ordinal_data == True and isin_ordinal_data_codings(meta_data, column)) or use_ordinal_data == False:
            input_df = filter_coding_type(input_df, remove_data_codings, column, True)
            
            dummies_df = pd.get_dummies(input_df[column], prefix = column, dummy_na = True) #drop_first=True
                                
            for col in dummies_df:
                if col.startswith(column) and not col.endswith('_nan'):
                    new_columns_list.append(col)
                    
                    dummies_df[col] = np.where(dummies_df[column+"_nan"] == 1, None, dummies_df[col])
                    input_df[col] = dummies_df[col]
                    
            input_df = input_df.drop([column], axis=1)
            
    return input_df, new_columns_list
