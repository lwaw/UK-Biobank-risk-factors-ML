import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import sys
from sklearn import metrics
import statistics

t_times = 0.001
t_range = 1001

#select significant features description
filter_columns_list = ["30001-0.0", "30004-0.0", "30011-0.0", "30014-0.0", "30021-0.0", "30024-0.0", "30030-0.0", "30031-0.0", "30034-0.0", "30041-0.0", "30044-0.0", "30051-0.0", "30054-0.0", "30061-0.0", "30064-0.0", "30071-0.0", "30074-0.0", "30081-0.0", "30084-0.0", "30091-0.0", "30094-0.0", "30101-0.0", "30104-0.0", "30111-0.0", "30114-0.0", "30121-0.0", "30124-0.0", "30131-0.0", "30134-0.0", "30141-0.0", "30144-0.0", "30151-0.0", "30154-0.0", "30161-0.0", "30164-0.0", "30171-0.0", "30174-0.0", "30180-0.0", "30181-0.0", "30184-0.0", "30190-0.0", "30191-0.0", "30194-0.0", "30200-0.0", "30201-0.0", "30204-0.0", "30210-0.0", "30211-0.0", "30214-0.0", "30220-0.0", "30221-0.0", "30224-0.0", "30230-0.0", "30231-0.0", "30234-0.0", "30240-0.0", "30241-0.0", "30244-0.0", "30251-0.0", "30254-0.0", "30261-0.0", "30264-0.0", "30271-0.0", "30274-0.0", "30281-0.0", "30284-0.0", "30290-0.0", "30291-0.0", "30294-0.0", "30301-0.0", "30304-0.0", "30503-0.0", "30513-0.0", "30523-0.0", "30533-0.0", "41200-0.0", "41210-0.0", "19-0.0", "21-0.0", "23-0.0", "51-0.0", "54-0.0", "55-0.0", "62-0.0", "77-0.0", "78-0.0", "96-0.0", "111-0.0", "129-0.0", "130-0.0", "189-0.0", "401-0.0", "402-0.0", "403-0.0", "404-0.0", "2129-0.0", "2684-0.0", "2704-0.0", "3061-0.0", "3065-0.0", "3081-0.0", "3082-0.0", "3083-0.0", "3084-0.0", "3085-0.0", "3086-0.0", "3137-0.0", "3140-0.0", "3144-0.0", "3146-0.0", "3147-0.0", "3659-0.0", "3700-0.0", "3786-0.0", "3809-0.0", "3872-0.0", "4081-0.0", "4092-0.0", "4093-0.0", "4095-0.0", "4096-0.0", "4101-0.0", "4103-0.0", "4104-0.0", "4105-0.0", "4106-0.0", "4120-0.0", "4122-0.0", "4123-0.0", "4124-0.0", "4125-0.0", "4186-0.0", "4200-0.0", "4250-0.0", "4253-0.0", "4254-0.0", "4255-0.0", "4256-0.0", "4259-0.0", "4260-0.0", "4268-0.0", "4269-0.0", "4270-0.0", "4272-0.0", "4275-0.0", "4276-0.0", "4277-0.0", "4279-0.0", "4281-0.0", "4283-0.0", "4285-0.0", "4287-0.0", "4288-0.0", "4290-0.0", "4291-0.0", "4292-0.0", "4293-0.0", "4294-0.0", "4849-0.0", "4924-0.0", "4935-0.0", "4946-0.0", "4957-0.0", "4968-0.0", "4979-0.0", "4990-0.0", "5001-0.0", "5012-0.0", "5556-0.0", "5985-0.0", "6020-0.0", "6038-0.0", "6039-0.0", "20003-0.0", "20004-0.0", "20005-0.0", "20012-0.0", "20013-0.0", "20014-0.0", "20023-0.0", "20074-0.0", "20075-0.0", "20077-0.0", "20079-0.0", "20080-0.0", "20081-0.0", "20082-0.0", "20083-0.0", "20129-0.0", "20130-0.0", "20133-0.0", "20152-0.0", "20154-0.0", "20165-0.0", "20167-0.0", "20169-0.0", "20171-0.0", "20173-0.0", "20175-0.0", "20177-0.0", "20179-0.0", "20248-0.0", "21003-0.0", "22000-0.0", "22003-0.0", "22004-0.0", "22005-0.0", "23099-0.0", "23111-0.0", "23115-0.0", "23119-0.0", "23123-0.0", "23127-0.0", "24020-0.0", "24021-0.0", "24022-0.0", "24023-0.0", "40001-0.0", "40006-0.0", "40018-0.0", "40019-0.0", "41201-0.0", "41202-0.0", "41204-0.0", "41229-0.0", "41230-0.0", "41231-0.0", "41244-0.0", "41245-0.0", "41246-0.0", "41247-0.0", "41249-0.0", "41250-0.0", "41251-0.0", "41253-0.0", "42013-0.0", "100260-0.0", "100460-0.0", "100890-0.0", "110005-0.0", "3731-0.0", "2664-0.0", "1618-0.0", "1628-0.0", "5364-0.0", "100022-0.0", "100510-0.0", "100580-0.0", "135-0.0", "40007-0.0", "41248-0.0", "87-0.0", "40008-0.0", "20009-0.0", "20008-0.0", "84-0.0", "20001-0.0", "20006-0.0", "2355-0.0", "2674-0.0", "40009-0.0", "20007-0.0", "40011-0.0", "20002-0.0", "2345-0.0", "40012-0.0", "3486-0.0", "2897-0.0", "20160-0.0", "6157-0.0", "2877-0.0", "3476-0.0", "1279-0.0", "3496-0.0", "20161-0.0", "2867-0.0", "3456-0.0", "2644-0.0", "20162-0.0", "2926-0.0", "6158-0.0", "2887-0.0", "1269-0.0", "2907-0.0", "3436-0.0", "3446-0.0", "2936-0.0", "3159-0.0", "23128-0.0", "728-0.0", "3761-0.0", "20010-0.0", "20011-0.0", "2966-0.0", "3506-0.0", "3486-0.0", "23124-0.0", "3526-0.0", "30070-0.0", "23113-0.0", "23114-0.0", "23118-0.0", "23117-0.0", "23125-0.0", "23122-0.0", "23121-0.0", "23126-0.0", "23129-0.0", "23130-0.0", "23098-0.0", "23109-0.0", "23107-0.0", "23110-0.0", "23108-0.0", "23112-0.0", "23116-0.0", "90015-0.0", "2754-0.0", "3591-0.0", "2764-0.0", "3829-0.0", "2724-0.0", "2794-0.0", "42007-0.0", "2415-0.0", "2744-0.0", "22001-0.0", "23129-0.0", "20151-0.0", "3063-0.0", "20150-0.0", "20153-0.0", "23101-0.0", "23098-0.0", "23104-0.0", "23121-0.0", "1239-0.0", "23122-0.0", "49-0.0", "23114-0.0", "23107-0.0", "23130-0.0", "2814-0.0", "23117-0.0", "23109-0.0", "20111-0.0", "23112-0.0", "23116-0.0", "23108-0.0", "23110-0.0", "23113-0.0", "23118-0.0", "23125-0.0", "20181-0.0", "30300-0.0", "4100-0.0", "3581-0.0", "20048-0.0", "20042-0.0", "20051-0.0", "4204-0.0", "20043-0.0", "42009-0.0", "3143-0.0", "6024-0.0", "20193-0.0", "20047-0.0", "20044-0.0", "4100-0.0", "4631-0.0", "21002-0.0", "46-0.0", "47-0.0", "48-0.0", "20018-0.0", "4119-0.0", "41214-0.0", "20045-0.0", "20046-0.0", "20041-0.0", "10711-0.0", "90016-0.0", "30050-0.0", "30260-0.0", "30040-0.0", "20191-0.0", "2824-0.0"]
columns_used = ['20524-0.0', '1538-0.0', '31-0.0', '30270-0.0', '30000-0.0', '699-0.0', '709-0.0', '3064-0.0', '20016-0.0', '30510-0.0', '22604-0.0', '6138-0.0', '100009-0.0', '30010-0.0', '1797-0.0', '100150-0.0', '6154-0.0', '1990-0.0', '1100-0.0', '6014-0.0', '2188-0.0', '22135-0.0', '20022-0.0', '22136-0.0', '20015-0.0', '20117-0.0', '30530-0.0', '23102-0.0', '20439-0.0', '4537-0.0', '3062-0.0', '3088-0.0', '6141-0.0', '6017-0.0', '4198-0.0', '6146-0.0', '1438-0.0', '4598-0.0', '6032-0.0', '21001-0.0', '6150-0.0', '3148-0.0', '41235-0.0', '680-0.0', '1835-0.0', '6142-0.0', '22140-0.0', '738-0.0', '30250-0.0', '100001-0.0', '50-0.0', '24024-0.0', '22139-0.0', '924-0.0', '1031-0.0', '22131-0.0', '1070-0.0', '20127-0.0', '22141-0.0', 'Column', '20116-0.0', '845-0.0', '22132-0.0', '30020-0.0', '23106-0.0', '21022-0.0', '23105-0.0', '1727-0.0', '1200-0.0', '1980-0.0', '3799-0.0', '136-0.0', '2149-0.0', '22138-0.0', '6148-0.0', '2492-0.0', '100920-0.0', '1080-0.0', '1558-0.0', '137-0.0']
significant_features = pd.read_csv("significant_total_bonferroni_corrected_p_values_all.tsv", sep='\t')

significant_features = significant_features[~significant_features.column.isin(filter_columns_list)]

for index, row in significant_features.iterrows():
    description = json.loads(row['description'].replace("'", '"'))
    significant_features.loc[index, 'description'] = description['description']
    
   
#select features that had enough data
used_features = significant_features[significant_features.column.isin(columns_used)]
used_features.to_csv('used_cleaned_significant_total_bonferroni_corrected_p_values_all.csv', index = False)


#prepare feature importance
files_list = ['abc', 'clf', 'mlp']

shap_feature_ranked_df = pd.read_csv("shap_feature_importance_ranked.csv")
shap_feature_ranked_dict = {}

for col in shap_feature_ranked_df:
    i = 0
    
    for item in shap_feature_ranked_df[col]:
        if item in shap_feature_ranked_dict:
            shap_feature_ranked_dict[item]['rank'].append(i)
        else:
            shap_feature_ranked_dict[item] = {'rank': [i]}
            
        i += 1
        
ranks_df = pd.DataFrame(columns=['feature', 'mean_rank', 'AdaBoost', 'RandomForest', 'MLP', 'lowest_rank_index'])

shap_feature_mean_values_list = []
for file in files_list:
     shap_feature_mean_values_list.append(pd.read_csv(file + "/shap_coef_df_summary.csv"))

for item in shap_feature_ranked_dict.copy():
    coef_list = []
    
    for shap_feature_mean_values_df in shap_feature_mean_values_list:
        df = shap_feature_mean_values_df[shap_feature_mean_values_df['Column'] == item]
        try:
            coef_list.append(df['coef'].values[0])
        except:
            coef_list.append(np.nan)
            
    lowest_rank = 1000
    rank_index = 0
    lowest_rank_index = 0
    for rank in shap_feature_ranked_dict[item]['rank']:
        if rank < lowest_rank:
            lowest_rank = rank
            lowest_rank_index = rank_index
            
        rank_index += 1
    
    shap_feature_ranked_dict[item]['mean_rank'] = statistics.mean(shap_feature_ranked_dict[item]['rank'])
    ranks_df = ranks_df.append(pd.Series([item, statistics.mean(shap_feature_ranked_dict[item]['rank']), coef_list[0], coef_list[1], coef_list[2], lowest_rank_index], index = ranks_df.columns), ignore_index = True)

ranks_df = ranks_df.sort_values(by = 'mean_rank')
number_of_features = 30
ranks_df = ranks_df.head(number_of_features)

ranks_df_copy = ranks_df[['feature']]

melt_ranks_df = pd.melt(ranks_df, id_vars = ['feature', 'mean_rank', 'lowest_rank_index'],var_name = 'model type', value_name = 'feature importance')
feature_names_dict = {'30530-0.0': "Sodium in urine", '30510-0.0': "Creatinine (enzymatic) in urine", '24024-0.0': "Average 24-hour sound level of noise pollution", '20116-0.0_2': "Smoking status current", '21022-0.0': "Age at recruitment", '20127-0.0': "Neuroticism score", '20117-0.0_1': "Alcohol drinker status previous", '1980-0.0': "Worrier / anxious feelings", '31-0.0': "Sex male", '6138-0.0_3': "Qualifications O levels/GCSEs or equivalent", '1727-0.0': "Ease of skin tanning", '1558-0.0': "Alcohol intake frequency.", '20022-0.0': "Birth weight", '22604-0.0': "Work hours - lumped category", '20116-0.0_1': "Smoking status previous", '6138-0.0_2': "Qualifications A levels/AS levels or equivalent", '1835-0.0_1': "Mother still alive", '2188-0.0': "Long-standing illness, disability or infirmity", '680-0.0_6': "Live in accommodation rent free", '6138-0.0_1': "Qualifications college or University degree", '1990-0.0': "Tense / 'highly strung'", '20117-0.0_0': "Alcohol drinker status never", '6138-0.0_-7': "No qualifications", '1538-0.0_2': "Major dietary changes in the last 5 years because other reasons", '30010-0.0': "Red blood cell (erythrocyte) count", '924-0.0_3': "Brisk usual walking pace", '4537-0.0': "Work/job satisfaction", '41235-0.0': "Spells in hospital", '23102-0.0': "Whole body water mass", '23106-0.0': "Impedance of whole body"}

for index, row in melt_ranks_df.iterrows():
    if row['lowest_rank_index'] == 0 and row['model type'] == "AdaBoost":
        melt_ranks_df.loc[index, 'lowest_rank_index'] = "highest rank"
    elif row['lowest_rank_index'] == 1 and row['model type'] == "RandomForest":
        melt_ranks_df.loc[index, 'lowest_rank_index'] = "highest rank"
    elif row['lowest_rank_index'] == 2 and row['model type'] == "MLP":
        melt_ranks_df.loc[index, 'lowest_rank_index'] = "highest rank"
    else:
        melt_ranks_df.loc[index, 'lowest_rank_index'] = "Other ranks"

for feature_name in feature_names_dict:
    melt_ranks_df.loc[melt_ranks_df.feature == feature_name, 'feature'] = feature_names_dict[feature_name]

from sklearn.preprocessing import StandardScaler
melt_ranks_np = StandardScaler().fit_transform(melt_ranks_df['feature importance'].to_numpy().reshape(-1, 1))
melt_ranks_df['feature importance'] = melt_ranks_np.reshape(3 * number_of_features,).tolist()

melt_ranks_df['highest rank'] = melt_ranks_df['lowest_rank_index']
plot = sns.scatterplot(data = melt_ranks_df, x = "feature importance", y = "feature", hue = "model type", style = "highest rank", markers = {'highest rank': 'p', 'Other ranks': 'o'}, palette = "colorblind")
sns.move_legend(plot, "lower left")
plot.set(ylabel = None)
plot.figure
plt.title("A)", loc = 'left')
plt.savefig("feature_importance_density" + ".png", bbox_inches = 'tight', dpi=600)
plt.close('all')

#plot individual ranks
ranks_individual_df = pd.DataFrame(columns = ['feature', 'AdaBoost', 'RandomForest', 'MLP'])
for item in shap_feature_ranked_dict.copy():
    try:
        rank_0 = shap_feature_ranked_dict[item]['rank'][0]
    except:
        rank_0 = np.nan
    try:
        rank_1 = shap_feature_ranked_dict[item]['rank'][1]
    except:
        rank_1 = np.nan
    try:
        rank_2 = shap_feature_ranked_dict[item]['rank'][2]
    except:
        rank_2 = np.nan
        
    ranks_individual_df = ranks_individual_df.append(pd.Series([item, rank_0, rank_1, rank_2], index = ranks_individual_df.columns), ignore_index = True)
    
ranks_df_copy_merge = ranks_df_copy.merge(ranks_individual_df, how = 'left')
melt_ranks_df_copy_merge = pd.melt(ranks_df_copy_merge, id_vars = ['feature'],var_name = 'model type', value_name = 'rank')

for feature_name in feature_names_dict:
    melt_ranks_df_copy_merge.loc[melt_ranks_df_copy_merge.feature == feature_name, 'feature'] = feature_names_dict[feature_name]

#plot = sns.histplot(melt_ranks_df_copy_merge, x = "rank", y = "feature", hue = "model type", legend = True, bins = 100)#.figure
plot = sns.scatterplot(data = melt_ranks_df_copy_merge, x = "rank", y = "feature", hue = "model type", palette = "colorblind")
sns.move_legend(plot, "upper right", title='Model type')
plot.set(ylabel = None)
plot.figure
plt.title("B)", loc = 'left')
plt.savefig("feature_importance_ranks_density" + ".png", bbox_inches = 'tight', dpi=600)
plt.close('all')

sys.exit()

#prepare thresholds
files_list = ['abc', 'clf', 'mlp', 'logreg', 'svc']
model_number_list = [3, 1, 2, 6, 4]
roc_list = []
baseline_dict = {}

#create shap values csv
shap_feature_mean_values_list = []

for file in files_list:
    shap_feature_mean_values_list.append(pd.read_csv(file + "/shap_coef_df_summary.csv"))
    
shap_feature_mean_values_df = shap_feature_mean_values_list[0][['Column', 'Description', 'data_coding']]

i = 0
while i < len(files_list):
    shap_feature_mean_values_df = pd.merge(left = shap_feature_mean_values_df, right = shap_feature_mean_values_list[i][['Column', 'coef']].rename(columns = {'coef': files_list[i]}), how = 'outer', on = ['Column'])
    i += 1

for index, row in shap_feature_mean_values_df.iterrows():
    try:
        shap_feature_mean_values_df.loc[index, 'data_coding'] = json.loads(row['data_coding'].replace("'", '"'))[row['Column'].split("_")[1]]
    except:
        pass
    
shap_feature_mean_values_df.to_csv('total_shap_feature_mean_values_df.csv', index = False)

for file in files_list:
    j = 0
    
    with open(file + '/y_probs.json', 'r') as f:
        for f_line in f:
            i = 0
            
            for f_line_split in f_line.split("}{"):
                if i == model_number_list[j]:
                    if not f_line_split.endswith("}"):
                        f_line_split = f_line_split + "}"
                    if not f_line_split.startswith("{"):
                        f_line_split = "{" + f_line_split
                    
                    y_probs_dict = json.loads(f_line_split)
                    
                i += 1
        
    roc = {'test_fpr': [], 'test_tpr': [], 'precision': [], 'recall': [], 'f1': [], 'accuracy': [], 'precission': [], 'fbeta0.5': [], 'confusion_matrix': []}
    
    for t in [t * t_times for t in range(1, t_range, 1)]:
        print(t)
        
        y_test_proba_adj = [1 if y >= t else 0 for y in y_probs_dict['probabilities']]
        tn, fp, fn, tp = metrics.confusion_matrix(y_probs_dict['y_true'], y_test_proba_adj, labels = [0, 1]).ravel()
        
        roc['test_fpr'].append(fp / (fp + tn))
        roc['test_tpr'].append(tp / (tp + fn)) #recall is the same as tpr
        roc['precision'].append(tp / (tp + fp))
        
        roc['accuracy'].append(metrics.accuracy_score(y_probs_dict['y_true'], y_test_proba_adj))
        roc['f1'].append(metrics.f1_score(y_probs_dict['y_true'], y_test_proba_adj))
        roc['recall'].append(metrics.recall_score(y_probs_dict['y_true'], y_test_proba_adj))
        roc['fbeta0.5'].append(metrics.fbeta_score(y_probs_dict['y_true'], y_test_proba_adj, beta = 0.5))
        roc['confusion_matrix'].append({'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp})
        
    roc_list.append(roc)
    
    baseline_dict[file] = sum(y_probs_dict['y_true']) / len(y_probs_dict['y_true'])
    
    j += 1

#get optimized threshold for f1
threshold_scorers_df = pd.DataFrame(columns=['accuracy', 'f1', 'recall', 'precision', 'fbeta0.5'])
i = 0
for roc in roc_list:
    j = 0
    highest_f1 = 0
    highest_f1_ind = 0
    
    for f1 in roc['f1']:
        if f1 > highest_f1:
            highest_f1 = f1
            highest_f1_ind = j
        
        j += 1
    
    with open(files_list[i] + "_moved_threshold_confusion.txt", 'w') as f:
        f.write(str(roc['confusion_matrix'][highest_f1_ind]))
        
    threshold_scorers_df.loc[files_list[i]] = [roc['accuracy'][highest_f1_ind], roc['f1'][highest_f1_ind], roc['recall'][highest_f1_ind], roc['precision'][highest_f1_ind], roc['fbeta0.5'][highest_f1_ind]]
    
    i += 1
    
threshold_scorers_df.to_csv('moved_threshold_confusion_scorers.csv', index = True)
#sys.exit()

abc_roc_values_df = pd.DataFrame({'False positive rate': roc_list[0]['test_fpr'], 'True positive rate': roc_list[0]['test_tpr'], 'Precision': roc_list[0]['precision'], 'fig': [1] * len(roc_list[0]['test_tpr'])})
clf_roc_values = pd.DataFrame({'False positive rate': roc_list[1]['test_fpr'], 'True positive rate': roc_list[1]['test_tpr'], 'Precision': roc_list[1]['precision'], 'fig': [2] * len(roc_list[1]['test_tpr'])})
mlp_roc_values = pd.DataFrame({'False positive rate': roc_list[2]['test_fpr'], 'True positive rate': roc_list[2]['test_tpr'], 'Precision': roc_list[2]['precision'], 'fig': [3] * len(roc_list[2]['test_tpr'])})
logreg_roc_values = pd.DataFrame({'False positive rate': roc_list[3]['test_fpr'], 'True positive rate': roc_list[3]['test_tpr'], 'Precision': roc_list[3]['precision'], 'fig': [4] * len(roc_list[3]['test_tpr'])})
svc_roc_values = pd.DataFrame({'False positive rate': roc_list[4]['test_fpr'], 'True positive rate': roc_list[4]['test_tpr'], 'Precision': roc_list[4]['precision'], 'fig': [5] * len(roc_list[4]['test_tpr'])})

#auc roc plot
sns.lineplot(data = abc_roc_values_df, x = 'False positive rate', y = 'True positive rate', label = 'AdaBoost [auroc = 0.94]').figure
sns.lineplot(data = clf_roc_values, x = 'False positive rate', y = 'True positive rate', label = 'RandomForest [auroc = 0.98]').figure
sns.lineplot(data = mlp_roc_values, x = 'False positive rate', y = 'True positive rate', label = 'MLP [auroc = 0.76]').figure
sns.lineplot(data = logreg_roc_values, x = 'False positive rate', y = 'True positive rate', label = 'LogisticRegression [auroc = 0.58]').figure
sns.lineplot(data = svc_roc_values, x = 'False positive rate', y = 'True positive rate', label = 'SVC [auroc = 0.52]').figure

plt.plot([0, 1], [0, 1], color = 'darkblue', linestyle='--')
plt.legend(fontsize = 5)
plt.savefig("roc_test_plot_" + ".png", bbox_inches = 'tight', dpi=600)
plt.close('all')

#precision recall plot nonlinear
baseline = []
for key in baseline_dict:
    baseline.append(baseline_dict[key])
baseline = statistics.mean(baseline)

sns.lineplot(data = abc_roc_values_df, x = 'True positive rate', y = 'Precision', label = 'AdaBoost [auprc = 0.50]').figure
sns.lineplot(data = clf_roc_values, x = 'True positive rate', y = 'Precision', label = 'RandomForest [auprc = 0.38]').figure
sns.lineplot(data = mlp_roc_values, x = 'True positive rate', y = 'Precision', label = 'MLP [auprc = 0.34]').figure

#plt.plot([0, 1], [1, 0], color = 'darkblue', linestyle='--')
baseline_label = "Baseline = " + str(round(baseline, 3))
plt.axhline(y = baseline, color = 'darkblue', linestyle = '--', label = baseline_label)
plt.title("A) Nonlinear", loc = 'left')
plt.legend(fontsize = 5)
plt.savefig("precision_recall_plot" + ".png", bbox_inches = 'tight', dpi=600)
plt.close('all')

#precision recall linear
sns.lineplot(data = logreg_roc_values, x = 'True positive rate', y = 'Precision', label = 'LogisticRegression [auprc = 0.007]').figure
sns.lineplot(data = svc_roc_values, x = 'True positive rate', y = 'Precision', label = 'SVC [auprc = 0.005]').figure

baseline_label = "Baseline = " + str(round(baseline, 3))
plt.axhline(y = baseline, color = 'darkblue', linestyle = '--', label = baseline_label)
plt.legend(fontsize = 5)
plt.title("B) Linear", loc = 'left')
plt.savefig("precision_recall_plot2" + ".png", bbox_inches = 'tight', dpi=600)
plt.close('all')

#precision recall threshold
abc_roc_values_df['Threshold'] = [t * t_times for t in range(1, t_range, 1)]
sns.lineplot(data = abc_roc_values_df, x = 'Threshold', y = 'Precision', label = 'Precision').figure
sns.lineplot(data = abc_roc_values_df, x = 'Threshold', y = 'True positive rate', label = 'True positive rate').figure

plt.legend(loc = 'upper right', fontsize = 5)
plt.title("A) AdaBoost", loc = 'left')
plt.ylabel(ylabel = None)
plt.savefig("abc_precision_recall_thresholds" + ".png", bbox_inches = 'tight', dpi=600)
plt.close('all')

##
clf_roc_values['Threshold'] = [t * t_times for t in range(1, t_range, 1)]
sns.lineplot(data = clf_roc_values, x = 'Threshold', y = 'Precision', label = 'Precision').figure
sns.lineplot(data = clf_roc_values, x = 'Threshold', y = 'True positive rate', label = 'True positive rate').figure

plt.legend(loc = 'upper right', fontsize = 5)
plt.title("B) RandomForest", loc = 'left')
plt.ylabel(ylabel = None)
plt.savefig("clf_precision_recall_thresholds" + ".png", bbox_inches = 'tight', dpi=600)
plt.close('all')

##
mlp_roc_values['Threshold'] = [t * t_times for t in range(1, t_range, 1)]
sns.lineplot(data = mlp_roc_values, x = 'Threshold', y = 'Precision', label = 'Precision').figure
sns.lineplot(data = mlp_roc_values, x = 'Threshold', y = 'True positive rate', label = 'True positive rate').figure

plt.legend(loc = 'upper right', fontsize = 5)
plt.title("C) MLP", loc = 'left')
plt.ylabel(ylabel = None)
plt.savefig("mlp_precision_recall_thresholds" + ".png", bbox_inches = 'tight', dpi=600)
plt.close('all')

##
logreg_roc_values['Threshold'] = [t * t_times for t in range(1, t_range, 1)]
sns.lineplot(data = logreg_roc_values, x = 'Threshold', y = 'Precision', label = 'Precision').figure
sns.lineplot(data = logreg_roc_values, x = 'Threshold', y = 'True positive rate', label = 'True positive rate').figure

plt.legend(loc = 'upper right', fontsize = 5)
plt.title("D) LogisticRegression", loc = 'left')
plt.ylabel(ylabel = None)
plt.savefig("logreg_precision_recall_thresholds" + ".png", bbox_inches = 'tight', dpi=600)
plt.close('all')

##
svc_roc_values['Threshold'] = [t * t_times for t in range(1, t_range, 1)]
sns.lineplot(data = svc_roc_values, x = 'Threshold', y = 'Precision', label = 'Precision').figure
sns.lineplot(data = svc_roc_values, x = 'Threshold', y = 'True positive rate', label = 'True positive rate').figure

plt.legend(loc = 'upper right', fontsize = 5)
plt.title("E) SVC", loc = 'left')
plt.ylabel(ylabel = None)
plt.savefig("svc_precision_recall_thresholds" + ".png", bbox_inches = 'tight', dpi=600)
plt.close('all')