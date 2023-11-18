import fire
import os
from glob import glob
import re
import sys
import logging
import json
import pandas as pd

method_metric_map = {
    "cross_logiqa": {
        "accuracy": ["Accuracy"],
        "consistency": ["Consistency", "consistency_3"],
        "ac3": ["AC3", "AC3_3"],
        "en_acc": ["Lang_Acc", "Accuracy_english"],
        "zh_acc": ["Lang_Acc", "Accuracy_chinese"],
        "vi_acc": ["Lang_Acc", "Accuracy_vietnamese"],
        "es_acc": ["Lang_Acc", "Accuracy_spanish"],
    }, 
    "cross_mmlu": {
        "accuracy": ["Accuracy"],
        "consistency": ["Consistency", "consistency_3"],
        "ac3": ["AC3", "AC3_3"],
        "en_acc": ["Lang_Acc", "Accuracy_english"],
        "zh_acc": ["Lang_Acc", "Accuracy_chinese"],
        "vi_acc": ["Lang_Acc", "Accuracy_vietnamese"],
        "es_acc": ["Lang_Acc", "Accuracy_spanish"],
    }
}

root = logging.getLogger()
root.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler(stream=sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
root.addHandler(handler)

def extract_info(path):
    # Define a regular expression pattern to extract the details
    pattern = r'^(.+)/([^/]+)_(p\d+).json$'
    
    # Search for the pattern in the given path
    match = re.search(pattern, path)
    # If a match is found, extract the details and return as a dictionary
    if match:
        return {
            'model_name': match.group(1),
            'eval_method': match.group(2),
            'prompt_id': match.group(3)
        }
    else:
        raise Exception('Unexpected file path pattern, please check manifest file')

def access_dict(target, keys):
    # Iterate over the list of keys to access the nested dictionary
    data = target.copy()
    for key in keys:
        if key in data:
            data = data[key]
        else:
            raise Exception('Unexpected dictionary structure. Please check the definition of access.')  # Key not found
    return data

def build_eval_result(eval_dict, eval_method):
    eval_result = {}
    for metric in method_metric_map[eval_method]:
        eval_result[metric] = access_dict(eval_dict, method_metric_map[eval_method][metric])
    return eval_result
    
# def aggreagate_log_folder(log_folder):
#     # <log_folder>/<model_name>/<eval_method>_<prompt_id>.json
#     eval_manifest = glob(os.path.join(log_folder, '*/*.json'), recursive=True)
#     root.info('Getting {} manifests'.format(len(eval_manifest)))
#     result_dict = {} # model_name, eval_method, prompt_id, eval_metric

#     for res in eval_manifest:
#         with open(res, 'r') as f:
#             eval_res = json.load(f)
        
#         manifest_info = extract_info(os.path.relpath(res, log_folder))
#         model_name, eval_method, prompt_id = manifest_info['model_name'], manifest_info['eval_method'], manifest_info['prompt_id']
#         root.info('Building information for {} {} {}'.format(model_name, eval_method, prompt_id))
#         result_dict[(model_name, eval_method, prompt_id)] = build_eval_result(eval_res, eval_method)

#     unique_model_names = set([key[0] for key in result_dict.keys()])
#     unique_eval_methods = set([key[1] for key in result_dict.keys()])
#     medium = 'Median'

#     for model in unique_model_names:
#         for method in unique_eval_methods:
#             result_dict[(model, method, medium)] = {key: 0 for key in method_metric_map[eval_method].keys()}

#     df = pd.DataFrame.from_dict(result_dict, orient='index').sort_index(axis=0)
#     df.index.names = ['model_name', 'eval_method', 'promptid']
#     # Assuming your dataframe is called df
#     df_pivot = df.pivot_table(values=['accuracy', 'consistency', 'ac3', 'en_acc', 'zh_acc', 'vi_acc', 'es_acc'], 
#                             index=['model_name'],
#                             columns=['eval_method', 'promptid'],
#                             aggfunc='first').reset_index()

#     # # If you want to remove the top level column names (like 'accuracy', 'consistency', etc.)
#     # df_pivot.columns = df_pivot.columns.droplevel(0)

#     # # Reset the column names
#     # df_pivot.columns.name = None

#     print(df_pivot)
#     df_pivot.to_excel(os.path.join(log_folder, 'results.xlsx'))

def aggreagate_log_folder(log_folder):
    eval_manifest = glob(os.path.join(log_folder, '*/*.json'), recursive=True)
    root.info('Getting {} manifests'.format(len(eval_manifest)))
    result_list = []  # List to hold tuples of (model_name, eval_method, prompt_id, eval_metric)

    # Extract evaluation results from each JSON file
    for res in eval_manifest:
        with open(res, 'r') as f:
            eval_res = json.load(f)
        
        manifest_info = extract_info(os.path.relpath(res, log_folder))
        model_name, eval_method, prompt_id = manifest_info['model_name'], manifest_info['eval_method'], manifest_info['prompt_id']
        root.info('Building information for {} {} {}'.format(model_name, eval_method, prompt_id))
        result_list.append((model_name, eval_method, prompt_id, build_eval_result(eval_res, eval_method)))

    # Create a DataFrame from the list of tuples
    df = pd.DataFrame(result_list, columns=['model_name', 'eval_method', 'prompt_id', 'eval_metrics'])
    df = pd.concat([df.drop(['eval_metrics'], axis=1), df['eval_metrics'].apply(pd.Series)], axis=1)
    # Calculate median for each model_name and eval_method combination
    median_df = df.groupby(['model_name', 'eval_method']).median(numeric_only=True).reset_index()
    median_df['prompt_id'] = 'Median'  # Set the prompt_id to 'Median'
    # Append the median DataFrame to the original DataFrame
    print(median_df)
    df = pd.concat([df, median_df], ignore_index=True)

    # Pivot the DataFrame to the desired shape
    df_pivot = df.pivot_table(index=['model_name'], columns=['eval_method', 'prompt_id'], aggfunc='first').reset_index()
    print(df_pivot)
    # Remove the top level of the column MultiIndex (if you want to flatten the column structure)
    # df_pivot.columns = [f'{i}_{j}' if j != '' else f'{i}' for i, j in df_pivot.columns]

    df_pivot.to_excel(os.path.join(log_folder, 'results.xlsx'))

if __name__ == '__main__':
    fire.Fire(aggreagate_log_folder)