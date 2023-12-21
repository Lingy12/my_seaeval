import os
import wandb
import fire
import json
from pathlib import Path
import pandas as pd

def extract_data(exp_root, data_name):
    results_data = []

    # Loop through each subdirectory in the given root directory
    for subdir, dirs, files in os.walk(exp_root):
        if subdir.endswith('results'):
            for test_folder in ['hidden_test', 'public_test']:
                log_path = os.path.join(subdir, test_folder, 'log')
                
                # Check if log folder exists
                if os.path.exists(log_path):
                    for file in os.listdir(log_path):
                        if file.endswith('.json'):
                            file_path = os.path.join(log_path, file)
                            
                            # Read and parse the JSON file
                            with open(file_path, 'r') as json_file:
                                data = json.load(json_file)
                                # print(os.path.pardir(subdir))
                                model_path = Path(subdir)
                                # print(model_path.parent.absolute())
        
                                extracted_data = {
                                    "name": f"{os.path.basename(model_path.parent.parent.absolute())}",
                                    "step": int(os.path.basename(subdir).split('-')[1]) if os.path.basename(subdir).split('-')[1] != 'best' else -1,
                                    "test_data": os.path.basename(file_path)[:-5],
                                    "test_mode": test_folder,
                                    'total_accuracy': data.get('Accuracy', None),
                                    'consistency_3': data.get('Consistency', {}).get('consistency_3', None),
                                    'AC3_3': data.get('AC3', {}).get('AC3_3', None),
                                }

                                lang_acc_data = data.get('Lang_Acc', {})
                                for key, value in lang_acc_data.items():
                                    extracted_data[key] = value
                                results_data.append(extracted_data)

    # print(results_data)
    data = pd.DataFrame.from_dict(results_data)
    data.sort_values(by=["name", "step"])
    data.to_excel(f'summary/{data_name}.xlsx')
    return data

def upload_to_wandb(exp_root, run_name):
    test_mode = ['hidden_test']
    test_data = ['cross_xquad_p1', 'cross_mmlu_p1', 'cross_logiqa_p1']
    metrics = ['consistency_3', 'AC3_3','Accuracy_english', 'Accuracy_chinese', 'Accuracy_vietnamese', 'Accuracy_spanish', 'total_accuracy']
    data = extract_data(exp_root, run_name)

    with wandb.init(project='seaeval_cl_v0', name=run_name) as run:
        for mode in test_mode:
            for data_name in test_data:
                for metric in metrics:
                    target_data = data[(data['test_mode'] == mode) & (data['test_data'] == data_name)].sort_values(by=['step'])
                    target_data = wandb.Table(data=target_data, columns=['step', metric, 'name'])
                    # wandb.log({f'{mode}/{data_name}/{metric}': wandb.plot.line(target_data, x='step', y=metric)})
                    # plt = wandb.plot_table(f"wandb/lineseris/v0", target_data, {"step": "step", "lineKey": "name", "lineVal": metric}, 
                                    # {"title": f"{mode}_{data_name}_{metric}", "xname": "step"})

                    wandb.log({f'{mode}/{data_name}/{metric}': target_data})
    


if __name__ == "__main__":
    fire.Fire(upload_to_wandb)
