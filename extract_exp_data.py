import os
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
                                    "name": f"{os.path.basename(model_path.parent.absolute())}",
                                    "step": os.path.basename(subdir).split('-')[1],
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

if __name__ == "__main__":
    fire.Fire(extract_data)