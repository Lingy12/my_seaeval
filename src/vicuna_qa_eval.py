import json
import os
import fire
import json

import logging
from pathlib import Path
from tqdm import trange
from collections import Counter
import json
from xquad_eval import normalize_answer, eval_dict
from dataset import Dataset
from model   import Model
EVAL_PROMPT = ("For question: {question}\nAnswer: {answer}\n\
Prediction: {prediction}\n\
Do you think the prediction can be treated as correct according to answer? Please answer in Yes or No: ")

def main(
        model_name  : str = "",
        batch_size  : int = 1, 
        data_file: str=""
):
    with open(data_file, 'r') as f:
        data = json.load(f)
    eval_res, accuracy_vicuna= eval_dict(model_name, batch_size, data)
        
    save_path = Path(data_file).with_suffix('.vicuna_eval.json')
    # print(json.dumps(eval_res, indent=2))
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(eval_res, f, ensure_ascii=False, indent=1)
        
    return accuracy_vicuna

if __name__ == "__main__":
    fire.Fire(main)