import os
import fire
import json

import logging

from tqdm import trange
from collections import Counter
import string
import re
import json
import sys
from dataset import Dataset
from model   import Model

from transformers import set_seed
set_seed(42) # ensure reproducability


# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  = 
logger = logging.getLogger(__name__)
logging.basicConfig(
    format  = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt = "%m/%d/%Y %H:%M:%S",
    level   = logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

PROMPT = ("{context} Please answer the following question with words in the above paragraph {question} Answer:  ")
EVAL_PROMPT = ("For question: {question}\nStudent A's answer: {answer}\n\
Student B's answer: {prediction}\n\
Do you think the answer from student A and student B are alternative of each other? Please answer in Yes or No only: ")


def main(
        model_name  : str = "",
        batch_size  : int = 1, 
        data_file: str=""
):
    
    logger.info("Model name: {}".format(model_name))
    logger.info("Batch size: {}".format(batch_size))

    model = Model(model_name)
    with open(data_file, 'r') as f:
        raw_data = json.load(f)

    data = []
    
    for article in raw_data['data']:
        for paragraph in article['paragraphs']:
            for question in paragraph['qas']:
                data.append({'context': paragraph['context'], 'question': question['question'], 'id': question['id'], 
                             'answer': ' '.join(list(map(lambda x: x['text'], question['answers'])))}) # prepare inference data

    predictions = do_inference(model, data, batch_size)
    eval_pred = {}
    prediction_dict = []


    for i in range(len(predictions)):
        eval_pred[data[i]['id']] = normalize_answer(predictions[i])
        with_pred = data[i]
        with_pred['prediction'] = predictions[i]
        prediction_dict.append(with_pred)
 
 
    vicuna_eval_res, vicuna_acc = eval_dict('./eval_model/vicuna-13b-v1.5', batch_size, prediction_dict)
    
    logger.info(vicuna_eval_res[0])
    
    xquad_res = {"vicuda_acc": vicuna_acc}
    
    model_save_name = os.path.basename(os.path.normpath(model_name))
    os.makedirs('log_predictions', exist_ok=True)
    os.makedirs('log/', exist_ok=True)
    
    os.makedirs('log/xsquad', exist_ok=True)
    os.makedirs('log_predictions/xsquad', exist_ok=True)
    predictions_path = os.path.join('log_predictions/xsquad', model_save_name + '_pred.json')
    res_path = os.path.join('log/xsquad', model_save_name + '_score.json')
    judge_path = os.path.join('log/xsquad', model_save_name + '_judge.json')
    
    with open(predictions_path, 'w', encoding='utf-8') as f:
        json.dump(prediction_dict, f, ensure_ascii=False, indent=1)
    with open(res_path, 'w', encoding='utf-8') as f:
        json.dump(xquad_res, f, ensure_ascii=False, indent=1)
    with open(judge_path, 'w', encoding='utf-8') as f:
        json.dump(vicuna_eval_res, f, ensure_ascii=False, indent=1)


def eval_dict(model_name, batch_size, data):
    model = Model(model_name_or_path=model_name, max_new_tokens=5)
    
    # normalize prediction and answer
    for i in range(len(data)):
        data[i]['prediction'] = normalize_answer(data[i]['prediction'])
        data[i]['answer'] = normalize_answer(data[i]['answer'])
    all_inputs = [EVAL_PROMPT.format_map(sample) for sample in data]
    
    # import pdb; pdb.set_trace()
    # input = ['我国最大的岛屿是什么？']
    # print(model.generate(input))
    print(all_inputs[2])
    predictions = []
    for i in trange(0, len(all_inputs), batch_size, leave=False):
        batch_inputs  = all_inputs[i:i+batch_size]

        batch_outputs = model.generate(batch_inputs)

        # import pdb; pdb.set_trace()
        predictions.extend(batch_outputs)
        eval_res = []
    for i in range(len(predictions)):
        eval_res.append({"question": data[i]['question'], "pred": data[i]['prediction'], "ans": data[i]['answer'], 
                         "judge": normalize_answer(predictions[i].strip())})
    
    res = list(map(lambda x: x['judge'], eval_res))
    res = list(map(lambda x: 1 if x == 'yes' else 0, res))
    accuracy_vicuna = res.count(1) / len(res)
    
    return eval_res, accuracy_vicuna

def do_inference(model, data, batch_size):
    all_inputs = [PROMPT.format_map(sample) for sample in data]

    # import pdb; pdb.set_trace()
    # input = ['我国最大的岛屿是什么？']
    # print(model.generate(input))
    predictions = []
    for i in trange(0, len(all_inputs), batch_size, leave=False):
        
        batch_inputs  = all_inputs[i:i+batch_size]

        batch_outputs = model.generate(batch_inputs)

        # import pdb; pdb.set_trace()
        predictions.extend(batch_outputs)

    return predictions

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

if __name__ == '__main__':
    fire.Fire(main)
