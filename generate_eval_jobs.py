import os
import fire
import re
from pathlib import Path
TEMPLATE = '.scripts/eval_template.sh'


with open(TEMPLATE, 'r') as f:
    template = f.read()
# print(template)

def get_place_holder(template):
    placeholders = re.findall(r'{{(.*?)}}', template)
    return list(placeholders)

def generate_jobs(model_dir, dest_dir, eval_mode):
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    
    model_abs = os.path.abspath(model_dir)
    models = os.listdir(model_dir)
    models = list(map(lambda x: os.path.join(model_abs, x), models))
    # print(models)
    # exit()
    params_required = get_place_holder(template)
    for model in models:
        params = {"ngpu": str(1), "model_name": str(model), "eval_mode":str(eval_mode), "script_dir": os.getcwd(), 
                  'model_name_base': str(os.path.basename(os.path.normpath(model)))}
        bash_script = template
        missing_key = set(params_required) - set(params.keys())
        if len(missing_key) == 0:
            print('All key are valid')
        else:
            raise Exception('{} are missing from params. '.format(','.join(missing_key)))
        for param, value in params.items():
                placeholder = '{{' + param + '}}'
                bash_script = bash_script.replace(placeholder, value)
        
        with open(os.path.join(dest_dir, f'job_run_{os.path.basename(model)}.sh'), 'w') as f:
            f.write(bash_script)
            
if __name__ == '__main__':
    fire.Fire(generate_jobs)