import fire
from transformers import AutoModel
import torch
import glob
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ""
def convert_checkpoint(checkpoint_folder, base_model, destination):
    parameter_file = glob.glob(os.path.join(checkpoint_folder, 'global_step*', 'mp_rank_00_model_states.pt'))
    model_state_dict = torch.load(parameter_file[0], map_location=torch.device('cpu'))

    print('Loading parameter file = {}'.format(parameter_file[0]))
    model = AutoModel.from_pretrained(base_model, state_dict = model_state_dict, device_map='auto')

    model.save_pretrained(destination)

if __name__ == "__main__":
    fire.Fire(convert_checkpoint)
