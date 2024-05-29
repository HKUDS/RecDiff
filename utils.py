import torch, os, pickle, random
import numpy as np
# from yaml import safe_load as yaml_load
# from json import dumps as json_dumps


def load_data(data_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_model(model,difussion_model ,save_path, optimizer1=None,optimizer2 = None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data2save = {
        'state_dict1': model.state_dict(),
        'optimizer1': optimizer1.state_dict(),
        'state_dict2': difussion_model.state_dict(),
        'optimizer2': optimizer2.state_dict(),

    }
    torch.save(data2save, save_path)



def load_model(model,model2, load_path, optimizer=None):
    data2load = torch.load(load_path, map_location='cpu')
    model.load_state_dict(data2load['state_dict1'])
    model2.load_state_dict(data2load['state_dict2'])
    if optimizer is not None and data2load['optimizer'] is not None:
        optimizer = data2load['optimizer']



def fix_random_seed_as(seed):
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    pass
