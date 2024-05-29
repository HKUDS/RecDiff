import os, argparse
# from yaml import safe_load as yaml_load
from json import dumps as json_dumps

def parse_args():
    parser = argparse.ArgumentParser(description='SDR Arguments')
    parser.add_argument('--desc', type=str, default='')

   #Configuration Arguments
    parser.add_argument('--cuda',   type=str, default='0')
    parser.add_argument('--seed',   type=int, default=2023)

   #Model Arguments
    parser.add_argument('--n_hid',     type=int,   default=64)
    parser.add_argument('--n_layers',  type=int,   default=2)
    parser.add_argument('--s_layers', type=int, default=2)
    parser.add_argument('--weight', type=bool, default=True, help='Add linear weight or not')



    #Train Arguments
    parser.add_argument('--dropout', type=float, default=0)

    #Optimization Arguments
    parser.add_argument('--lr',         type=float, default=5e-3)
    parser.add_argument('--difflr', type=float, default=1e-3)
    parser.add_argument('--reg',        type=float, default=1e-2)
    parser.add_argument('--decay',      type=float, default=0.985)
    parser.add_argument('--decay_step', type=int,   default=1)
    parser.add_argument('--n_epoch',    type=int,   default=150)
    parser.add_argument('--batch_size', type=int,   default=2048)
    parser.add_argument('--patience',   type=int,   default=20)

    # Valid/Test Arguments 
    parser.add_argument('--topk',            type=int, default=20)
    parser.add_argument('--test_batch_size', type=int, default=1024)

    #Data Arguments
    parser.add_argument('--dataset',       type=str, default="epinions")
    parser.add_argument('--num_workers',   type=int, default=0)
    parser.add_argument('--save_name', type=str, default='tem')
    parser.add_argument('--checkpoint',    type=str, default="./Model/epinions/_tem_.pth")
    parser.add_argument('--model_dir',     type=str, default="./Model/epinions/")
   
    # params for the denoiser
    parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
    parser.add_argument('--dims', type=int, default=64, help='the dims for the DNN')
    parser.add_argument('--norm', type=bool, default=True, help='Normalize the input or not')
    parser.add_argument('--emb_size', type=int, default=16, help='timestep embedding size')

    # params for diffusions
    parser.add_argument('--steps', type=int, default=20, help='diffusion steps')
    parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
    parser.add_argument('--noise_scale', type=float, default=1, help='noise scale for noise generating')
    parser.add_argument('--noise_min', type=float, default=0.0001, help='noise lower bound for noise generating')
    parser.add_argument('--noise_max', type=float, default=0.01, help='noise upper bound for noise generating')
    parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
    parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
    parser.add_argument('--reweight', type=bool, default=True,
                        help='assign different weight to different timestep or not')



    return parser.parse_args()

args = parse_args()


# ciao 5e-3, 1e-3, 1 1e-2, 20, emb_size:16



