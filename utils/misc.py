import json
import math, torch, os
from datetime import datetime
from random import random

import torch.nn as nn
import matplotlib.pyplot as plt
from torch.backends import cudnn


def normalize(yn, X, bs):
    if X is None:
        maxVal, _ = torch.max(torch.abs(yn.reshape(bs, -1)), dim=1)
        maxVal[maxVal == 0] = 1
        if len(yn.shape) == 3:
            return maxVal, yn / maxVal[:, None, None, None]
        return maxVal, yn / maxVal[:, None, None, None]
    else:
        maxVal, _ = torch.max(torch.abs(yn.reshape(bs, -1)), dim=1)
        if len(X.shape) == 3:
            return maxVal, yn / maxVal[:, None, None, None], X / maxVal[:, None, None]
        return maxVal, yn / maxVal[:, None, None, None], X / maxVal[:, None, None, None]


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_save_path(args, ):
    timeStamp = datetime.now().strftime("%m%d-%H%M")
    args.save_path = args.path + args.file_name + 'eps=' + str(args.noise_level) + '_' + timeStamp
    args.save_path = args.save_path if args.train else args.save_path + '_val'
    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        print('joined successfully!')
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(args, net, load_path):
    print('Loading Pre-trained model from', load_path)
    net.load_state_dict(torch.load(args.load_path)['state_dict'])


def set_seed(SEED):
    cuda = True if torch.cuda.is_available() else False
    print(os.getenv('CUDA_VISIBLE_DEVICES'), flush=True)
    cudnn.benchmark = True
    random.seed(SEED)
    torch.manual_seed(SEED)


def compute_psnr(X, Y):
    criteria = nn.MSELoss()
    return 20 * math.log10(1 / math.sqrt(criteria(X, Y)))

