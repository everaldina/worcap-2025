import torch
import argparse
import os
import numpy as np
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from loss import DiceLoss
import torch.nn as nn
import time
from FCN_2D import FCN_2D
import pandas as pd
from dataloader import WorCapDataset, WorCapDiffDataset
import pickle


def inference(model, valid_loader, device, save_img_path):
    model.eval()
    for i, batch in enumerate(valid_loader):
        img, label, id = batch[0].float(), batch[1].float(), str(batch[2].item())
        img, label = img.to(device), label.to(device)

        with torch.no_grad():
            outputs = model(img)
        outputs = torch.sigmoid(outputs)
        outputs = outputs.squeeze(1)
        prediction = outputs.cpu().detach().numpy()
        
        #prediction[prediction >= 0.5] = 1
        #prediction[prediction < 0.5] = 0

        with open(os.path.join(save_img_path, f'recorte_{id}.pkl'), 'wb') as f:
            pickle.dump(prediction, f)
    
    

def args_input():
    p = argparse.ArgumentParser(description='cmd parameters')
    p.add_argument('load_num', type=int)
    p.add_argument('--gpu_index', type=int, default=0)
    p.add_argument('--layers', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--epochs',type=int, default=100)
    p.add_argument('--channels', type=int, default=8)
    p.add_argument('--rl', type=float, default=0.001)
    return p.parse_args()


if __name__ == '__main__':
    args = args_input()
    gpu_index = args.gpu_index
    load_num = args.load_num
    layers = args.layers
    num_workers = args.num_workers
    models_path = os.path.abspath('.') + '/results'
    channels = args.channels
    input_size = [channels, 128, 128]
    data_paths = {
        'before': 'data/dataset/t1',
        'after': 'data/dataset/t2',
        'mask': 'data/dataset/mask'
    }
    split_paths = 'split_ids.csv'
    split_df = pd.read_csv(split_paths)
    val_ids = split_df[split_df['split'] == 'val']['ID'].tolist()
    
    if load_num is None or load_num <= 0:
        raise ValueError("load_num must be a positive integer.")
    
    # print args
    print(f"gpu_index: {gpu_index}")
    print(f"load_num: {load_num}")
    print(f"layers: {layers}")
    print(f"num_workers: {num_workers}")
    print(f"models_path: {models_path}")
    print(f"channels: {channels}")
    print(f"input_size: {input_size}")
    

    model_save_path = os.path.join(models_path, f'FCN_2D_{channels}ch_{layers}lyr')
    os.makedirs(model_save_path, exist_ok=True)
    
    infer_path = os.path.join(model_save_path, 'infer')
    os.makedirs(infer_path, exist_ok=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_index)
    torch.cuda.set_device(0)
    torch.backends.cudnn.enabled = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if channels == 8:
        dataset_val = WorCapDataset(data_paths["before"], data_paths["after"], data_paths['mask'], val_ids)
    elif channels == 1:
        dataset_val = WorCapDiffDataset(data_paths["before"], data_paths["after"], data_paths['mask'], val_ids)
    test_loader = DataLoader(dataset_val, batch_size=1, shuffle=False)

    net = FCN_2D(channels, layers).to(device)
    net.load_state_dict(torch.load(model_save_path + '/net_%d.pkl' % load_num))

    
    inference(net, test_loader, device, infer_path)