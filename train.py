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


def train(model, criterion, train_loader, opt, device, e):
    model = model.to(device)
    model.train()
    train_sum = 0
    for j, batch in enumerate(train_loader):
        #print(batch[0].shape, batch[1].shape)
        img, label = batch[0].float(), batch[1].float()
        img, label = img.to(device), label.to(device)
        outputs = model(img)
        opt.zero_grad()
        loss = criterion(outputs, label)
        print('Epoch {:<3d}  |  Step {:>3d}/{:<3d}  | train loss {:.4f}'.format(e, j, len(train_loader), loss.item()))
        train_sum += loss.item()
        loss.backward()
        opt.step()
    return train_sum / len(train_loader)


def valid(model, criterion, valid_loader, device, e):
    model.eval()
    valid_sum = 0
    for j, batch in enumerate(valid_loader):
        img, label = batch[0].float(), batch[1].float()
        img, label = img.to(device), label.to(device)

        with torch.no_grad():
            outputs = model(img)
            loss = criterion(outputs, label)
        valid_sum += loss.item()
        print('Epoch {:<3d}  |Step {:>3d}/{:<3d}  | valid loss {:.4f}'.format(e, j, len(valid_loader), loss.item()))

    return valid_sum / len(valid_loader)

def args_input():
    p = argparse.ArgumentParser(description='cmd parameters')
    p.add_argument('--gpu_index', type=int, default=0)
    p.add_argument('--load_num', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=32)
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
    batch_size = args.batch_size
    layers = args.layers
    num_workers = args.num_workers
    epochs=args.epochs
    result_path = os.path.abspath('.') + '/results'
    channels = args.channels
    input_size = [channels, 128, 128]
    learning_rate = args.rl
    data_paths = {
        'before': 'data/dataset/t1',
        'after': 'data/dataset/t2',
        'mask': 'data/dataset/mask'
    }
    split_paths = 'split_ids.csv'
    split_df = pd.read_csv(split_paths)
    train_ids = split_df[split_df['split'] == 'train']['ID'].tolist()
    val_ids = split_df[split_df['split'] == 'val']['ID'].tolist()
    
    # print args
    print(f"gpu_index: {gpu_index}")
    print(f"load_num: {load_num}")
    print(f"batch_size: {batch_size}")
    print(f"layers: {layers}")
    print(f"num_workers: {num_workers}")
    print(f"epochs: {epochs}")
    print(f"result_path: {result_path}")
    print(f"channels: {channels}")
    print(f"input_size: {input_size}")
    print(f"learning_rate: {learning_rate}")
    

    model_save_path = os.path.join(result_path, f'FCN_2D_{channels}ch_{layers}lyr')
    os.makedirs(model_save_path, exist_ok=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_index)
    torch.cuda.set_device(0)
    torch.backends.cudnn.enabled = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    if channels == 8:
        dataset_train = WorCapDataset(data_paths["before"], data_paths["after"], data_paths['mask'], train_ids)
        dataset_val = WorCapDataset(data_paths["before"], data_paths["after"], data_paths['mask'], val_ids)
    elif channels == 1:
        dataset_train = WorCapDiffDataset(data_paths["before"], data_paths["after"], data_paths['mask'], train_ids)
        dataset_val = WorCapDiffDataset(data_paths["before"], data_paths["after"], data_paths['mask'], val_ids)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    net = FCN_2D(channels, layers).to(device)

    if load_num == 0:
        for m in net.modules():
            if isinstance(m, (nn.Conv3d)):
                nn.init.orthogonal_(m.weight)
    else:
        net.load_state_dict(torch.load(model_save_path + '/net_%d.pkl' % load_num))
        load_num = load_num + 1

    net_opt = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = DiceLoss()

    if load_num == 0:
        train_loss_set = []
        valid_loss_set = []
        epoch_list = []
        duration = []
    else:
        records = pd.read_csv(os.path.join(model_save_path, 'train_record.csv'))
        train_loss_set = records['train_loss'].tolist()
        valid_loss_set = records['valid_loss'].tolist()
        epoch_list = records['epoch'].tolist()
        duration = records['duration'].tolist()

    
    for e in range(load_num, epochs):
        time_start = time.time()
        print("=============train=============")
        train_loss = train(net, criterion, train_loader, net_opt, device, e)
        print("=============valid=============")
        valid_loss = valid(net, criterion, test_loader, device, e)

        train_loss_set.append(train_loss)
        valid_loss_set.append(valid_loss)
        epoch_list.append(e)
        print("train_loss:%f || valid_loss:%f" % (train_loss, valid_loss))
        time_end = time.time()
        duration.append(time_end - time_start)
        
        if (e+1) % 5 == 0 or e == (epochs - 1):
            torch.save(net.state_dict(), model_save_path + '/net_%d.pkl' % (e+1))

    record = dict()
    record['epoch'] = epoch_list
    record['train_loss'] = train_loss_set
    record['valid_loss'] = valid_loss_set
    record['duration'] = duration
    record = pd.DataFrame(record)
    record.to_csv(os.path.join(model_save_path, 'train_record.csv'), index=False)
    
