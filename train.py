from scipy.ndimage import zoom
import torch
import argparse
import os
import numpy as np
from utils.utils import get_csv_split
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from loss import DiceLoss
import torch.nn as nn
import time
from FCN_2D import FCN_2D
import multiprocessing
from tqdm import tqdm
import yaml
import re
from utils.Calculate_metrics import Cal_metrics
from utils.utils import reshape_img
from utils.parallel import parallel
import pandas as pd
from dataloader import WorCapDataset


def train(model, criterion, train_loader, opt, device, e):
    model = model.to(device)
    model.train()
    train_sum = 0
    for j, batch in enumerate(train_loader):
        img, label = batch['image'].float(), batch['label'].float()
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
        img, label = batch['image'].float(), batch['label'].float()
        img, label = img.to(device), label.to(device)

        with torch.no_grad():
            outputs = model(img)
            loss = criterion(outputs, label)
        valid_sum += loss.item()
        print('Epoch {:<3d}  |Step {:>3d}/{:<3d}  | valid loss {:.4f}'.format(e, j, len(valid_loader), loss.item()))

    return valid_sum / len(valid_loader)


def inference(model, criterion, train_loader, valid_loader, device, save_img_path, is_infer_train=True):
    model.eval()
    if is_infer_train:
        for batch in tqdm(train_loader):
            img, label = batch['image'].float(), batch['label'].float()
            img, label = img.to(device), label.to(device)
            # file_name = batch['id_index'][0]

            with torch.no_grad():
                outputs = model(img)
                loss = criterion(outputs, label)
            outputs = torch.sigmoid(outputs)
            outputs = outputs.squeeze(1)
            pre = outputs.cpu().detach().numpy()

            ID = batch['image_index']
            affine = batch['affine']
            img_size = batch['image_size']
            os.makedirs(save_img_path, exist_ok=True)
            batch_save(ID, affine, pre, img_size, save_img_path)

    for batch in tqdm(valid_loader):
        img, label = batch['image'].float(), batch['label'].float()
        img, label = img.to(device), label.to(device)

        with torch.no_grad():
            outputs = model(img)
        outputs = torch.sigmoid(outputs)
        outputs = outputs.squeeze(1)
        pre = outputs.cpu().detach().numpy()

        ID = batch['image_index']
        affine = batch['affine']
        img_size = batch['image_size']
        os.makedirs(save_img_path, exist_ok=True)
        batch_save(ID, affine, pre, img_size, save_img_path)


def batch_save(ID, affine, pre, img_size, save_img_path):
    batch_size = len(ID)
    save_list = [save_img_path] * batch_size
    parallel(save_picture, pre, affine, img_size, save_list, ID, thread=True)


def save_picture(pre, affine, img_size, save_name, id):
    pre_label = pre
    pre_label[pre_label >= 0.5] = 1
    pre_label[pre_label < 0.5] = 0
    pre_label = reshape_img(pre_label, img_size.numpy())
    os.makedirs(os.path.join(save_name, id), exist_ok=True)
    nib.save(nib.Nifti1Image(pre_label, affine), os.path.join(save_name, id + '/pre_label.nii.gz'))

def args_input():
    p = argparse.ArgumentParser(description='cmd parameters')
    p.add_argument('--gpu_index', type=int, default=0)
    p.add_argument('--load_num', type=int, default=0)
    p.add_argument('--is_infer', action='store_true', default=True)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--layers', type=int, default=8)
    p.add_argument('--pools', type=int, default=1)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--epochs',type=int, default=30)
    p.add_argument('--channels', type=int, default=4)
    p.add_argument('--rl', type=float, default=0.001)
    return p.parse_args()


if __name__ == '__main__':
    args = args_input()
    gpu_index = args.gpu_index
    load_num = args.load_num
    batch_size = args.batch_size
    layers = args.layers
    pool_nums = args.pools
    num_workers = args.num_workers
    is_train = not args.is_infer
    epochs=args.epochs
    result_path = './results'
    channels = args.channels

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_index)
    torch.cuda.set_device(0)
    torch.backends.cudnn.enabled = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    input_size = [channels, 128, 128]

    learning_rate = args.rl


    model_save_path = r'%s/%s/%s/fold_%d/model_save' % (result_path, model_name, parameter_record, k)
    save_label_path = r'%s/%s/%s/fold_%d/pre_label' % (result_path, model_name, parameter_record, k)
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(save_label_path, exist_ok=True)
    

    dataset = WorCapDataset("/kaggle/input/worcap-2025/dataset_kaggle/dataset/t1", 
                            "/kaggle/input/worcap-2025/dataset_kaggle/dataset/t2", 
                            "/kaggle/input/worcap-2025/dataset_kaggle/dataset/mask")
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    dataset = loader.dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=loader.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=loader.batch_size, shuffle=False)


    net = FCN_2D(channels, layers).to(device)

    model_list = os.listdir(model_save_path)

    if load_num == 0:
        for m in net.modules():
            if isinstance(m, (nn.Conv3d)):
                nn.init.orthogonal_(m.weight)
    else:
        net.load_state_dict(torch.load(model_save_path + '/net_%d.pkl' % load_num))
        load_num = load_num + 1

    net_opt = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = DiceLoss()


    train_loss_set = []
    valid_loss_set = []
    epoch_list = []

    
    for e in range(load_num, epochs):
        print("=============train=============")
        train_loss = train(net, criterion, train_loader, net_opt, device, e)
        print("=============valid=============")
        valid_loss = valid(net, criterion, test_loader, device, e)

        train_loss_set.append(train_loss)
        valid_loss_set.append(valid_loss)
        epoch_list.append(e)
        print("train_loss:%f || valid_loss:%f" % (train_loss, valid_loss))
        
        if (e+1) % 5 == 0 or e == (epochs - 1):
            torch.save(net.state_dict(), model_save_path + '/net_%d.pkl' % (e+1))
    record = dict()
    record['epoch'] = epoch_list
    record['train_loss'] = train_loss_set
    record['valid_loss'] = valid_loss_set
    record = pd.DataFrame(record)
    record_name = time.strftime("%Y_%m_%d_%H.csv", time.localtime())
    record.to_csv(r'%s/%s/%s/fold_%d/%s' % (result_path, model_name, parameter_record, k, record_name), index=False)
    
