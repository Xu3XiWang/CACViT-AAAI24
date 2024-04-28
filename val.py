import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF

import timm

assert timm.__version__ == "0.3.2"  # version check

import util.misc as misc

os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

# load data from FSC147
data_path = '/data2/wangzhicheng/Dataset/FSC147/'
anno_file = data_path + 'annotation_FSC147_384.json'
data_split_file = data_path + 'Train_Test_Val_FSC_147.json'
im_dir = data_path + 'images_384_VarV2'
gt_dir = data_path + 'gt_density_map_adaptive_384_VarV2'

with open(anno_file) as f:
    annotations = json.load(f)

with open(data_split_file) as f:
    data_split = json.load(f)

class TestData(Dataset):
    def __init__(self, dataset='val'):
        self.img = data_split[dataset]
        self.img_dir = im_dir

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        im_id = self.img[idx]
        anno = annotations[im_id]
        bboxes = anno['box_examples_coordinates']

        dots = np.array(anno['points'])

        image = Image.open('{}/{}'.format(im_dir, im_id))
        image.load() 
        W, H = image.size

        new_H = 16*int(H/16)
        new_W = 16*int(W/16)
        scale_factor = float(new_W)/ W
        image = transforms.Resize((new_H, new_W))(image)
        Normalize = transforms.Compose([transforms.ToTensor()])
        image = Normalize(image)

        rects = list()
        for bbox in bboxes:
            x1 = int(bbox[0][0]*scale_factor)
            y1 = bbox[0][1]
            x2 = int(bbox[2][0]*scale_factor)
            y2 = bbox[2][1]
            rects.append([y1, x1, y2, x2])

        boxes = list()
        scale_x = []
        scale_y = []
        cnt = 0
        for box in rects:
            cnt+=1
            if cnt>3:
                break
            box2 = [int(k) for k in box]
            y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
            scale_x1 = torch.tensor((x2-x1+1)/384)
            scale_x.append(scale_x1)
            scale_y1 = torch.tensor((y2-y1+1)/384)
            scale_y.append(scale_y1)
            bbox = image[:,y1:y2+1,x1:x2+1]
            bbox = transforms.Resize((64, 64))(bbox)
            boxes.append(bbox.numpy())
        scale_xx = torch.stack(scale_x).unsqueeze(-1)
        scale_yy = torch.stack(scale_y).unsqueeze(-1)
        scale = torch.cat((scale_xx,scale_yy),dim=1)
        boxes = np.array(boxes)
        boxes = torch.Tensor(boxes)

        # Only for visualisation purpose, no need for ground truth density map indeed.
        gt_map = np.zeros((image.shape[1], image.shape[2]),dtype='float32')
        for i in range(dots.shape[0]):
            gt_map[min(new_H-1,int(dots[i][1]))][min(new_W-1,int(dots[i][0]*scale_factor))]=1
        gt_map = ndimage.gaussian_filter(gt_map, sigma=(1, 1), order=0)
        gt_map = torch.from_numpy(gt_map)
        gt_map = gt_map *60
        
        sample = {'image':image,'dots':dots, 'boxes':boxes, 'pos':rects, 'gt_map':gt_map, 'scale':scale}
        return sample['image'], sample['dots'], sample['boxes'], sample['pos'] ,sample['gt_map'],im_id,sample['scale']


def val_func(model,device,dataset='val'):
    dataset_test = TestData(dataset)
    print(dataset_test)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
        print("Sampler_test = %s" % str(sampler_test))
    else:
        sampler_test = torch.utils.data.RandomSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=1,
        num_workers=10,
        pin_memory=True,
        drop_last=False,
    )
    # test
    epoch = 0
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    # some parameters in training
    train_mae = 0
    train_rmse = 0
    pred_cnt = 0
    gt_cnt = 0

    loss_array = []
    gt_array = []
    wrong_id = []
    model.eval()
    # device = model.device

    for data_iter_step, (samples, gt_dots, boxes, pos, gt_map,im_id,scale) in enumerate(metric_logger.log_every(data_loader_test, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        gt_dots = gt_dots.to(device, non_blocking=True).half()
        boxes = boxes.to(device, non_blocking=True)
        scale = scale.to(device, non_blocking=True)
        pos = pos
        gt_map = gt_map.to(device, non_blocking=True)
        
        _,_,h,w = samples.shape 

        r_cnt = 0
        s_cnt = 0
        for rect in pos:
            r_cnt+=1
            if r_cnt>3:
                break
            if rect[2]-rect[0]<10 and rect[3] - rect[1]<10:
                s_cnt +=1
        
        if s_cnt >= 1:
            r_images = []
            r_images.append(TF.crop(samples[0], 0, 0, int(h/3), int(w/3)))
            r_images.append(TF.crop(samples[0], int(h/3), 0, int(h/3), int(w/3)))
            r_images.append(TF.crop(samples[0], 0, int(w/3), int(h/3), int(w/3)))
            r_images.append(TF.crop(samples[0], int(h/3), int(w/3), int(h/3), int(w/3)))
            r_images.append(TF.crop(samples[0], int(h*2/3), 0, int(h/3), int(w/3)))
            r_images.append(TF.crop(samples[0], int(h*2/3), int(w/3), int(h/3), int(w/3)))
            r_images.append(TF.crop(samples[0], 0, int(w*2/3), int(h/3), int(w/3)))
            r_images.append(TF.crop(samples[0], int(h/3), int(w*2/3), int(h/3), int(w/3)))
            r_images.append(TF.crop(samples[0], int(h*2/3), int(w*2/3), int(h/3), int(w/3)))

            pred_cnt = 0
            for r_image in r_images:
                r_image = transforms.Resize((h, w))(r_image).unsqueeze(0)
                density_map = torch.zeros([h,w])
                density_map = density_map.to(device, non_blocking=True)
                start = 0
                prev = -1
                
                with torch.no_grad():
                    while start + 383 < w:
                        input_x = [r_image[:,:,:,start:start+384], boxes,scale]
                        output = model(input_x)
                        output=output.squeeze(0)
                        b1 = nn.ZeroPad2d(padding=(start, w-prev-1, 0, 0))
                        d1 = b1(output[:,0:prev-start+1])
                        b2 = nn.ZeroPad2d(padding=(prev+1, w-start-384, 0, 0))
                        d2 = b2(output[:,prev-start+1:384])            
                        
                        b3 = nn.ZeroPad2d(padding=(0, w-start, 0, 0))
                        density_map_l = b3(density_map[:,0:start])
                        density_map_m = b1(density_map[:,start:prev+1])
                        b4 = nn.ZeroPad2d(padding=(prev+1, 0, 0, 0))
                        density_map_r = b4(density_map[:,prev+1:w])

                        density_map = density_map_l + density_map_r + density_map_m/2 + d1/2 +d2

                        prev = start + 383
                        start = start + 128
                        if start+383 >= w:
                            if start == w - 384 + 128: break
                            else: start = w - 384

                pred_cnt += torch.sum(density_map/60).item()
            pred_cnt_new = 0
            density_map = torch.zeros([h,w])
            density_map = density_map.to(device, non_blocking=True)
            start = 0
            prev = -1
            with torch.no_grad():
                while start + 383 < w:
                    input_x = []
                    a = samples[:,:,:,start:start+384]
                    input_x.append(a)
                    input_x.append(boxes)
                    input_x.append(scale)
                    output = model(input_x)
                    output=output.squeeze(0)
                    b1 = nn.ZeroPad2d(padding=(start, w-prev-1, 0, 0))
                    d1 = b1(output[:,0:prev-start+1])
                    b2 = nn.ZeroPad2d(padding=(prev+1, w-start-384, 0, 0))
                    d2 = b2(output[:,prev-start+1:384])            
                    
                    b3 = nn.ZeroPad2d(padding=(0, w-start, 0, 0))
                    density_map_l = b3(density_map[:,0:start])
                    density_map_m = b1(density_map[:,start:prev+1])
                    b4 = nn.ZeroPad2d(padding=(prev+1, 0, 0, 0))
                    density_map_r = b4(density_map[:,prev+1:w])

                    density_map = density_map_l + density_map_r + density_map_m/2 + d1/2 +d2

                    prev = start + 383
                    start = start + 128
                    if start+383 >= w:
                        if start == w - 384 + 128: break
                        else: start = w - 384

            pred_cnt_new = torch.sum(density_map/60).item()

            e_cnt = 0
            cnt = 0
            for rect in pos:
                cnt+=1
                if cnt>3:
                    break
                e_cnt += torch.sum(density_map[rect[0]:rect[2]+1,rect[1]:rect[3]+1]/60).item()
            e_cnt = e_cnt / 3
            if e_cnt > 1.8:
                pred_cnt_new /= e_cnt
            
            if pred_cnt>pred_cnt_new * 9:
                pred_cnt = pred_cnt_new
        else: 
            density_map = torch.zeros([h,w])
            density_map = density_map.to(device, non_blocking=True)
            start = 0
            prev = -1
            with torch.no_grad():
                while start + 383 < w:
                    input_x = []
                    a = samples[:,:,:,start:start+384]
                    input_x.append(a)
                    input_x.append(boxes)
                    input_x.append(scale)
                    output = model(input_x)
                    output=output.squeeze(0)
                    b1 = nn.ZeroPad2d(padding=(start, w-prev-1, 0, 0))
                    d1 = b1(output[:,0:prev-start+1])
                    b2 = nn.ZeroPad2d(padding=(prev+1, w-start-384, 0, 0))
                    d2 = b2(output[:,prev-start+1:384])            
                    
                    b3 = nn.ZeroPad2d(padding=(0, w-start, 0, 0))
                    density_map_l = b3(density_map[:,0:start])
                    density_map_m = b1(density_map[:,start:prev+1])
                    b4 = nn.ZeroPad2d(padding=(prev+1, 0, 0, 0))
                    density_map_r = b4(density_map[:,prev+1:w])

                    density_map = density_map_l + density_map_r + density_map_m/2 + d1/2 +d2

                    prev = start + 383
                    start = start + 128
                    if start+383 >= w:
                        if start == w - 384 + 128: break
                        else: start = w - 384

            pred_cnt = torch.sum(density_map/60).item()

            e_cnt = 0
            cnt = 0
            for rect in pos:
                cnt+=1
                if cnt>3:
                    break
                e_cnt += torch.sum(density_map[rect[0]:rect[2]+1,rect[1]:rect[3]+1]/60).item()
            e_cnt = e_cnt / 3
            if e_cnt > 1.8:
                pred_cnt /= e_cnt
        
        gt_cnt = gt_dots.shape[1]
        cnt_err = abs(pred_cnt - gt_cnt)
        if cnt_err > 200:
            wrong_id.append(im_id)
            print(im_id)
        train_mae += cnt_err
        train_rmse += cnt_err ** 2

        print(f'{data_iter_step}/{len(data_loader_test)}: pred_cnt: {pred_cnt},  gt_cnt: {gt_cnt},  error: {cnt_err},  AE: {cnt_err},  SE: {cnt_err ** 2} ')

        loss_array.append(cnt_err)
        gt_array.append(gt_cnt)
        
        torch.cuda.synchronize(device=0)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    log_stats = {'MAE': train_mae/(len(data_loader_test)),
                'RMSE':  (train_rmse/(len(data_loader_test)))**0.5}
    mae = train_mae/(len(data_loader_test))
    mse = (train_rmse/(len(data_loader_test)))**0.5
    print('Current MAE: {:5.2f}, RMSE: {:5.2f} '.format( train_mae/(len(data_loader_test)), (train_rmse/(len(data_loader_test)))**0.5))
    return mae, mse

