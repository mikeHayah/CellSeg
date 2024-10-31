import argparse
import os
import torch.nn as nn
import numpy
from solver import Solver
from patched_data_loader import get_loader
from torch.backends import cudnn
import random
import cv2
import predict_patched
from PIL import Image
import torch
import atexit


def main(config):
    cudnn.benchmark = True
    if config.model_type not in ['U_Net','R2U_Net','AttU_Net','R2AttU_Net','StandardUNet']:
        print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net')
        print('Your input for model_type was %s'%config.model_type)
        return

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path,config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    
    lr = random.random()*0.0005 + 0.0000005
    augmentation_prob= random.random()*0.7
    epoch = random.choice([100,150,200,250])
    decay_ratio = random.random()*0.8
    decay_epoch = int(epoch*decay_ratio)

    config.augmentation_prob = augmentation_prob
    config.num_epochs = 150 #epoch
    config.lr = lr
    config.num_epochs_decay = decay_epoch

    print(config)
        


    train_loader = get_loader(image_path=config.train_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='train',
                            augmentation_prob=config.augmentation_prob)
    valid_loader = get_loader(image_path=config.valid_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='valid',
                            augmentation_prob=0.)
    test_loader = get_loader(image_path=config.test_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='test',
                            augmentation_prob=0.)

    solver = Solver(config, train_loader, valid_loader, test_loader)
    
    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        
        predict_patched.make_predictions("./models/StandardUNet-0.0002-149-3.2712.pkl", config.test_path)      #update this part to include all images in the test folder
    elif config.mode == 'retrain':
        solver.retrain("./models/U_Net-150-0.0002-27-0.0261_retrianed.pkl")
    


def create_labels():
    original_GT = 'C:/Users/zerialadmin/Documents/Image_Segmentation/dataset/valid_GT/'
    labeled_GT ='C:/Users/zerialadmin/Documents/Image_Segmentation/dataset/valid_GT_binary/'
    if not os.path.exists(labeled_GT):
        os.makedirs(labeled_GT)
    paths = list(map(lambda x: os.path.join(original_GT, x), os.listdir(original_GT)))
    for image_path in paths:
        if image_path.endswith('.tif'):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            _, binary_image = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY)
            cv2.imwrite(os.path.join(labeled_GT,image_path.split('/')[-1]), binary_image)
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    
    parser.add_argument('--image_size', type=int, default=(1024,1024))
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')
    
    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=1)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_epochs_decay', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    
    parser.add_argument('--augmentation_prob', type=float, default=0.4)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)

    # misc
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--model_type', type=str, default='StandardUNet', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net/StandardUNet')
    parser.add_argument('--model_path', type=str, default='./models/')
    parser.add_argument('--train_path', type=str, default='./datasetCell/train/')
    parser.add_argument('--valid_path', type=str, default='./datasetCell/valid/')
    parser.add_argument('--test_path', type=str, default='./datasetCell/test/')
    parser.add_argument('--result_path', type=str, default='./result/')

    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()

    
    main(config)
    
    