#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os

import pandas as pd

from tqdm import tqdm
import copy
import argparse
import time
torch.multiprocessing.set_sharing_strategy('file_system')

from MASC import cnn_create
from MASC import angle_pytorch as angle

import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#torch.multiprocessing.set_sharing_strategy('file_system')
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import shutil


def test_loading_batch(ds):
    batch_size=1
    if ds=='MNIST':
        _, test_loader = cnn_create.load_data("test",num_workers=1,batch_size=batch_size,mnist=True)
        
    if ds=='FashionMNIST':
        _, test_loader = cnn_create.load_data("test",num_workers=1,batch_size=batch_size,fashion=True)
        
    if ds=='CIFAR10':
        _, test_loader = cnn_create.load_data("test",num_workers=1,batch_size=batch_size,cifar10=True)
        
    if ds=='CIFAR100':
        _, test_loader = cnn_create.load_data("test",num_workers=1,batch_size=batch_size)
        
    if ds=='TinyImageNet':
        
        _, test_loader , _, _, _ = cnn_create.get_cifar_dataloaders_corrupted(batch_size=batch_size,
                                                                   tiny_imagenet=True)
    
    return test_loader

def sorted_model_numbers(network_path,corrupt,run):
    model_list=os.listdir(f'{network_path}/{corrupt}/Run_{run}')
    initialized_model = 'initialized_model.pth'
    remaining_models = [model for model in model_list if model != initialized_model]
    remaining_models.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    sorted_model_list = [initialized_model] + remaining_models
    return sorted_model_list


# In[2]:


def temp_store(ds,type_network,corrupt,run,n_value,seed_value,epoch,layer_number):
    #path for temprary activation storage
    temp_path = f'Network_data/masc_{ds}_{type_network}_{corrupt}_{run}_{seed_value}_{epoch}_{layer_number}'
    os.makedirs(temp_path,exist_ok=True)
      
    results_folder=f'results/{seed_value}/angle_results_{n_value}/{ds}_{type_network}'
    os.makedirs(results_folder,exist_ok=True)
    
    #results folders for corrupted subspace
    results_corr={}
    file_name=f'{results_folder}/results_{corrupt}/angle_results'
    os.makedirs(file_name,exist_ok=True)
    results_corr['angle_1']=file_name
    

    #results pca
    ba_folder=f'/mnt/8TB/simran/MASC/pca_saved/TMLR_compare'

    
    results_folder=f'{ba_folder}/{seed_value}/angle_results_{n_value}/{ds}_{type_network}'
    file_name=f'{results_folder}/results_{corrupt}/pca_corrupted'
    os.makedirs(file_name,exist_ok=True)
    results_corr['pca']=file_name

    
    return temp_path,results_corr


# In[3]:


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select corrupt, run, layer_number.")

    corrution_prob =[0.0,0.2,0.4,0.6,0.8] 
    run_values=[1,2,3]
    layer_numbers=[0,1,2,3,4,5,6,7,8]
    parser.add_argument(
        "-corr", type=float, required=True, choices=corrution_prob, help="select corruption"
    )

#     parser.add_argument(
#         "-run", type=int, required=True, choices=run_values, help="select run"
#     )
    parser.add_argument(
        "-layer_number", type=int, required=True, choices=layer_numbers, help="select layer_number"
    )
    args = parser.parse_args()
    corrupt = args.corr
#     run=args.run
    layer_number=args.layer_number
    if corrupt not in corrution_prob:
        args.print_help()
#     if run not in run_values:
#         args.print_help()
    if layer_number not in layer_numbers:
        args.print_help()
    

    type_network = 'ResNet18'
    ds ='CIFAR10'

    n_value=0.99
#     n_value=1

    runs=4  
    seed_value=40
    subspace_type='corrupt'

    #     for corrupt in corrution_prob:

    torch.manual_seed(seed_value)
    network_path=cnn_create.path_network_fn(type_network,ds)

    test_loader=test_loading_batch(ds)

    corrupted_train,og_targets,cor_targets=cnn_create.train_loading(ds,batch_size=1,
                                                                    corrupt=corrupt)

    for run in range(1,runs):
        print(ds,type_network,run)
#         model_numbers=sorted_model_numbers(network_path,corrupt,run)
        model_numbers=cnn_create.selected_model_list(network_path,corrupt,run)
        for model_epoch in model_numbers:  
            print(model_epoch)

            epoch=cnn_create.epochnumber(model_epoch)
            temp_path,results_corr=temp_store(ds,type_network,
                                          corrupt,run,n_value,seed_value,epoch,layer_number)
            print(results_corr)
            dummy_model=cnn_create.model_create(type_network,ds)
            path_file_load=f'{network_path}/{corrupt}/Run_{run}/{model_epoch}'
            dummy_model.load_state_dict(torch.load(path_file_load))
            cnn_create.ResNet18_loading_saving_activations_layerwise(temp_path,
                                                                     dummy_model,corrupted_train,
                                           test_loader,og_targets,dev,type_network,layer_number)
            angle.masc_probe(type_network,ds,temp_path,run,results_corr,n_value,epoch,dev,
                     subspace_type,layer_number)
            shutil.rmtree(temp_path)
            torch.cuda.empty_cache()

        print(f"done {run} ")
    print(f"done {corrupt} {layer_number}")

