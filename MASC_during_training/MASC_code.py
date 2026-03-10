#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import sys
# sys.path.append('/home/simrank/work/pca_during_training_angle')

from MASC import angle_pytorch as angle
from CNN_code import cnn_create

import os
import warnings
warnings.filterwarnings("ignore")
import torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.multiprocessing.set_sharing_strategy('file_system')
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import gc
import time
import shutil
import argparse


def select_model(values):
    selected_values = []
    # Starting index and increment
    i = 0
    increment = 2

    # Loop through the list with dynamic increments
    while i < len(values):
        if i==0:
            selected_values.append(values[i])
            selected_values.append(values[i+1])
        else:
            selected_values.append(values[i])
        i += increment
        # Update increment every time it reaches 20 to increase by 5
        if i < 20:
            increment = 2
        else:
            increment = 5

    return selected_values

def selected_model_list(path,corrupt,run):
    model_list=os.listdir(f'{path}/{corrupt}/Run_{run}')
    initialized_model = 'initialized_model.pth'
    remaining_models = [model for model in model_list if model != initialized_model]
    # Sort remaining models numerically
    remaining_models.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    # Combine them back with 'initialized_model' at the start
    sorted_model_list = [initialized_model] + remaining_models
    return select_model(sorted_model_list)

def epochnumber(epoch):
    if epoch!='initialized_model.pth':
        return int(epoch.split('_')[1].split('.')[0])+1
    else:
         return 0


if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(description="Select model_type, datasets and corruption.")
    
    corrution_prob = [0.0,0.2,0.4,0.6,0.8,1.0]
    model_type = ['CNN','MLP','AlexNet']
    datasets = ['CIFAR10','MNIST','FashionMNIST','TinyImageNet']
    run_values=[1,2,3]

    parser.add_argument(
        "-corr", type=float, required=True, choices=corrution_prob, help="select corruption"
    )
    parser.add_argument(
        "-model", type=str, required=True, choices=model_type, help="select model_type"
    )
    parser.add_argument(
        "-dataset", type=str, required=True, choices=datasets, help="select dataset"
    )

    parser.add_argument(
        "-run", type=int, required=True, choices=run_values, help="select run"
    )
    args = parser.parse_args()

    # Access arguments
    corrupt = args.corr
    type_network = args.model
    ds = args.dataset
    run=args.run

    if corrupt not in corrution_prob:
        args.print_help()
    if type_network not in model_type:
        args.print_help()
    if ds not in datasets:
        args.print_help()
        
    if run not in run_values:
        args.print_help()

    t0 = time.time()
    torch.manual_seed(42)
        
    n=0.99
#     n=1
    runs = 4
    #path to the models folder
    if type_network=='MLP':
        if ds=='MNIST':
            network_path = 'Model/MLP_MNIST/Network'
            _, test_loader , _,_,_= cnn_create.get_cifar_dataloaders_corrupted(batch_size=1,mnist=True)
            corrupted_train, _ , _, og_targets, _ = cnn_create.get_cifar_dataloaders_corrupted(corrupt_prob=corrupt, batch_size=1,mnist=True)
            
        if ds=='CIFAR10':
            network_path = 'Model/MLP_CIFAR10/Network'
            _, test_loader , _,_,_= cnn_create.get_cifar_dataloaders_corrupted(batch_size=1,cifar10=True)
            corrupted_train, _ , _, og_targets, _ = cnn_create.get_cifar_dataloaders_corrupted(corrupt_prob=corrupt, batch_size=1,cifar10=True)
    if type_network =='CNN':
        if ds=='FashionMNIST':
            network_path = 'Model/FashionMNIST/Network'
            _, test_loader , _,_,_= cnn_create.get_cifar_dataloaders_corrupted(batch_size=1,fashion=True)
            corrupted_train, _ , _, og_targets, _ = cnn_create.get_cifar_dataloaders_corrupted(corrupt_prob=corrupt, batch_size=1,fashion=True)
        if ds=='MNIST':
            network_path = 'Model/MNIST/Network'
            _, test_loader , _,_,_= cnn_create.get_cifar_dataloaders_corrupted(batch_size=1,mnist=True)
            corrupted_train, _ , _, og_targets, _ = cnn_create.get_cifar_dataloaders_corrupted(corrupt_prob=corrupt, batch_size=1,mnist=True)
        if ds=='CIFAR10':
            network_path = 'Model/CIFAR_10_Wodrop/Network'
            _, test_loader , _,_,_= cnn_create.get_cifar_dataloaders_corrupted(batch_size=1,cifar10=True)
            corrupted_train, _ , _, og_targets, _ = cnn_create.get_cifar_dataloaders_corrupted(corrupt_prob=corrupt, batch_size=1,cifar10=True)
            
    if type_network=='AlexNet':
        if ds=='TinyImageNet':
            network_path = 'Model/TinyImagenet_Alexnet/Network'
            _, test_loader , _,_,_= cnn_create.get_cifar_dataloaders_corrupted(batch_size=1,tiny_imagenet=True)
            corrupted_train, _ , _, og_targets, _ = cnn_create.get_cifar_dataloaders_corrupted(corrupt_prob=corrupt, batch_size=1,tiny_imagenet=True)
            
            
    #path for temprary activation storage
    temp_path = f'Network_data_{corrupt}_{ds}_{type_network}'
    os.makedirs(temp_path,exist_ok=True)
    results_folder=f'angle_results_1pc/{ds}_{type_network}'
    os.makedirs(results_folder,exist_ok=True)

    
    #results foldersfor corrupted subspace
    results_corr={}
    os.makedirs(f'{results_folder}/results_{corrupt}',exist_ok=True)
    os.makedirs(f'{results_folder}/results_{corrupt}/angle_results',exist_ok=True)
    results_corr['angle_1']=f'{results_folder}/results_{corrupt}/angle_results'
    #original training labels + corrupted training subspaces : exp2
    os.makedirs(f'{results_folder}/results_{corrupt}/angle_results_exp2',exist_ok=True)
    results_corr['angle_2']=f'{results_folder}/results_{corrupt}/angle_results_exp2'
    #results pca
    os.makedirs(f'{results_folder}/results_{corrupt}/pca',exist_ok=True)
    results_corr['pca']=f'{results_folder}/results_{corrupt}/pca'  
    
    
#     for run in range(1,runs): 
    for epoch in selected_model_list(network_path,corrupt,run):

        if type_network=='CNN':
            if ds=='CIFAR10':
                dummy_model = cnn_create.NgnCnn()
            else:
                dummy_model = cnn_create.NgnCnn(channels=1)
        if type_network=='MLP':
            if ds=='CIFAR10':
                dummy_model = cnn_create.mlp(mnist=False)
            else:
                dummy_model = cnn_create.mlp(mnist=True) 
        if type_network=='AlexNet':
            dummy_model = cnn_create.AlexNet(num_classes=200, tiny_imagenet=True)

        dummy_model.load_state_dict(torch.load(
            f'{network_path}/{corrupt}/Run_{run}/{epoch}',map_location =dev))
        cnn_create.loading_saving_activations(
            temp_path,dummy_model,corrupted_train,test_loader,
            og_targets,dev,type_network)

        del dummy_model
        angle.angle_work_corrupt(type_network,temp_path,run,results_corr,n,epochnumber(epoch))   
    print(f"run {run} done")
    print(f'corrupt {corrupt} done')
    
    shutil.rmtree(temp_path)
    t1 = time.time()
    total = t1-t0
    print(f'total time taken {total} sec')




