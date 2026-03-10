#!/usr/bin/env python
# coding: utf-8

# In[1]:


from MASC import angle_pytorch as angle
from CNN_code import cnn_create
from MAVC import MAVC_pytorch as mavc
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
import copy 
from tqdm import tqdm
import pandas as pd

from MASC.angle_retrain import *
from MAVC.retrain import *


# In[2]:


def temp_store(ds,type_network,corrupt,run,peak_epoch):
    #path for temprary activation storage
    temp_path = f'Network_data/VeLPIC_intervention_{ds}_{type_network}_{corrupt}_{run}_{peak_epoch}'
    os.makedirs(temp_path,exist_ok=True)
      
    results_folder=f'results/angle_results_{peak_epoch}/{ds}_{type_network}'
    os.makedirs(results_folder,exist_ok=True)
    
    #results foldersfor corrupted subspace
    results_corr={}
    file_name=f'{results_folder}/results_{corrupt}/angle_results'
    os.makedirs(file_name,exist_ok=True)
    results_corr['angle_1']=file_name

    #results pca
    file_name=f'{results_folder}/results_{corrupt}/pca_corrupted'
    os.makedirs(file_name,exist_ok=True)
    results_corr['pca']=file_name
       
    
    return temp_path,results_corr


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

def epoch_number(model_epoch):
    if model_epoch =='model_100.pth':
        return 100
    else:
        return model_epoch[-6:-4]
    
def subspace_function(type_network,ds,temp_path,run,results,n_value,epoch_present,dev):

    data_layer_name,data_output_name,num_class=layer_names(type_network,ds)

    results_angle1=results['angle_1']
    results_pca=results['pca']
    batch_size=100
    epoch=epoch_present
    #results folders 
    path2=f'{results_angle1}/Run_{run}/{epoch}'
    os.makedirs(path2,exist_ok=True)

    path2_pca=f'{results_pca}/Run_{run}/{epoch}' 
    os.makedirs(path2_pca,exist_ok=True)

    with torch.no_grad(): 
        for data,data_out in zip(data_layer_name,data_output_name):
            flops_train=subspace_creation(temp_path,n_value,path2_pca,data,data_out,num_class)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Select model_type, datasets and corruption.")
    
    corrution_prob = [0.0,0.2,0.4,0.6,0.8]
    model_type = ['CNN','MLP','AlexNet']
    datasets = ['CIFAR10','MNIST','FashionMNIST','TinyImageNet']
    peak_epochs=[1,10,40]
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
        "-peak_epoch", type=int, required=True, choices=peak_epochs, help="select peak_epoch value"
    )
    parser.add_argument(
        "-run", type=int, required=True,choices=run_values, help="select run"
    )

    
    args = parser.parse_args()

    # Access arguments
    corrupt = args.corr
    type_network = args.model
    ds = args.dataset
    peak_epoch=args.peak_epoch
    run=args.run
    if corrupt not in corrution_prob:
        args.print_help()
    if type_network not in model_type:
        args.print_help()
    if ds not in datasets:
        args.print_help()
    if peak_epoch not in peak_epochs:
        args.print_help()
    if run not in run_values:
        args.print_help()



    t0 = time.time()
    torch.manual_seed(42)

    n_value=1

    pca_layer,_,_=mavc.layer_name(type_network)

    p_layer=pca_layer[-1]

    _,run_path,_=results_folder_MAVC(type_network,ds,run,corrupt,p_layer,
                                                   peak_epoch,once_relabel=True)

    temp_path,results_corr=temp_store(ds,type_network,corrupt,run,peak_epoch)

    test_loader=test_loading_batch(ds)


    corrupted_train,og_targets,cor_targets=cnn_create.train_loading(ds,batch_size=1,
                                                                    corrupt=corrupt)

    for model_epoch in sorted(os.listdir(run_path)):

        dummy_model=cnn_create.model_create(type_network,ds)
        file_path=f'{run_path}/{model_epoch}'
        dummy_model.load_state_dict(torch.load(file_path,map_location =dev))
        epoch=epoch_number(model_epoch)
        cnn_create.loading_saving_activations(temp_path,dummy_model,corrupted_train,
                                       test_loader,og_targets,dev,type_network)
        del dummy_model

        subspace_function(type_network,ds,temp_path,run,results_corr,n_value,epoch,dev)
        path_pca=results_corr['pca']
        mavc.MAVC_fn_neg_pca(type_network,temp_path,path_pca,run,results_corr,n_value,
                             epoch,subspace_type='corrupt')

        print(f'{model_epoch} done')
    shutil.rmtree(temp_path)
    torch.cuda.empty_cache()   
    print(f'run : {run} corrupt : {corrupt} done ')




