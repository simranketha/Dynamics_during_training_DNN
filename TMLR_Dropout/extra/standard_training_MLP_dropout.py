#!/usr/bin/env python
# coding: utf-8

# In[1]:


from code_required import cnn_create
import torch
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.multiprocessing.set_sharing_strategy('file_system')
dev = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
import argparse


# In[2]:


def results_folder(type_network,ds,run,corrupt,dropout=False):  
    
    data='results/model_training'

    results=f'{data}/{type_network}_{ds}_dropout/{corrupt}/Run_{run}'
    os.makedirs(results,exist_ok=True)
    
    return results


# In[5]:


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Select model_type, datasets and corruption.")
    corrution_prob = [0.0,0.2,0.4,0.6,0.8,1.0]
    model_type = ['CNN','MLP','AlexNet']
    datasets = ['CIFAR10','MNIST','FashionMNIST','TinyImageNet']
    parser.add_argument(
        "-corr", type=float, required=True, choices=corrution_prob, help="select corruption"
    )
    parser.add_argument(
        "-model", type=str, required=True, choices=model_type, help="select model_type"
    )
    parser.add_argument(
        "-dataset", type=str, required=True, choices=datasets, help="select dataset"
    )

    args = parser.parse_args()
    corrupt = args.corr
    type_network = args.model
    ds = args.dataset
    if corrupt not in corrution_prob:
        args.print_help()
    if type_network not in model_type:
        args.print_help()
    if ds not in datasets:
        args.print_help()
        
        

    # corrupt =0.0
    # type_network = 'MLP'
    # ds = 'MNIST'
    dropout=True
    torch.manual_seed(42)
    runs = 4
    total_epochs=501
    run=1    

    network_path=cnn_create.network_path_dropout(type_network,ds)
    network_rpath=f'{network_path}/{corrupt}/Run_{run}'
    os.makedirs(network_rpath,exist_ok=True)
    _,test_loader,corrupted_train,_=cnn_create.original_dataset(type_network,ds,corrupt)
    # for run in range(1,runs):
    results_path=results_folder(type_network,ds,run,corrupt,dropout)
    epoch='initialized_model.pth'
    epoch_present=cnn_create.epochnumber(epoch)
    
    dummy_model,loss_func,optimizer,batch_size=cnn_create.model_build_drop(type_network,ds,dropout)
    dummy_model.to(dev)
    cnn_create.epoch_inference(corrupted_train,test_loader,
                                dummy_model,loss_func,
                                epoch_present,results_path,run,dev)

    torch.save(dummy_model.to('cpu').state_dict(),
                               os.path.join(network_rpath,f'initialized_model.pth'))
    
    epoch_start=epoch_present

    for epoch_present in range(epoch_start+1,total_epochs):
        dummy_model.to(dev)
        dummy_model=cnn_create.training_epoch_model(corrupted_train,dummy_model,loss_func,optimizer,dev)

        train_acc=cnn_create.epoch_inference(corrupted_train,test_loader,
                    dummy_model,loss_func,
                    epoch_present,results_path,run,dev)
        print(train_acc)
        torch.save(dummy_model.to('cpu').state_dict(),
                   os.path.join(network_rpath,f'model_{epoch_present}.pth'))
        
        print(f'{epoch_present} done')
        if train_acc==100 or train_acc==100.0:
            break
    del dummy_model
    print('run done') 

