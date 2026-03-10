#!/usr/bin/env python
# coding: utf-8

# In[1]:


from CNN_code import cnn_create,loading_saving_data,logistic_regression_probe

import os
import torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.multiprocessing.set_sharing_strategy('file_system')
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import pandas as pd
import argparse
import shutil
import time

# In[2]:


def linear_probe_logistic_regression(type_network,run,results,epoch_present,dev):
    print(type_network)
    pca_layer,data_layer,_=loading_saving_data.layer_name(type_network)

    #results folders 
    path2=f'{results}/Run_{run}/{epoch_present}'
    os.makedirs(path2,exist_ok=True)

    for p_layer,d_layer in zip(pca_layer,data_layer):
        print(p_layer,d_layer)
        train_x,train_y=loading_saving_data.data_loading_train(temp_path,d_layer)
   
        test_x,original_testy=loading_saving_data.data_loading_test(temp_path,d_layer)
        train_loader,test_loader=logistic_regression_probe.create_data_loader(train_x,train_y,test_x,original_testy) 

        #data setup  
        input_dim = train_x.shape[1]
        num_classes = len(torch.unique(train_y))
        #model setup
        model = logistic_regression_probe.LogisticRegression(input_dim, num_classes).to(dev)

        logistic_regression_probe.training_probe(model,train_loader,dev)

        test_acc=logistic_regression_probe.inference(model,test_loader,dev)

        filename='acc_overall_test'
        d = {'acc_overall':[test_acc]}
        df1 = pd.DataFrame(data=d)
        df1.to_csv(f"{path2}/layer_{p_layer}_{filename}.csv") 
def linear_probe_logistic_regression_layerwise(type_network,run,results,epoch_present,dev,layernumber):
    print(type_network)
    pca_layer,data_layer,_=loading_saving_data.layer_name(type_network)
    
    #results folders 
    path2=f'{results}/Run_{run}/{epoch_present}'
    os.makedirs(path2,exist_ok=True)
    p_layer=pca_layer[layernumber]
    d_layer=data_layer[layernumber]
#     for p_layer,d_layer in zip(pca_layer,data_layer):
    print(p_layer,d_layer)
    train_x,train_y=loading_saving_data.data_loading_train(temp_path,d_layer)

    test_x,original_testy=loading_saving_data.data_loading_test(temp_path,d_layer)
    train_loader,test_loader=logistic_regression_probe.create_data_loader(train_x,train_y,test_x,original_testy) 

    #data setup  
    input_dim = train_x.shape[1]
    num_classes = len(torch.unique(train_y))
    #model setup
    model = logistic_regression_probe.LogisticRegression(input_dim, num_classes).to(dev)

    logistic_regression_probe.training_probe(model,train_loader,dev)

    test_acc=logistic_regression_probe.inference(model,test_loader,dev)

    filename='acc_overall_test'
    d = {'acc_overall':[test_acc]}
    df1 = pd.DataFrame(data=d)
    df1.to_csv(f"{path2}/layer_{p_layer}_{filename}.csv") 

# In[ ]:


if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(description="Select model_type, datasets and corruption.")
    
    corrution_prob = [0.0,0.2,0.4,0.6,0.8,1.0]
    model_type = ['CNN','MLP','AlexNet','ResNet18']
    datasets = ['CIFAR10','MNIST','FashionMNIST','TinyImageNet']
    layer_numbers=[0,1,2,3,4,5,6,7,8,999]
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
#     parser.add_argument(
#         "-run", type=int, required=True, choices=run_values, help="select run"
#     )
    parser.add_argument(
        "-layer_number", type=int, required=False, default=8, choices=layer_numbers, help="select layer_number"
    )
    args = parser.parse_args()

#     # Access arguments
    corrupt = args.corr
    type_network = args.model
    ds = args.dataset
#     run=args.run
    layer_number=args.layer_number
    
    if corrupt not in corrution_prob:
        args.print_help()
    if type_network not in model_type:
        args.print_help()
    if ds not in datasets:
        args.print_help()
#     if run not in run_values:
#         args.print_help()
    if layer_number not in layer_numbers:
        args.print_help()
    
    torch.manual_seed(42)
    t0 = time.time()
    runs = 4
    # for corrupt in corrution_prob:

    results_folder=f'results/logistic_regression/{ds}_{type_network}'
    results=loading_saving_data.results_fol_name(results_folder,corrupt)

    network_path,test_loader,corrupted_train,og_targets=cnn_create.original_dataset(type_network,ds,corrupt)

    for run in range(1,runs): 

        #path for temprary activation storage
        temp_path = f'tempfolder/logistic_regression/Network_data_{corrupt}_{ds}_{type_network}_{run}_{layer_number}'
        os.makedirs(temp_path,exist_ok=True)

        model_numbers=cnn_create.selected_model_list(network_path,corrupt,run)
        for model_epoch in model_numbers:  
            print(model_epoch)
            dummy_model=cnn_create.model_create(type_network,ds)

            dummy_model.load_state_dict(torch.load(f'{network_path}/{corrupt}/Run_{run}/{model_epoch}',
                                                   map_location =dev))

            if type_network=='ResNet18':
                cnn_create.ResNet18_loading_saving_activations_layerwise(temp_path,dummy_model,corrupted_train,
                                                   test_loader,og_targets,dev,type_network,layer_number)
                del dummy_model
                epoch_present=cnn_create.epochnumber(model_epoch)
                linear_probe_logistic_regression_layerwise(type_network,run,results,epoch_present,dev,layer_number)

            else:
                cnn_create.loading_saving_activations(temp_path,dummy_model,
                                              corrupted_train,test_loader, 
                                              og_targets,dev,type_network)
                del dummy_model
                epoch_present=cnn_create.epochnumber(model_epoch)

                linear_probe_logistic_regression(type_network,run,results,epoch_present,dev)
                print(f"model {model_epoch} done")

        print(f"run {run} done")

        shutil.rmtree(temp_path)
    print(f'corrupt {corrupt} done')
    t1=time.time()
    total1 = t1-t0
    print(f'total time taken {total1} sec')

