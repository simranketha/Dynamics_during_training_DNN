import torch
import os
import pandas as pd
import numpy as np
import gc
from MASC import subspace_pytorch as subspace
from MASC import scratch_pca as sp
import copy
from torch.utils.data import TensorDataset, DataLoader,Subset
from tqdm import tqdm
import torch.nn.functional as F

def results_folder_plot(type_network,ds,run,corrupt,data_out,load_epoch=10,n=1,once_relabel=False,
                   model=False,twice_relabel=False,model_masc=False,old=False,normal=False,largest=False
                       ,shortest=False,clip=False,cliplr=False,standard=False,noisymavc=False,
                       ncm=False,bi_50_mavc_50=False,bi_25_mavc_75=False,bi_75_mavc_25=False):
    if once_relabel:
        folder_relabel='once_relabel'
        if normal:
            folder_relabel='once_relabel_normalize' 
        if largest:
            folder_relabel='once_relabel_largest' 
        if shortest:
            folder_relabel='once_relabel_shortest' 
        if clip:
            folder_relabel='once_relabel_clipped' 
        if cliplr:
            folder_relabel='once_relabel_clippedlr' 
        if noisymavc:
            folder_relabel='once_relabel_noisymavc' 
        if ncm:
            folder_relabel='once_relabel_ncm' 
        if bi_50_mavc_50:
            folder_relabel='once_relabel_b_50_mavc_50'
        if bi_25_mavc_75:
            folder_relabel='once_relabel_b_25_mavc_75' 
        if bi_75_mavc_25:
            folder_relabel='once_relabel_b_75_mavc_25' 
    else:
        folder_relabel='weight_change_final'
    
    angle='angle_results'     

    if standard:
        pred_model='standard'
    else:
        pred_model='mavc'
    
    data='MAVC_retraining_again'
    
    if standard:
        b_path=f'{data}/{angle}/{folder_relabel}/{load_epoch}'
        results=f'{b_path}/{type_network}_{ds}/{corrupt}/Run_{run}/{pred_model}'
        os.makedirs(f'{results}',exist_ok=True)

        b_path=f'/mnt/8TB/simran/PCA/{data}/{angle}/{folder_relabel}/{load_epoch}'
        run_path=f'{b_path}/{type_network}_{ds}/{corrupt}/Run_{run}/{pred_model}'
        os.makedirs(f'{run_path}',exist_ok=True)

        temp_path=f'temp_folder/{pred_model}_{angle}_{folder_relabel}_{load_epoch}_{type_network}_{ds}_{corrupt}'
        os.makedirs(f'{temp_path}',exist_ok=True)
    else:
        b_path=f'{data}/{angle}/{folder_relabel}/{load_epoch}'
        results=f'{b_path}/{type_network}_{ds}/{corrupt}/Run_{run}/{pred_model}/{data_out}'
        os.makedirs(f'{results}',exist_ok=True)

        b_path=f'/mnt/8TB/simran/PCA/{data}/{angle}/{folder_relabel}/{load_epoch}'
        run_path=f'{b_path}/{type_network}_{ds}/{corrupt}/Run_{run}/{pred_model}/{data_out}'
        os.makedirs(f'{run_path}',exist_ok=True)

        temp_path=f'temp_folder/{pred_model}_{angle}_{folder_relabel}_{load_epoch}_{type_network}_{ds}_{data_out}_{corrupt}'
        os.makedirs(f'{temp_path}',exist_ok=True)

    return results,run_path,temp_path



def results_folder_MAVC(type_network,ds,run,corrupt,data_out,load_epoch=10,once_relabel=False,normal=False,
                        largest=False,shortest=False,clip=False,cliplr=False,
                        standard=False,noisymavc=False,ncm=False,
                       bi_50_mavc_50=False,bi_25_mavc_75=False,bi_75_mavc_25=False):
    if once_relabel:
        folder_relabel='once_relabel'
        if normal:
            folder_relabel='once_relabel_normalize' 
        if largest:
            folder_relabel='once_relabel_largest' 
        if shortest:
            folder_relabel='once_relabel_shortest' 
        if clip:
            folder_relabel='once_relabel_clipped' 
        if cliplr:
            folder_relabel='once_relabel_clippedlr' 
        if noisymavc:
            folder_relabel='once_relabel_noisymavc' 
        if ncm:
            folder_relabel='once_relabel_ncm' 
        if bi_50_mavc_50:
            folder_relabel='once_relabel_b_50_mavc_50'
        if bi_25_mavc_75:
            folder_relabel='once_relabel_b_25_mavc_75' 
        if bi_75_mavc_25:
            folder_relabel='once_relabel_b_75_mavc_25' 
    else:
        folder_relabel='weight_change_final'

    angle='angle_results' 
    
    if standard:
        pred_model='standard'
    else:
        pred_model='mavc'
    
    data='MAVC_retraining_again'
    
    if standard:
        b_path=f'{data}/{angle}/{folder_relabel}/{load_epoch}'
        results=f'{b_path}/{type_network}_{ds}/{corrupt}/Run_{run}/{pred_model}'
        os.makedirs(f'{results}',exist_ok=True)

        b_path=f'/mnt/SSD2TB/simran/PCA/{data}/{angle}/{folder_relabel}/{load_epoch}'
        run_path=f'{b_path}/{type_network}_{ds}/{corrupt}/Run_{run}/{pred_model}'
        os.makedirs(f'{run_path}',exist_ok=True)

        temp_path=f'temp_folder/{pred_model}_{angle}_{folder_relabel}_{load_epoch}_{type_network}_{ds}_{corrupt}'
        os.makedirs(f'{temp_path}',exist_ok=True)
    else:
        
        b_path=f'{data}/{angle}/{folder_relabel}/{load_epoch}'
        results=f'{b_path}/{type_network}_{ds}/{corrupt}/Run_{run}/{pred_model}/{data_out}'
        os.makedirs(f'{results}',exist_ok=True)

        b_path=f'/mnt/SSD2TB/simran/PCA/{data}/{angle}/{folder_relabel}/{load_epoch}'
        run_path=f'{b_path}/{type_network}_{ds}/{corrupt}/Run_{run}/{pred_model}/{data_out}'
        os.makedirs(f'{run_path}',exist_ok=True)

        temp_path=f'temp_folder/{pred_model}_{angle}_{folder_relabel}_{load_epoch}_{type_network}_{ds}_{data_out}_{corrupt}'
        os.makedirs(f'{temp_path}',exist_ok=True)

    return results,run_path,temp_path

def pca_load_mavc(path_pca,run,epoch_present,subspace_type,p_layer,pca_neg_avg_train,num_class):
    pca_class = []
    for class_projection in range(0,num_class,1):
        folder_pca=f'{path_pca}/Run_{run}/{epoch_present}'
        file_name=f'pca_train_{subspace_type}_{p_layer}_{class_projection}.pt'
        pca_class1= torch.load(f'{folder_pca}/{file_name}').cuda()

        if pca_neg_avg_train[class_projection]==True:
            pca_class1=-pca_class1

        pca_class.append(pca_class1.to(torch.float32))

    final_tensor = torch.cat(pca_class, dim=0)

    return final_tensor

def weight_change(dummy_model,final_tensor,withbias=False,type_network='MLP'):
    if type_network=='MLP' or type_network=='AlexNet' :
        if withbias:
            if type_network=='MLP':
                dict_withbias=copy.deepcopy(dummy_model.output.state_dict())
                dict_withbias['weight']=final_tensor
                dummy_model.output.load_state_dict(dict_withbias)
                
            else:
                dict_withbias=copy.deepcopy(dummy_model.fc3.state_dict())
                dict_withbias['weight']=final_tensor
                dummy_model.fc3.load_state_dict(dict_withbias)

        else:
            if type_network=='MLP':
                dict_withoutbias=copy.deepcopy(dummy_model.output.state_dict())
                dict_withoutbias['weight']=final_tensor
                dict_withoutbias['bias']=torch.zeros(dict_withoutbias['bias'].shape[0], 
                                                     dtype=torch.float32,
                                                     device='cuda')
                dummy_model.output.load_state_dict(dict_withoutbias)
            else:
                dict_withoutbias=copy.deepcopy(dummy_model.fc3.state_dict())
                dict_withoutbias['weight']=final_tensor
                dict_withoutbias['bias']=torch.zeros(dict_withoutbias['bias'].shape[0], 
                                                     dtype=torch.float32,
                                                     device='cuda')
                dummy_model.fc3.load_state_dict(dict_withoutbias)
                
    if type_network=='CNN':
        dict_withbias=copy.deepcopy(dummy_model.fc3.state_dict())
        dict_withbias['weight']=final_tensor
        dummy_model.fc3.load_state_dict(dict_withbias)
    return dummy_model


def testing_model(model,loss_func,loader,dev):
    model.to
    model.eval()
        
    acc = 0
    test_count = 0
    total_loss = 0
    batches = 0

    with torch.no_grad():
        for i, (x_batch, y_batch) in tqdm(
            enumerate(loader), total=len(loader)):
            x_batch = x_batch.to(dev)
            y_batch = y_batch.to(dev)
            y_pred = model(x_batch)
            loss = loss_func(y_pred, y_batch.type(torch.int64))
            total_loss += loss.item()
            acc += (torch.argmax(y_pred, 1) == y_batch).float().sum().item()

            test_count += len(y_batch)
            batches = i + 1
    
    
    test_acc = (acc / test_count) * 100 if test_count > 0 else 0.0
    
    return test_acc

def epoch_inference(corrupted_train,train_loader,test_loader,
                    model,loss_func,epoch_present,results_path,dev):
    results_dic = {
      'epoch': [],
      'Train_accuracy': [],
      'Train_accuracy_org': [],
       'Test_accuracy': [],
    }

    train_acc=testing_model(model,loss_func,corrupted_train,dev)
    results_dic['Train_accuracy'].append(train_acc)
    results_dic['epoch'].append(epoch_present)
    print(f'\n  Epoch: {epoch_present},Training accuracy:{train_acc}')
       
    test_acc=testing_model(model,loss_func,train_loader,dev)
    results_dic['Train_accuracy_org'].append(test_acc)
    print(f'\n Epoch: {epoch_present} training accuracy original:{test_acc}')
    
    
    test_acc=testing_model(model,loss_func,test_loader,dev)
    results_dic['Test_accuracy'].append(test_acc)
    print(f'\n Epoch: {epoch_present} Testing accuracy:{test_acc}')
    
    file_name=f"{results_path}/Accuracy_retrain.csv"


    file_exists = os.path.isfile(file_name)
    df = pd.DataFrame(data=results_dic)

    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    df.to_csv(file_name, mode='a', header=not file_exists, index=False) 
    
    
def inference_flip_unflip(model,loss_func,loader,results_path,epoch_present,dev,flip):
    
    results_dic = {
      'epoch': [],
       'Accuracy': [],
    }

    train_acc=testing_model(model,loss_func,loader,dev)
    results_dic['Accuracy'].append(train_acc)
    results_dic['epoch'].append(epoch_present)
    print(f'\n  Epoch: {epoch_present},flip:{flip}, accuracy:{train_acc}')
    
    if flip:
        file_name=f"{results_path}/flipped_accuracy_log_final.csv"
    else: 
        file_name=f"{results_path}/unflipped_accuracy_log_final.csv"

    df = pd.DataFrame(data=results_dic)

    file_exists = os.path.isfile(file_name)

    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    df.to_csv(file_name, mode='a', header=not file_exists, index=False)  
    

def training_epoch_model(corrupted_train,model,loss_func,optimizer,dev):
    
    model.train()
    for idx, (inputs, labels) in tqdm(enumerate(corrupted_train), 
                                      total=len(corrupted_train),desc="Training"):
        inputs, labels = inputs.to(dev), labels.to(dev)
        model.zero_grad()
        output = model(inputs)
        loss = loss_func(output, labels.type(torch.int64))
        loss.backward()
        optimizer.step()
    return model

def weight_normalize(dummy_model,type_network):
    if type_network=='MLP':
        dict_withoutbias=copy.deepcopy(dummy_model.output.state_dict())
        temp_matrix=dict_withoutbias['weight']
        normalized_tensor = F.normalize(temp_matrix, p=2, dim=1)
        dict_withoutbias['weight']=normalized_tensor
        dummy_model.output.load_state_dict(dict_withoutbias)

    if type_network=='CNN' or type_network=='AlexNet':
        dict_withbias=copy.deepcopy(dummy_model.fc3.state_dict())
        temp_matrix=dict_withbias['weight']
        normalized_tensor = F.normalize(temp_matrix, p=2, dim=1)
        dict_withbias['weight']=normalized_tensor
        dummy_model.fc3.load_state_dict(dict_withbias)
    return dummy_model




def dataset_split(corrupted_train,og_targets,batch_size):
    unflipped_X, unflipped_y = [], []
    flipped_X, flipped_y = [], []

    idx = 0  # global sample index tracker

    for X_batch, y_batch in corrupted_train:
        batch_size = X_batch.size(0)
        # Convert og_targets slice to torch.Tensor, same device as y_batch
        true_labels_batch = torch.tensor(
            og_targets[idx:idx + batch_size],
            dtype=y_batch.dtype,
            device=y_batch.device
        )
        is_unflipped = y_batch == true_labels_batch

        unflipped_X.append(X_batch[is_unflipped])
        unflipped_y.append(y_batch[is_unflipped])

        flipped_X.append(X_batch[~is_unflipped])
        flipped_y.append(y_batch[~is_unflipped])

        idx += batch_size

    # Concatenate all batches
    unflipped_X = torch.cat(unflipped_X)
    unflipped_y = torch.cat(unflipped_y)
    flipped_X = torch.cat(flipped_X)
    flipped_y = torch.cat(flipped_y)

    # Create datasets and loaders
    unflipped_dataset = TensorDataset(unflipped_X, unflipped_y)
    flipped_dataset = TensorDataset(flipped_X, flipped_y)

    unflipped_loader = DataLoader(unflipped_dataset, batch_size=batch_size, shuffle=False)
    flipped_loader = DataLoader(flipped_dataset, batch_size=batch_size, shuffle=False)
    
    return unflipped_loader,flipped_loader

