import torch
import os
import pandas as pd
import numpy as np
import gc
from code_required import subspace_pytorch as subspace
from code_required import scratch_pca as sp
import pickle
import time

def angle(X_output_layer, X_pca_new):
    layer_angle = []
    for layer in range(len(X_pca_new)):
        dot_product = torch.diag(torch.matmul(X_output_layer[layer], X_pca_new[layer].T))
        norm_X_output = torch.norm(X_output_layer[layer], dim=1)
        norm_X_pca_new = torch.norm(X_pca_new[layer], dim=1)
        cos_theta = dot_product / (norm_X_output * norm_X_pca_new)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # Ensure cos_theta is within the valid range
        layer_angle.append(torch.acos(cos_theta) * (180.0 / torch.pi))
    return torch.stack(layer_angle)

def angle_layer(X_output_layer, X_pca_new):
    dot_product = torch.diag(torch.matmul(X_output_layer, X_pca_new.T))
    norm_X_output = torch.norm(X_output_layer, dim=1)
    norm_X_pca_new = torch.norm(X_pca_new, dim=1)
    cos_theta = dot_product / (norm_X_output * norm_X_pca_new)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    layer_angle = torch.acos(cos_theta) * (180.0 / torch.pi)
    return layer_angle

def least_class(class_angle, num_images, number_class=10):
    temp_final = []
    for layer in range(class_angle.shape[0]):
        temp = [[torch.tensor(200000.0), torch.tensor(200000.0)] for _ in range(num_images)]
        for j in range(num_images):
            for k in range(number_class):
                num = k * num_images + j
                if class_angle[layer][num] < temp[j][1]:
                    temp[j] = [k, class_angle[layer][num]]
        temp_final.append(temp)
    return temp_final




def least_class_layer(class_angle, num_images, number_class=10):
#     tin = time.time()
    # Reshape class_angle to (number_class, num_images) for efficient comparison
    class_angle = class_angle.view(number_class, num_images)
    
    # Find the minimum angle and its corresponding class index for each image
    min_values, min_indices = torch.min(class_angle, dim=0)
    
    # Stack the results together (class index and the corresponding minimum angle)
    result = torch.stack([min_indices.float(), min_values], dim=1)
    return result

def accuracy_angle(y_pred, y):
    score_f = []
    for layer in range(len(y_pred)):
        score_l = 0
        for image_i in range(len(y)):
            if y_pred[layer][image_i][0] == y[image_i]:
                score_l += 1
        score_f.append(round(score_l / len(y), 4))
    return score_f
def accuracy_angle_layer(y_pred, y):
#     tin = time.time()
    score_l = 0
    for image_i in range(len(y)):
        if y_pred[image_i][0] == y[image_i]:
            score_l += 1
    score_l = round(score_l / len(y), 4)
    return score_l
def acc_class_angle(y_pred, y, number_class=10):
#     tin = time.time()
    score_f = []
    for class_i in range(number_class):
        score_c = []
        for layer in range(len(y_pred)):
            score_l = 0
            count = 0
            for image_i in range(len(y)):
                if y[image_i] == class_i:
                    count += 1
                    if y_pred[layer][image_i][0] == y[image_i]:
                        score_l += 1
            if count > 0:
                score_c.append(round(score_l / count, 4))
        score_f.append(score_c)
    return score_f

def acc_class_angle_layer(y_pred, y, number_class=10):
#     tin = time.time()
    score_f = []
    for class_i in range(number_class):
        score_l = 0
        count = 0
        for image_i in range(len(y)):
            if y[image_i] == class_i:
                count += 1
                if y_pred[image_i][0] == y[image_i]:
                    score_l += 1
        if count > 0:
            score_f.append(round(score_l / count, 4))
            
#     tout = time.time()
#     total = tout-tin
#     print(f'total time taken {total} sec inside acc_class_angle_layer function')
    return score_f


def acc_class_angle_layer2(y_pred, y, number_class=10):
#     tin = time.time()
    # Assuming y_pred is a 2D tensor with predictions in the first column
    y_pred = y_pred[:, 0]  # Extract predicted labels

    # Initialize a tensor to store class-wise accuracies
    score_f = torch.zeros(number_class, dtype=torch.float32, device=y.device)

    # Calculate accuracy for each class
    for class_i in range(number_class):
        mask = y == class_i  # Mask for samples belonging to the current class
        total = mask.sum().item()  # Total samples in this class
        if total > 0:
            correct = (y_pred[mask] == y[mask]).sum().float()  # Correct predictions for this class
            accuracy = correct / total  # Calculate accuracy
            score_f[class_i] = torch.round(accuracy * 10000) / 10000  # Round to 4 decimals

#     tout = time.time()
#     total = tout-tin
#     print(f'total time taken {total} sec inside acc_class_angle_layer2 function')
    return score_f.detach().cpu().tolist()

def layer_names(type_network,ds):
    #cnn model
    if type_network =='CNN':        
        data_layer_name=['input_fc_0',
                         'output_fc_0_after_noise_relu','output_fc_0_after_dropout',
                         'output_fc_1_after_noise_relu','output_fc_1_after_dropout',
                         'output_fc_2_after_noise_relu','output_fc_2_after_dropout','y_value_corrupted']
#         'input_layer',
        data_output_name=['flattern',
            'fc1','fc1_dropout','fc2','fc2_dropout','fc3','fc3_dropout']
#     'input',
        num_class=10
    return data_layer_name,data_output_name,num_class


def subspace_creation(temp_path,n,path2_pca,data,data_out,num_class,subspace_type):
    #training data
    with open(f'{temp_path}/{data}_train.pkl', 'rb') as file: 
        myvar = pickle.load(file)  

    layer_output = [torch.tensor(np.array(myvar[values])).cuda() for values in range(len(myvar))]
    layer_output = torch.stack(layer_output).cuda()

    #corrupt labels
    if subspace_type=='corrupt':
        with open(f'{temp_path}/y_value_corrupted_train.pkl', 'rb') as file: 
            myvar = pickle.load(file) 

    cor_targets = [torch.tensor(np.array(myvar[values])).cuda() for values in range(len(myvar))]
    cor_targets = torch.stack(cor_targets).cuda()

    
    obj=subspace.classwise_data(layer_output,cor_targets,number_class=num_class)
    X_pca_class=[]
    X_pca_values=[]

    for class_projection in range(0,num_class,1):
        X_output_layer = torch.tensor(obj[f'class{class_projection}']).cuda()
        X_output_layer_add = subspace.Added_data_layer(X_output_layer)
        
        if n>=1:
            pca_vectors,percent_var=sp.PCA_new_layer(X_output_layer_add.cpu(), n)
            X_pca_values.append(percent_var)
        else:
            pca_vectors=sp.PCA_new_layer(X_output_layer_add.cpu(), n)

        X_ppca = torch.from_numpy(pca_vectors).cuda().to(torch.float64)

        X_pca_class.append(X_ppca.shape[0])

        torch.save(X_ppca.cpu(),f"{path2_pca}/pca_train_{subspace_type}_{data_out}_{class_projection}.pt")       


    del layer_output,cor_targets,obj
    gc.collect()
    torch.cuda.empty_cache() 
    
def layer_names_ResNet18(type_network, ds, layer_number):
    layer_map = {
        0: ("after_layer_0", "l0"),
        1: ("after_layer_0_1", "l0_1"),
        2: ("after_layer_0_2", "l0_2"),
        3: ("after_layer_0_3", "l0_3"),
        4: ("after_layer_1", "l1"),
        5: ("after_layer_2", "l2"),
        6: ("after_layer_3", "l3"),
        7: ("after_layer_4", "l4"),
        8: ("before_fc", "bf_last"),
        9: ("after_dropout", "after_dropout"),
    }

    layer_name, out_name = layer_map[layer_number]
    return [layer_name, "y_value_corrupted"], [out_name], 10

def subspace_creations_all_layers(type_network,ds,temp_path,run,results,n,epoch_present,dev,subspace_type,layer_name=None):
    if type_network=='ResNet18':
        data_layer_name,data_output_name,num_class=layer_names_ResNet18(type_network,ds,layer_name)
    else:
        data_layer_name,data_output_name,num_class=layer_names(type_network,ds)
    
    results_angle1=results['angle_1']
    results_pca=results['pca']
    batch_size=100
    epoch=epoch_present


    path2_pca=f'{results_pca}/Run_{run}/{epoch}' 
    os.makedirs(path2_pca,exist_ok=True)

    with torch.no_grad(): 
        for data,data_out in zip(data_layer_name,data_output_name):
            subspace_creation(temp_path,n,path2_pca,data,data_out,
                                          num_class,subspace_type)       
            
                
 