import os
import pandas as pd
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import ast
import numpy as np
plt.style.use('tableau-colorblind10')
from matplotlib.lines import Line2D

colors_blind=['#006BA4', '#FF800E', '#ABABAB', '#595959',
                 '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']
# color pallet for colors_blind
#https://viscid-hub.github.io/Viscid-docs/docs/dev/styles/tableau-colorblind10.html


mapping = {0.0:0,0.2:20, 0.4:40,0.6:60, 0.8:80,1.0:100}
def get_corrupt(val):
    return mapping.get(val, val)

mapping2 = {'MNIST':'MNIST','FashionMNIST':'Fashion-MNIST',
            'CIFAR10':'CIFAR-10','CIFAR100':'CIFAR-100',
            'TinyImageNet':'Tiny ImageNet'}

def get_datasetname(val):
    return mapping2.get(val, val)


def fun_1(train_overall,classnumber=99):
    if classnumber==99:
        max_value = list(map(max, zip(train_overall[0], train_overall[1],train_overall[2])))
        min_value = list(map(min, zip(train_overall[0], train_overall[1],train_overall[2])))
        avg_value = list(map(sum, zip(train_overall[0], train_overall[1],train_overall[2])))
        avg_value = [x/3 for x in avg_value]
    else:
        max_value = list(map(max, zip(train_overall[0][classnumber], train_overall[1][classnumber],train_overall[2][classnumber])))
        min_value = list(map(min, zip(train_overall[0][classnumber], train_overall[1][classnumber],train_overall[2][classnumber])))
        avg_value = list(map(sum, zip(train_overall[0][classnumber], train_overall[1][classnumber],train_overall[2][classnumber])))
        avg_value = [x/3 for x in avg_value]
    return max_value,min_value,avg_value


def path(network,data_type,corrupt,n,mavc=False,train=False):
    if mavc:
        if train:
            if network=='ResNet18':
                typex=f'../TMLR_ResNet18_during_training/results_velpic/40/angle_results_{n}/{data_type}_{network}/results_{corrupt}'
                type_acc=f'../../TMLR_compare/Modern_backbones/models/{data_type}_{network}/Accuracy_results/{corrupt}'
#             angle_results
            else:
                #this is test data result folder 
                typex=f'../MAVC_during_training/results/MAVC_angle_results_1pc_train/{data_type}_{network}/results_{corrupt}'
                type_acc=f'../pca_during_training_angle/angle_results/{data_type}_{network}/Accuracy_results/{corrupt}'
        else:
            typex=f'../MAVC_during_training/results/MAVC_angle_results_1pc/{data_type}_{network}/results_{corrupt}'
            type_acc=f'../pca_during_training_angle/angle_results/{data_type}_{network}/Accuracy_results/{corrupt}'
        
    else:
        if n==0.99:
            typex=f'../pca_during_training_angle/angle_results/{data_type}_{network}/results_{corrupt}'
            type_acc=f'../pca_during_training_angle/angle_results/{data_type}_{network}/Accuracy_results/{corrupt}'
        if n==1:
            typex=f'../pca_during_training_angle/angle_results_1pc/{data_type}_{network}/results_{corrupt}'
            type_acc=f'../pca_during_training_angle/angle_results/{data_type}_{network}/Accuracy_results/{corrupt}'
        if network=='ResNet18':
            typex=f'../TMLR_ResNet18_during_training/results/40/angle_results_{n}/{data_type}_{network}/results_{corrupt}'
            type_acc=f'../../TMLR_compare/Modern_backbones/models/{data_type}_{network}/Accuracy_results/{corrupt}'
    
    return typex,type_acc

def path_weight(network,data_type,corrupt):
    if network=='ResNet18':
        typex=f'../TMLR_ResNet18_during_training/results/MAVC_weight_change/{data_type}_{network}/results_{corrupt}'
        type_acc=f'../../TMLR_compare/Modern_backbones/models/{data_type}_{network}/Accuracy_results/{corrupt}'
    else:
        typex=f'../MAVC_during_training/results/MAVC_weight_change/{data_type}_{network}/results_{corrupt}'
        type_acc=f'../pca_during_training_angle/angle_results/{data_type}_{network}/Accuracy_results/{corrupt}'
       
    return typex,type_acc

def path_angle(network,data_type,corrupt):
    typex=f'../MAVC_during_training/results/MAVC_angle_weight/{data_type}_{network}/results_{corrupt}'
          
    return typex


def layer_name(type_network):
    if type_network =='AlexNet':
        data_layer=['input_layer','after_flatten',
                         'after_relu_fc1','after_relu_fc2']
        pca_layer=['input','flattern','fc1','fc2']
        num_class=200
    #cnn model
    if type_network =='CNN':
        data_layer=['input_layer','input_fc_0','output_fc_0_after_noise_relu',
                         'output_fc_1_after_noise_relu',
                         'output_fc_2_after_noise_relu']
        pca_layer=['input','flattern','fc1','fc2','fc3']
        num_class=10
    #mlp model
    if type_network =='MLP':
        
        data_layer=['input_layer','after_relu_fc1',
                         'after_relu_fc2',
                         'after_relu_fc3', 'after_relu_fc4']
        
        pca_layer=['input','fc1','fc2','fc3','fc4']
        num_class=10
    return pca_layer,data_layer,num_class

def network_layer_name(network,labels_name):
    if labels_name=='Input':
        layer='input'
        
    if labels_name=='MLP-FC1 (128)/CNN-Flat (576/1024)':
        if network =='MLP':
            layer='fc1'  
        else:
            layer='flattern'   
    if labels_name=='MLP-FC2 (512)/CNN-FC1 (250)':
        if network =='MLP':
            layer='fc2'  
        else:
            layer='fc1' 
    if labels_name=='MLP-FC3 (2048)/CNN-FC2 (250)':
        if network =='MLP':
            layer='fc3'  
        else:
            layer='fc2' 
    if labels_name=='MLP-FC4 (2048)/CNN-FC3 (250)':
        if network =='MLP':
            layer='fc4'  
        else:
            layer='fc3'        
    return layer

def layer_name_acc(type_network):
    if type_network =='ResNet18':
        data_layer=['l0','l1','l2','l3','l4','bf_last'] #
    elif type_network =='AlexNet':
        data_layer=['flattern','fc1','fc2'] #'input',
    elif type_network =='CNN':
        data_layer=['input','flattern','fc1','fc2','fc3']
    #mlp model
    elif type_network =='MLP':
        data_layer=['input','fc1','fc2','fc3','fc4']
    return data_layer


def xaxis_ticks_print(list_value):
    new_xvalues=list_value[::20]
    new_xvalues.append(list_value[-1])
    return new_xvalues

def epoch_all(path_x_values,tiny=False):
    len_x_run=[]
    last_values_xaxis=[]
    for run_type in range(1,4,1):
        path_model_num=f'{path_x_values}/Run_{run_type}/'
        len_x_run.append(len(os.listdir(path_model_num)))
        if tiny:
            last_values_xaxis.append(sorted(os.listdir(path_model_num),key=int)[-1])
    
    
    len_x=min(len_x_run)
    ind_len=len_x_run.index(len_x)                     
    path_model=f'{path_x_values}/Run_{ind_len+1}/'      
    if tiny:
        values_xaxis=sorted(os.listdir(path_model),key=int)[:-1]
        values_xaxis=values_xaxis+last_values_xaxis
    else:
        values_xaxis=sorted(os.listdir(path_model),key=int)
    
    x=[i for i in range(len_x)]
    return values_xaxis,x

def epoch_all_accuracy(path_x_values,tiny=False):
    len_x_run=[]
    last_values_xaxis=[]
    for run_type in range(1,4,1):
        path_model_num=f'{path_x_values}/angle_results/Run_{run_type}/'
        len_x_run.append(len(os.listdir(path_model_num)))
        if tiny:
            last_values_xaxis.append(sorted(os.listdir(path_model_num),key=int)[-1])
    
    
    len_x=min(len_x_run)
    ind_len=len_x_run.index(len_x)                     
    path_model=f'{path_x_values}/angle_results/Run_{ind_len+1}/'      
    if tiny:
        values_xaxis=sorted(os.listdir(path_model),key=int)[:-1]
        values_xaxis=values_xaxis+last_values_xaxis
    else:
        values_xaxis=sorted(os.listdir(path_model),key=int)
    
    x=[i for i in range(len_x)]
    return values_xaxis,x

# tiny_imagenet
def final_epoch_tiny(corrupt,run):
    if corrupt==0.0:
        epoch=[175,178,167]
    if corrupt==0.2:
        epoch=[184,194,193]
    if corrupt==0.4:
        epoch=[263,240,272]
    if corrupt==0.6:
        epoch=[449,426,428]
    if corrupt==0.8 or corrupt==1.0:
        epoch=[499,499,499]

    return epoch[run-1]+1,min(epoch)+1


## plotting over training angle between 1pc of original and 1 pc of corrupt
def path_pca_fn(data_type,network,corrupt):
    path_pca=f'angle_results_1pc/{data_type}_{network}/results_{corrupt}/pca'
    return path_pca

def value_return(angle_class,num_class):
    value_re=[]
    for i in range(num_class):
        value_re.append(float(angle_class[i].split('[')[1].split(']')[0]))

    return value_re

def funprint_angle_org_corrupt(path_pca,p_layer,num_class):
    ep_all,_=epoch_all(path_pca)
    max_vs=[]
    min_vs=[]
    avg_vs=[]
    for epoch in ep_all:
        value_re=[]
        for run in range(1,4,1):
            path_present=f'{path_pca}/Run_{run}/{epoch}/angle_results'
            angle_class=pd.read_csv(f'{path_present}/pca_{p_layer}_angle_1pc.csv')['angle_class_corr_org']
            value_re.append(value_return(angle_class,num_class))  
        max_value,min_value,avg_value=fun_1(value_re,classnumber=99)
        max_vs.append(max_value)
        min_vs.append(min_value)
        avg_vs.append(avg_value)
    avg_vs=np.array(avg_vs)
    min_vs=np.array(min_vs)
    max_vs=np.array(max_vs)

    return avg_vs,min_vs,max_vs


# plotting over training  variance captured by 1PC testing dataset
def fun_var_train_test(path_pca,p_layer,subspace_type,type_data):
    ep_all,_=epoch_all(path_pca)
    if type_data=='train':
        col_name='training_data'
    if type_data=='test':
        col_name='test_data'
    max_train=[]
    min_train=[]
    avg_train=[]

    for epoch in ep_all:
        var_train=[]
        for run in range(1,4,1):
            path_present=f'{path_pca}/Run_{run}/{epoch}'
            var_train.append(pd.read_csv(f'{path_present}/pca_{subspace_type}_{p_layer}_variance.csv')[col_name].tolist())
        
        max_value,min_value,avg_value=fun_1(var_train,classnumber=99)
        max_train.append(max_value)
        min_train.append(min_value)
        avg_train.append(avg_value)

    avg_train=np.array(avg_train)
    min_train=np.array(min_train)
    max_train=np.array(max_train)

    return avg_train,min_train,max_train
