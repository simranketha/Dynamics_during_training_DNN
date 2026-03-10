import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
plt.style.use('tableau-colorblind10')
# /Run_1/mavc/{}
def path_load(name,ds,net,corr):
    
    if name=='masc':
        file_name=f'../pca_during_training_angle/angle_results/{ds}_{net}/results_{corr}/angle_results'
    if name=='velpic':
        file_name=f'../MAVC_during_training/results/MAVC_angle_results_1pc_train/{ds}_{net}/results_{corr}/angle_results'
    if name=='retrain':   
        base_retrain_file=f'../RETRAINING_WITHOUT_RELABELING/MAVC_retraining_again/angle_results/once_relabel/1'
        file_name=f'{base_retrain_file}/{net}_{ds}/{corr}'
    return file_name
mapping = {0.0:0,0.2:20,
        0.4:40,0.6:60,
        0.8:80,1.0:100}
def get_value(val):
    return mapping.get(val, val)

def bars(corruption, dss, networks):
    types = ['MASC', 'VeLPIC', 'Retrain', 'Max_Retrain']
    runs = ["Run_1", "Run_2", "Run_3"]

    fig, axs = plt.subplots(2, 2, figsize=(18, 15))
    axs = axs.flatten()

    for i, corr in enumerate(corruption):
        all_data = {t: {f'{net}-{ds}': [] for ds, net in zip(dss, networks)} for t in types}
        print(f"corruption: {corr}")

        for ds,net in zip(dss,networks):
            if ds=='TinyImageNet':
                runs = ["Run_1"]
            else:
                runs = ["Run_1", "Run_2", "Run_3"]
                
            print(f"network: {net}, dataset: {ds}")
            
            masc = list()
            velpic = list()
            retrain = list()
            max_retrain = list()
            
            for run in runs:
                temp_masc = list()
                temp_velpic = list()
                masc_path=path_load('masc',ds,net,corr)
                masc_path = os.path.join(masc_path, run)
            
                masc_dirs = [ int(f) for f in os.listdir(masc_path) if os.path.isdir(os.path.join(masc_path, f)) and '.' not in f]
                if 100 in masc_dirs:
                    masc_path = os.path.join(masc_path, '100')
                else:
                    masc_path = os.path.join(masc_path, str(max(masc_dirs)))
                
                print(f'masc_path: {masc_path}')
                
                masc_csvs = [ f for f in os.listdir(masc_path)
                              if f.endswith(".csv") and 'acc_overall_test' in f and  f != "layer_input_acc_overall_test.csv" and f != "Accuracy_retrain.csv" ]
                
                velpic_path=path_load('velpic',ds,net,corr)
                velpic_path = os.path.join(velpic_path, run)
            
                velpic_dirs = [ int(f) for f in os.listdir(velpic_path) if os.path.isdir(os.path.join(velpic_path, f)) and '.' not in f]
                if 100 in velpic_dirs:
                    velpic_path = os.path.join(velpic_path, '100')
                else:
                    velpic_path = os.path.join(velpic_path, str(max(velpic_dirs)))
                
                print(f'velpic_path: {velpic_path}')
                
                velpic_csvs = [ f for f in os.listdir(velpic_path)
                                if f.endswith(".csv") and 'acc_overall_test' in f and f != "layer_input_acc_overall_test.csv" and f != "Accuracy_retrain.csv" ]

                print(f"masc_csvs: {masc_csvs}")
                print(f"velpic_csvs: {velpic_csvs}")
                
                temp_path = masc_path
                for files in masc_csvs:
                    masc_path = os.path.join(temp_path, files)
                    temp_masc.append(pd.read_csv(masc_path)["acc_overall"].iloc[-1]*100)
            
                masc.append(max(temp_masc))
            
                temp_path = velpic_path
                for files in velpic_csvs:
                    velpic_path = os.path.join(temp_path, files)
                    temp_velpic.append(pd.read_csv(velpic_path)["acc_overall"].iloc[-1]*100)
            
            
                velpic.append(max(temp_velpic))
            
            
                retrain_path=path_load('retrain',ds,net,corr)
                    
                retrain_path = os.path.join(retrain_path, run,'mavc')
                for direc in os.listdir(retrain_path):
                    if os.path.isdir(os.path.join(retrain_path,direc)):
                        retrain_path = os.path.join(retrain_path,direc, "Accuracy_retrain.csv")
                
                print(f"retrain_path: {retrain_path}")
                retrain_data = pd.read_csv(retrain_path)["Test_accuracy"]
                retrain.append(retrain_data.iloc[-1])
                max_retrain.append(retrain_data.max())
 

            # Append to data structure
            all_data['MASC'][f'{net}-{ds}'].extend(masc)
            all_data['VeLPIC'][f'{net}-{ds}'].extend(velpic)
            all_data['Retrain'][f'{net}-{ds}'].extend(retrain)
            all_data['Max_Retrain'][f'{net}-{ds}'].extend(max_retrain)

        ax = axs[i]

        # Plotting
        x = np.arange(len(networks))
        width = 0.2
        n_types = len(types)
        total_group_width = width * n_types
        offsets = np.linspace(-total_group_width/2 + width/2, total_group_width/2 - width/2, n_types)

        for idx, t in enumerate(types):
            means = []
            errors = [[], []]  # [lower, upper]

            for ds,net in zip(dss,networks):
                values = all_data[t][f'{net}-{ds}']
                mean = np.mean(values)
                min_val = np.min(values)
                max_val = np.max(values)

                means.append(mean)
                errors[0].append(mean - min_val)  # lower error
                errors[1].append(max_val - mean)  # upper error

            ax.bar(x + offsets[idx], means, width, yerr=errors, capsize=5, label=t)

        ax.set_ylim([0,100])
        ax.set_xlabel(f'Model-dataset',fontsize=14)
        ax.set_ylabel('Test Accuracy (%)',fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(all_data['MASC'].keys(), rotation=20)
        ax.legend()
        ax.set_title(f'{get_value(corr)}% corruption degree',fontsize=18)
        # ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'AAAI_PAPER_PLOTS/compare_accuracy_subplot.pdf',format="pdf", bbox_inches="tight")
        
    plt.show()
    plt.close()
    

if __name__ == "__main__":
    corruption=[0.2,0.4,0.6,0.8]
    dss=["MNIST","CIFAR10","MNIST","FashionMNIST","CIFAR10","TinyImageNet"]
    networks=["MLP","MLP","CNN","CNN","CNN","AlexNet"]
    bars(corruption, dss, networks)

