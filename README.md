# On the Dynamics \& Transferability of Latent Generalization during Memorization



## Contents

- [Abstract](#abstract)
- [Overview](#overview)
- [Installation Guide](#installation-guide)
- [Citation](#citation)
- [License](./LICENSE)

# Abstract

Deep networks have been known to have extraordinary generalization abilities, via mechanisms that aren't yet well understood. It is also known that upon shuffling labels in the training data to varying degrees, deep networks, trained with standard methods, can still achieve perfect or high accuracy on this corrupted training data. This phenomenon is called {\em memorization}, and typically comes at the cost of poorer generalization to true labels. Recent work has demonstrated, surprisingly, that the internal representations of such models retain significantly better latent generalization abilities than is directly apparent from the model. In  particular, it has been shown that such latent generalization can be recovered via simple probes (called MASC probes) on the layer-wise representations of the model. However, the origin and dynamics over training of this latent generalization during memorization is not well understood. Here, we track the training dynamics, empirically, and find that latent generalization abilities largely peak early in training, with model generalization. Next, we investigate whether the specific nature of the MASC probe is critical for our ability to extract latent generalization from the model's layerwise outputs. To this end, we first examine the mathematical structure of the MASC probe and show that it is a quadratic classifier, i.e. is non-linear. This brings up the possibility that this latent generalization is not linearly decodable, and that the model is fundamentally incapable of generalizing as well as the MASC probe, given corrupted training data. To investigate this, we designed a new linear probe for this setting, and find, surprisingly, that it has superior generalization performance in comparison to the quadratic probe, in most, but not all cases. Given that latent generalization is linearly decodable in most cases, we ask if there exists a way to leverage probes on layerwise representations, to directly edit model weights to immediately manifest the latent generalization to model generalization. To this end, we devise a way to transfer the latent generalization present in last-layer representations to the model using the new linear probe. This immediately endows such models with improved generalization in most cases, i.e. without additional training. We also explore training dynamics, when the aforementioned weight editing is done midway during training. Our findings provide a more detailed account of the rich dynamics of latent generalization during memorization, provide clarifying explication on the specific role of the probe in latent generalization, as well as demonstrate the means to leverage this understanding to directly transfer this generalization to the model.

# Overview
This repository provides python implementation of the algorithms described in the paper.

We have used code from [Decoding Generalization from Memorization in Deep Neural Networks](https://github.com/simranketha/MASC_DNN) paper for training the models with different corruption degrees and the implementation of Minimum Angle Subspace Classifier (MASC).

* Experiment where MASC is used on layer wise outputs of the network during training is available in MASC_during_training folder.
* Experiment where VeLPIC is used on layer wise outputs of the network during training is available in VeLPIC_during_training folder.
* Experiment on only ResNet-18 model trained on CIFAR-10 is available in TMLR_ResNet18_during_training folder.
* Experiment for comparing with logistic regression probe is available in BASELINES folder. 
* Experiment for comparing models with dropout is available in TMLR_Dropout folder.
* Experiment where we used changed the weights for the model with VeLPIC vectors on the last layer at 40th epoch and then performed standard training are  available in training_using_VeLPIC folder.
* Plotting codes are  available in TMLR_plotting_during folder.
* Every folder has its own instruction.txt file for clarification on running the codes.


# Installation Guide


 * Clone or download the current version from project page in github or from git command line:
```
git clone git@github.com:simranketha/Dynamics_during_training_DNN.git
```

 * Install the related packages:

```
conda create --name pytorch_new python==3.8.10
conda init bash
source .bashrc
conda activate pytorch_new 
pip install scikit-learn tqdm
pip install pandas

for gpu-version: pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 
for cpu-version: pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu 
```



