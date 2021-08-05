# requirements
conda create -n  skeleton_contrast python=3.7 anaconda
conda install pytorch==1.2.0 torchvision==0.4.0 -c pytorch
pip install tensorboard


# Skeleton-Contrastive 3D Action Representation Learning

![arch](images/teaser.png)

This repository contains the implementation of our ACM Multi Media 2021 paper:

* Skeleton-Contrastive 3D Action Representation Learning

### Link: 

[[PDF]](to add)
[[Arxiv]](to add)

### Requirements
```
 conda create -n  skeleton_contrast python=3.7 anaconda
 conda install pytorch==1.2.0 torchvision==0.4.0 -c pytorch
 pip install tensorboard

```

### Data prepreprocessing instructions
*  Download NTU RGB+D 60 and 120 skeletons and save in ./data
```
cd data_gen
python ntu_gendata.py
```
