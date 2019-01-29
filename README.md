# Multiscale-Super-Spectral  

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1286315.svg)](https://doi.org/10.5281/zenodo.1286315)
[![AUR](https://img.shields.io/aur/license/yaourt.svg?style=plastic)](LICENSE)   

# [Accurate Spectral Super-resolution from Single RGB Image Using Multi-scale CNN](https://arxiv.org/abs/1806.03575)

***
**Cite our work**  

> @inproceedings{yan2018accurate,  
>   title={Accurate Spectral Super-Resolution from Single RGB Image Using Multi-scale CNN},  
>   author={Yan, Yiqi and Zhang, Lei and Li, Jun and Wei, Wei and Zhang, Yanning},  
>   booktitle={Chinese Conference on Pattern Recognition and Computer Vision (PRCV)},  
>   pages={206--217},  
>   year={2018},  
>   organization={Springer}  
> } 

[**Project webpabe**](https://saoyan.github.io/posts/2018/06/09)  

***

## Pre-trained models  
Download pre-trained model from [Baidu Cloud](https://pan.baidu.com/s/1E-cJM4ftyTzbprFFGMz3dQ) or [Google Drive](https://drive.google.com/open?id=1ufcRw8P3bWSpsNB_4o88qjUldOtmqNMM)  
Place them at *TestLog*.

## Contents of this repository  
* dataset.py: data pre-processing
* model.py & train.py: our model
* model_ref.py & train_ref.py: the comparison model
* test.py: test the models and report the evaluation results
* utilities.py: some auxiliary functions

## How to run  

### 1. Dependences  
* PyTorch (<0.4)  
* OpenCV for Python3
* [HDF5 for Python](http://www.h5py.org/)
* [tensorboardX](https://github.com/lanpa/tensorboard-pytorch)
* [Matplotlib](https://matplotlib.org/)

### 2. Download data  

[NTIRE 2018 challenge on spectral reconstruction from RGB images (Track 1)](https://competitions.codalab.org/competitions/18034)  

Place the data at *data* and arrange the directories as follows:

**data/Train_Spectral/**  
--BGU_HS_00001.mat  
... ...  
--BGU_HS_00256.mat  

**data/Train_RGB/**  
--BGU_HS_00001_clean.png  
... ...  
--BGU_HS_00256_clean.png  

**data/Test_Spectral/**  
--BGU_HS_00257.mat  
--BGU_HS_00259.mat  
--BGU_HS_00261.mat  
--BGU_HS_00263.mat  
--BGU_HS_00265.mat  

**data/Test_RGB/**  
--BGU_HS_00257_clean.png  
--BGU_HS_00259_clean.png  
--BGU_HS_00261_clean.png  
--BGU_HS_00263_clean.png  
--BGU_HS_00265_clean.png  

### 3. Train our model  
If you are running the code for the first time, remember data pre-processing.  
```
python3 train.py --preprocess True
```
Otherwise, run training directly.  
```
python3 train.py
```
You may also adjust hyper-parameters such as batch size, initial learning rate, dropout rate, etc.

### 4. Train the comparison model  
Reference paper: [Learned Spectral Super-resolution](https://arxiv.org/abs/1703.09470)  
The data pre-processing is exactly the same as above, so you can run training directly.
```
python3 train_ref.py
```

### 5. Test our model  
Dropout rate: 0.2
```
python3 test.py --model Model \
  --dropout 2
```
Dropout rate: 0
```
python3 test.py --model Model \
  --dropout 0
```

### 6. Test the comparison model
Dropout rate: 0.5
```
python3 test.py --model Ref \
  --dropout 5
```
Dropout rate: 0
```
python3 test.py --model Ref \
  --dropout 0
```
