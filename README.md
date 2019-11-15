# Towards High-Resolution Salient Object Detection
## Introduction
This package contains the source code for [Towards High-Resolution Salient Object Detection](https://drive.google.com/open?id=15o-Fel0BSyNulGoptrxfHR0t22qMHlTr), ICCV2019.
## Datasets 
HRSOD [Download](https://drive.google.com/open?id=1bmDGlkzqHoduNigi_GO4Qy9sA9sIaZcY)
DAVIS-S [Download](https://drive.google.com/open?id=1q1H7yoITLS6i2n-PhgYMIxLdjyhge5AR)
## Pre-computed Saliency Maps
Saliency maps of this paper along with compared state-of-the-art methods can be can be downloaded   
[HRSOD](https://drive.google.com/open?id=1Ch0byKXXFqE5IgP9TDMYLAe04q7z2SdD)  
[DAVIS-S](https://drive.google.com/open?id=1l7bUlc5H3Q4Z4srkpX8uf6tGD2o4JnrI)  

More saliency maps of this paper on widely used benchmarks can be downloaded   
[THUR](https://pan.baidu.com/s/1ZcI9Z9cQevdf-UFnsxihqA)  
[HKU-IS](https://pan.baidu.com/s/1dA1rsu20-rDtVTNJphQ-lw)   
[DUTS-Test](https://pan.baidu.com/s/11oyy-Y-IGpUlCXxyenzO5w)   
[PASCAL-S](https://pan.baidu.com/s/1n61Kcxlq9wnypnd1B9xdsw)   
[DUT-OMRON](https://pan.baidu.com/s/1RM84W1GqjO4_hpneYT7vJw)   
[ECSSD](https://pan.baidu.com/s/1_SF1DJu6qlMwW8TyB8w5FA)   
[HRSOD](https://pan.baidu.com/s/1-QhCX9QAmAO9zMYjShjIdA)   
[DAVIS-S](https://pan.baidu.com/s/1v8KA6vWh9P0le4hz9jedZQ)  

## Source Code
### Usage
1. Download our code [ToHR](https://pan.baidu.com/s/1auM0xI1Lgf85IQlcQzpygQ) into your computer 

2. Cd to 'HRSOD-master/caffe-master', follow the official instructions to build caffe. The code has been tested successfully on Ubuntu 14.04 with CUDA 8.0.

3. Make caffe & matcaffe  

        make all -j  
        make matcaffe -j  
    
4. Download pretrained caffemodel from [baidu yun](https://pan.baidu.com/s/1nATblFyypAx_3U5kAnCevg#list/path=%2F) and put the file under the root directory HRSOD-master/.

5. Change parameters in init_iccv19_demo.m and then run test_iccv19_demo.m to get the saliency maps. The results will be saved in HRSOD-master/results/.
## Citation
If you find our datasets useful in your research, please consider citing:

    @InProceedings{Zeng_2019_ICCV,
      author = {Zeng, Yi and Zhang, Pingping and Zhang, Jianming and Lin, Zhe and Lu, Huchuan},
      title = {Towards High-Resolution Salient Object Detection},
      booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
      month = {October},
      year = {2019}
    }
