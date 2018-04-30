# SSR-Net
[IJCAI18] SSR-Net: A Compact Soft Stagewise Regression Network for Age Estimation
+ A real-time age estimation model with 0.32MB.

<img src="https://github.com/shamangary/SSR-Net/blob/master/demo/TGOP_tvbs.png" height="240"/> <img src="https://github.com/shamangary/SSR-Net/blob/master/demo/the_flash_cast.png" height="240"/>

## Abstract
This paper presents a novel CNN model called Soft Stagewise Regression Network (SSR-Net) for age estimation from a single image with a compact model size. Inspired by DEX, we address age estimation by performing multi-class classification and then turning classification results into regression by calculating the expected values. SSR-Net takes a coarse-to-fine strategy and performs multi-class classification with multiple stages. Each stage is only responsible for refining the decision of the previous stage. Thus, each stage performs a task with few classes and requires few neurons, greatly reducing the model size. For addressing the quantization issue introduced by grouping ages into classes, SSR-Net assigns a dynamic range to each age class by allowing it to be shifted and scaled according to the input face image. Both the multi-stage strategy and the dynamic range are incorporated into the formulation of soft stagewise regression. A novel network architecture is proposed for carrying out soft stagewise regression. The resultant SSR-Net model is very compact and takes only **0.32 MB**. Despite of its compact size, SSR-Net’s performance approaches those of the state-of-the-art methods whose model sizes are more than 1500x larger.

## Platform
+ Keras
+ GTX-1080Ti
+ Ubuntu

## Dependencies
+ OpenCV
+ Dlib
+ MTCNN for demo
```
pip install mtcnn
```
+ MobileNet
https://github.com/fchollet/keras/blob/master/keras/applications/mobilenet.py
+ DenseNet(already in the codes)
https://github.com/titu1994/DenseNet
+ Face alignment (already in the codes)
https://github.com/xyfeng/average_portrait

## Codes

There are three different section of this project.
1.Data pre-processing
2.Training and testing
3.Video demo section
We will go through the details in the following sections.

### 1.Data pre-processing
+ Download IMDB-WIKI dataset from https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/.
+ Unzip it under './data'
+ Morph2 dataset requires application form https://www.faceaginggroup.com/morph/
+ Run the following codes for dataset pre-processing.
```
cd ./data
python TYY_IMDBWIKI_create_db.py --db imdb --output imdb.npz
python TYY_IMDBWIKI_create_db.py --db wiki --output wiki.npz
python TYY_MORPH_create_db.py --output morph_db_align.npz
```
### 2.Training and testing
For MobileNet and DenseNet:
```
cd ./training_and_testing
sh run_all.sh
```
For SSR-Net:
```
cd ./training_and_testing
sh run_ssrnet.sh
```
### 3.Video demo section
Pure CPU demo command:
```
cd ./demo
KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES='' python TYY_demo_mtcnn.py TGOP.mp4
```
+ Note: You may choose different pre-trained models. However, the morph2 dataset is much more smaller than IMDB and WIKI, the pre-trained models from morph2 may perform badly on in-the-wild images. Therefore, IMDB or WIKI pre-trained models are recommended for in-the-wild images or video demo.
