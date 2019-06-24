# SSR-Net
**[IJCAI18] SSR-Net: A Compact Soft Stagewise Regression Network for Age Estimation**
+ A real-time age estimation model (0.32MB)
+ Gender regression is also added!
+ Megaage-Asian implementation is provided in https://github.com/b02901145/SSR-Net_megaage-asian
+ Core ML model (0.17MB) is provided in https://github.com/shamangary/Keras-to-coreml-multiple-inputs-example

**Code Author: Tsun-Yi Yang**

**Last update: 2018/08/06 (Adding megaage_asian training. Typo fixed.)**


<img src="https://media.giphy.com/media/ygBDe4FIU4Cybbfh2N/giphy.gif" height="240"/> <img src="https://media.giphy.com/media/bZvHMOp2hBsusr96fa/giphy.gif" height="240"/> 

<img src="https://github.com/shamangary/SSR-Net/blob/master/demo/TGOP_tvbs.png" height="240"/> <img src="https://github.com/shamangary/SSR-Net/blob/master/demo/the_flash_cast.png" height="240"/>

<img src="https://github.com/shamangary/SSR-Net/blob/master/table1.png" height="240"/>


### Real-time webcam demo

<img src="https://media.giphy.com/media/AhXTjtGO9tnsyi2k6Q/giphy.gif" height="240"/> <img src="https://github.com/shamangary/SSR-Net/blob/master/age_gender_demo.png" height="240"/>


## Paper

### PDF
https://github.com/shamangary/SSR-Net/blob/master/ijcai18_ssrnet_pdfa_2b.pdf

### Paper authors
**[Tsun-Yi Yang](http://shamangary.logdown.com/), [Yi-Husan Huang](https://github.com/b02901145), [Yen-Yu Lin](https://www.citi.sinica.edu.tw/pages/yylin/index_zh.html), [Pi-Cheng Hsiu](https://www.citi.sinica.edu.tw/pages/pchsiu/index_en.html), and [Yung-Yu Chuang](https://www.csie.ntu.edu.tw/~cyy/)**

## Abstract
This paper presents a novel CNN model called Soft Stagewise Regression Network (SSR-Net) for age estimation from a single image with compact model size. Inspired by DEX, we address age estimation by performing multi-class classification and then turning classification results into regression by calculating the expected values. SSR-Net takes a coarse-to-fine strategy and performs multi-class classification with multiple stages. Each stage is only responsible for refining the decision of the previous stage. Thus, each stage performs a task with few classes and requires few neurons, greatly reducing the model size. For addressing the quantization issue introduced by grouping ages into classes, SSR-Net assigns a dynamic range to each age class by allowing it to be shifted and scaled according to the input face image. Both the multi-stage strategy and the dynamic range are incorporated into the formulation of soft stagewise regression. A novel network architecture is proposed for carrying out soft stagewise regression. The resultant SSR-Net model is very compact and takes only **0.32 MB**. Despite its compact size, SSR-Net’s performance approaches those of the state-of-the-art methods whose model sizes are more than 1500x larger.

## Platform
+ Keras
+ Tensorflow
+ GTX-1080Ti
+ Ubuntu

## Dependencies
+ A guide for most dependencies. (in Chinese)
http://shamangary.logdown.com/posts/3009851
+ Anaconda
+ OpenCV
+ dlib
+ MTCNN (for running demo)
+ MobileNet (already included in this repository)
https://github.com/fchollet/keras/blob/master/keras/applications/mobilenet.py
+ DenseNet (already included in this repository)
https://github.com/titu1994/DenseNet
+ Face alignment (already included in this repository)
https://github.com/xyfeng/average_portrait
+ Others
```
conda install -c conda-forge moviepy
conda install -c cogsci pygame
conda install -c conda-forge requests
conda install -c conda-forge pytables
```
To install all the dependencies, run the following command from the root folder of the project (*`$SSR_NET_ROOT`*) which contains `requirements.txt`:
```
pip install -r requirements.txt
```

## Codes

This project is broadly divided into the following four sections: 
1. Data pre-processing
2. Training and testing
3. Running the Video Demo
4. Extension


### 1. Data pre-processing
In this repository, we use the IMDB, WIKI and Morph2 datasets to train and test our model.
Follow the following steps to set up the data for training the model
+ Download the IMDB-WIKI dataset (face only) from https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
+ Download the Morph2 dataset (requires filling an application form) https://www.faceaginggroup.com/morph/
+ Unzip the data under `*$SSR_NET_ROOT*/data`
+ Run the following commands to pre-process the datasets
```
cd ./data
python TYY_IMDBWIKI_create_db.py --db imdb --output imdb_db.npz
python TYY_IMDBWIKI_create_db.py --db wiki --output wiki_db.npz
python TYY_MORPH_create_db.py --output morph2_db_align.npz
```

### 2. Training and Testing

The experiments are done by randomly choosing 80% of the dataset as training and 20% of the dataset as validation (or testing). The detailed settings of each dataset are in the paper.

Run the following commands to train the SSR-Net model:
```
cd ./training_and_testing
sh run_ssrnet.sh
```

For comparison purposes, you can run the following commands to train MobileNet and DenseNet models:
```
cd ./training_and_testing
sh run_all.sh
```
<img src="https://github.com/shamangary/SSR-Net/blob/master/merge_val_morph2.png" height="300"/>

> **Note:** We provide several different hyper-parameter combinations in this repository. If you only want a single hyper-parameter set, please alter the command inside `*$SSR_NET_ROOT*/training_and_testing/run_ssrnet.sh`.

**Plotting the results:**
Let's assume that after training the model using the IMDB dataset, you want to plot a curve to show the results.
To do this, copy `plot.sh`, `ssrnet_plot.sh`, and `plot_reg.py` from `training_and_testing` into the `imdb_models` folder.
Running the following commands should plot the results of the training process.
```
sh plot.sh
sh ssrnet_plot.sh
```
>**Note:** If you face any permission issues while running any `.sh` files, just run the following command with the filename of the shell script before running the script itself:
```
chmod +x <filename.sh>
```

### 3. Running the Video Demo
To run the demo purely on CPU use the following command:
```
cd ./demo
KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES='' python TYY_demo_mtcnn.py TGOP.mp4

# Or you can use

KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES='' python TYY_demo_mtcnn.py TGOP.mp4 '3'
```
> **Note:** You may choose different pre-trained models. The Morph2 dataset is under a well-controlled environment and it is much smaller than the IMDB and WIKI datasets, therefore the pre-trained models from Morph2 may perform poorly on "in-the-wild" images. Hence, IMDB or WIKI pre-trained models are recommended for "in-the-wild images" and video demo.

+ We use DLib face detection and alignment in the previous experimental section since the face data is well organized. However, DLib cannot provide satisfactory face detection results for "in-the-wild" video data. Therefore, we use MTCNN as the detection process in this demo section.

**Running the Real-time Webcam Demo:**

Considering the face detection process (MTCNN or Dlib) is not fast enough for a real-time demo, we show a real-time webcam demo by using LBP Face Detector.

To run the Real-time Webcam Demo, run the following commands:
```
cd ./demo
KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES='' python TYY_demo_ssrnet_lbp_webcam.py
```

> **Note:** The covered region of face detection is different when you use MTCNN, Dlib, or LBP. You should choose a similar size between the inference and the training.

>Also, the pre-trained models are mainly for the evaluation of datasets. They are not really for real-world images. You should always retrain the model with your own dataset. In webcam demo, we found that the Morph2 pre-trained model actually performs better than the WIKI pre-trained model. The discussion will be included in our future work.

> If you are Asian, you might want to use the megaage_asian pre-trained model.

> The Morph2 pre-trained model is good for webcam but the gender model is overfitted and not practical.

### 4. Extension

**Training the gender model**

We can reformulate the binary classification problem into a regression problem, and SSR-Net can be used to predict the confidence.
As an example, we provide a gender regression demo in this repository as an extension of the project.

Training the gender network:
```
cd ./training_and_testing
sh run_ssrnet_gender.sh
```
>**Note:** The score can be between [0,1] and the 'V' inside SSR-Net can be changed into 1 for general propose regression.


## Third Party Re-Implementation
|MXNET| https://github.com/wayen820/gender_age_estimation_mxnet|
|-----|---------------------|
