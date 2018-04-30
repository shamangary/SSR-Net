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
+ Dlib
+ MTCNN for demo
```
pip install mtcnn
```
## Code
Coming soon...


