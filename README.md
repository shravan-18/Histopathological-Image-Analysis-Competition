# Cancer Image Analysis Competition

My repository for the GDSC Cancer Image Analysis Competition. This project focuses on utilizing multiple variations of deep learning techniques for analyzing and classifying cancerous images. 

## Table of Contents
- [Overview](#overview)
- [What I've Done](#what-ive-done)
  - [VGG16 Image Classifier](#vgg16-image-classifier)
  - [CycleGAN for Image Generation](#cyclegan-for-image-generation)
  - [Graph Neural Networks (GNNs)](#graph-neural-networks-gnns)
- [Images and Results](#images-and-results)
- [References](#references)

## Overview
The goal of this project is to explore different methods for analyzing histopathological images to detect cancerous cells. The approaches include traditional convolutional neural networks (CNNs), generative adversarial networks (GANs), and graph neural networks (GNNs).

## What I've Done

### VGG16 Image Classifier
- **Sample and Augmentation**: Took a stratified sample of 500 images from the training data, performed data augmentation, and trained a VGG16 image classifier.
- **Model Training**: The VGG16 model was fine-tuned to improve its performance on the cancer image dataset.

![VGG16 Image Classification](https://github.com/shravan-18/Histopathological-Image-Analysis-Competition/blob/main/misc/images/VGG16-Architecture.png)
*Figure 1: VGG16 Image Classification Architecture*

### CycleGAN for Image Generation
- **Learning Source**: Trained a CycleGAN model using the techniques learned from [this specialization course](https://www.coursera.org/account/accomplishments/specialization/certificate/MK2MTM8QZ9NC).
- **Functionality**: The CycleGAN was used to generate cancerous images from non-cancerous ones and vice versa.
  - The code used can be found [here](https://github.com/shravan-18/Histopathological-Image-Analysis-Competition/blob/main/AMD-Cloud-Runs/CycleGAN/cycle_gan.py).
  - The results produced can be found [here](https://github.com/shravan-18/Histopathological-Image-Analysis-Competition/tree/main/AMD-Cloud-Runs/CycleGAN/CycleGAN_images).
  - The weights for the trained CycleGAN can be found [here](https://github.com/shravan-18/Histopathological-Image-Analysis-Competition/tree/main/ckpt/CycleGAN).

![CycleGAN Generated Images](https://github.com/shravan-18/Histopathological-Image-Analysis-Competition/blob/main/AMD-Cloud-Runs/CycleGAN/CycleGAN_images/fake_step_11025.jpg)
![CycleGAN Generated Images](https://github.com/shravan-18/Histopathological-Image-Analysis-Competition/blob/main/AMD-Cloud-Runs/CycleGAN/CycleGAN_images/fake_step_10500.jpg)
![CycleGAN Generated Images](https://github.com/shravan-18/Histopathological-Image-Analysis-Competition/blob/main/AMD-Cloud-Runs/CycleGAN/CycleGAN_images/fake_step_9975.jpg)
*Figure 2: CycleGAN Generated Images*

### Graph Neural Networks (GNNs)
Explored multiple methods for implementing GNNs to analyze the histopathological images. Here are the approaches:

#### Method 1: Segmentation Using Otsu's Thresholding
- **Segmentation**: Used Otsu's thresholding to segment foreground objects, specifically nuclei, from the background in grayscale images.
- **Feature Extraction**: Extracted features such as area, perimeter, and eccentricity for each nucleus, capturing structural and textural aspects.
- **Adjacency Graph Construction**: Constructed a graph where nodes represent nuclei and edges connect nearby nuclei based on a distance threshold.
- **Weights**: [Model Weights](https://github.com/shravan-18/Histopathological-Image-Analysis-Competition/blob/main/ckpt/GNN/gcn_model-otsu.pth)

#### Method 2: Segmentation Using Simple Linear Iterative Clustering (SLIC)
- **SLIC Segmentation**: Segmented images into superpixels based on color similarity, determined by number of segments, compactness, and sigma.
- **Feature Extraction**: Calculated mean color of each segment as a feature vector (not the primary method due to its limitations).
- **Adjacency Graph Construction**: Derived the graph structure from the shape of the segmented image.
- **Weights**: [Model Weights](https://github.com/shravan-18/Histopathological-Image-Analysis-Competition/blob/main/ckpt/GNN/gcn_model-slic.pth)

## Images and Results
Here are some of the images and results from the various models and techniques used in this project:

## References
- [VGG16 Paper](https://arxiv.org/abs/1409.1556)
- [CycleGAN Paper](https://arxiv.org/abs/1703.10593)
- [Otsu's Thresholding](https://ieeexplore.ieee.org/document/4310076)
- [SLIC Superpixel Segmentation](https://ieeexplore.ieee.org/document/6205760)

For more details, check out the [GitHub repository](https://github.com/shravan-18/Histopathological-Image-Analysis-Competition).
