# How to Perform Face Detection with Deep Learning

Face detection is a computer vision problem that involves finding faces in photos.

It is a trivial problem for humans to solve and has been solved reasonably well by classical feature-based techniques, such as the cascade classifier. More recently deep learning methods have achieved state-of-the-art results on standard benchmark face detection datasets. One example is the Multi-task Cascade Convolutional Neural Network, or MTCNN for short.
In this tutorial, we will discover how to perform face detection in Python using classical and deep learning models.

After completing this tutorial, we will know:
 - Face detection is a non-trivial computer vision problem for identifying and localizing faces in images.
 - Face detection can be performed using the classical feature-based cascade classifier using the OpenCV library.
 - State-of-the-art face detection can be achieved using a Multi-task Cascade CNN via the MTCNN library.

## Tutorial Overview

This tutorial is divided into four parts; they are:
- Face Detection
- Test Photographs
- Face Detection With OpenCV
- Face Detection With Deep Learning

## Implementation

Perhaps one of the more popular approaches is called the “Multi-Task Cascaded Convolutional Neural Network,” or MTCNN for short, described by Kaipeng Zhang, et al. in the 2016 paper titled “Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks.”

The MTCNN is popular because it achieved then state-of-the-art results on a range of benchmark datasets, and because it is capable of also recognizing other facial features such as eyes and mouth, called landmark detection.

The network uses a cascade structure with three networks; first the image is rescaled to a range of different sizes (called an image pyramid), then the first model (Proposal Network or P-Net) proposes candidate facial regions, the second model (Refine Network or R-Net) filters the bounding boxes, and the third model (Output Network or O-Net) proposes facial landmarks.

The model is called a multi-task network because each of the three models in the cascade (P-Net, R-Net and O-Net) are trained on three tasks, e.g. make three types of predictions; they are: face classification, bounding box regression, and facial landmark localization.

The three models are not connected directly; instead, outputs of the previous stage are fed as input to the next stage. This allows additional processing to be performed between stages; for example, non-maximum suppression (NMS) is used to filter the candidate bounding boxes proposed by the first-stage P-Net prior to providing them to the second stage R-Net model.

The MTCNN architecture is reasonably complex to implement. Thankfully, there are open source implementations of the architecture that can be trained on new datasets, as well as pre-trained models that can be used directly for face detection. Of note is the official release with the code and models used in the paper, with the implementation provided in the Caffe deep learning framework.

Perhaps the best-of-breed third-party Python-based MTCNN project is called “MTCNN” by Iván de Paz Centeno, or ipazc, made available under a permissive MIT open source license. As a third-party open-source project, it is subject to change, therefore I have a fork of the project at the time of writing available here.

The MTCNN project, which we will refer to as ipazc/MTCNN to differentiate it from the name of the network, provides an implementation of the MTCNN architecture using TensorFlow and OpenCV. There are two main benefits to this project; first, it provides a top-performing pre-trained model and the second is that it can be installed as a library ready for use in your own code.

## Installation
- !pip install opencv-python
- !pip install mtcnn

## Results
![bounding_box2](https://github.com/dreamboat26/leonidas/assets/125608791/f4b4a817-8c06-43ae-87b6-b5e2bee1c79e)
![bounding_box1](https://github.com/dreamboat26/leonidas/assets/125608791/89e0ac24-86dd-400a-88ed-f9eb14b51364)
![Capture](https://github.com/dreamboat26/leonidas/assets/125608791/1bf6c7ab-ad2f-4c8c-946a-1333f0235a38)
![patch-1](https://github.com/dreamboat26/leonidas/assets/125608791/4d4107fb-00f1-40fb-b647-634205d1b335)
![Plot-of-Each-Separate-Face-Detected-in-a-Photograph-of-a-Swim-Team-1024x169](https://github.com/dreamboat26/leonidas/assets/125608791/a51cb1a7-cd14-431c-a36b-f97511363812)

Tutorial Link :- https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/
