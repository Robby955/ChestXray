# Chest Xray Data
Working through the Kaggle Chest Xray dataset in Python and Keras/Tensorflow.This data consists of thousands of chest xrays. The chests are either normal(Healthy) , or sick(Pneumonia). The goal is to recoginize if a chest is healthy or sick based off the chest xray.

![What is this](images/xray1.png)


Is an image of some of the training data.


To accomplish this we utilize Convolutional Neural Networks with relu activations, and we utilize tensorflows built in functionallity. We utilize ImageDataGenerator to augmnet images to artifically increase training data size. We also visualize some inner layers.
