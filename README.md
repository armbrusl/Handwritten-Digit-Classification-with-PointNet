# Handwritten Digit Classification with PointNet

Utilizing a 3D PointCloud to classify handwritten digits using an augmented version of the MNIST dataset. To accelarate the computation tensorflow uses a local GPU. <br>

## Overview

1. Data Augmentation : The existing MNIST dataset is augmented by expanding the matrices, rotating, and adding noise to create a more versatile dataset. This process is carried out in the dataPreProcessing.py file.

2. Model Training    : A 3D PointCloud model is trained on the generated 'augmented mnist' dataset using the main.py file. The selection of appropriate hyperparameters is essential. The variable N_Points allows for the adjustment of how many of the     'largest values' in our point cloud are considered during training, and keeping this number as low as possible is crucial.

3. Training Analysis: The analyzingTraining.ipynb file contains a detailed examination of the loss terms from the training process, facilitating a deeper understanding of the model's performance.

4. Model Testing    : The model's performance in terms of classification accuracy can be tested on new, unseen data using the prediction.py file, providing insights into its generalization capability.

## Best Model Performance

The best model achieved so far has demonstrated an accuracy of 84.1%, taking about 1ms per PointCloud prediction, showcasing its efficiency and effectiveness in classifying handwritten digits:

    
    N_Points        = 128               # how many points should be taken from the pointcloud ?
    N_Dimensions    = 3                 # number of input dimensions
    N_variables     = 10                # number of output dimensions
    NetworkScale    = 0.5               # scaling factor for the neural network
    n_epochs        = 2000              # number of epochs
    initial_lr      = 2e-3              # initial learning rate
    batchSize       = 240    

## Preproducing Results

The 'requirements.txt' file includes the libraries contained in the enviroment I used.


