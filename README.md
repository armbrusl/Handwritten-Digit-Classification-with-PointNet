# Handwritten Classification with PointNet (HCwPN)

## Using a 3D PointCloud to classify handwritten digits using the MNIST dataset.

## 1. The excisitng mnist dataset is augmented by expaning the matrices, rotationg and adding noise. This is done in the 'dataPreProcessing.py' file.

## 2. Using the 'main.py' file we can train a 3D pointcloud on the generated 'augmented mnist' dataset. Choosing the right hyperparameters is crucial during this step. The 'N_Points' varaible allows us to choose how many of the 'biggest values' in our pointcloud are actually considered during training. Keeping this number as low as possible is crucial.

## 3. In the 'analyzingTraining.ipynb' file we can look at the Loss terms from training

## 4. In the 'predictio.py' file we can test the model on testdata.


## The bes tmodel I have archieved thus far has an accuracy of 83 % and takes about 1ms to per pointCloud prediction.
