import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

def load_data_from_npy():
    InputName = 'n_TestInput'
    OutputName = 'n_TestOutput'

    INPUT = []
    OUTPUT = []

    input1 = np.load('/media/ymos/armbrusl/Projects/HCwPN/data/' + InputName + str(0) + '.npy')
    input2 = np.load('/media/ymos/armbrusl/Projects/HCwPN/data/' + InputName + str(1) + '.npy')
    input3 = np.load('/media/ymos/armbrusl/Projects/HCwPN/data/' + InputName + str(2) + '.npy')
    input4 = np.load('/media/ymos/armbrusl/Projects/HCwPN/data/' + InputName + str(3) + '.npy')
    
    output1 = np.load('/media/ymos/armbrusl/Projects/HCwPN/data/' + OutputName + str(0) + '.npy')
    output2 = np.load('/media/ymos/armbrusl/Projects/HCwPN/data/' + OutputName + str(1) + '.npy')
    output3 = np.load('/media/ymos/armbrusl/Projects/HCwPN/data/' + OutputName + str(2) + '.npy')
    output4 = np.load('/media/ymos/armbrusl/Projects/HCwPN/data/' + OutputName + str(3) + '.npy')
        
    INPUT.append(np.concatenate((input1, input2, input3, input4), axis=0))
    OUTPUT.append(np.concatenate((output1, output2, output3, output4), axis=0))
          
    return INPUT[0], OUTPUT[0]

def check_tensor_sizes(TestInput, TestOutput):
    D = [TestInput, TestOutput]
    totaltrajectories = 0
    
    tensornames = ['test input', 'test output']
    print(f"{'Tensor Name': <30}{'Shape': <30}")
    print("_"*60)
    for _ in range(len(D)):
        totaltrajectories += D[_].shape[0]
        print(f"{tensornames[_]: <30}{str(D[_].shape): <30}")
        
        if(_ % 2 != 0):
            print('')
        
    print('')
    print("There are ", int(totaltrajectories/2), ' different images with their corresponding labels.')

def decreasePointCloud(size, Input):

    n_trainInput = np.zeros((len(Input), size, 3))
    c = 0
    for input in Input:
        largest_indices = np.argsort(input[:, 2])[-size:]
        n_trainInput[c] = np.take(input, largest_indices, axis=0)
        c += 1
        
    return n_trainInput


TestInput, TestOutput= load_data_from_npy()
check_tensor_sizes(TestInput, TestOutput)
n_TestInput = decreasePointCloud(300, TestInput)
check_tensor_sizes(TestInput, TestOutput)


# Load the model
model = load_model('/home/ymos/MODELNEW.keras')
model.summary()




predictions = model.predict(n_TestInput)

indices = np.arange(0, len(n_TestInput), 1)
random_indices = np.random.choice(indices, 10)

for i in random_indices:
        
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(n_TestInput[i, :, 0], n_TestInput[i, :, 1], c= n_TestInput[i, :, 2], s=5, cmap='jet')
    plt.title(str(np.where(TestOutput[i]==1)[0]))
    plt.axis('scaled')
    plt.colorbar(shrink=0.5)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.plot(np.linspace(0, 9, 10), predictions[i])
    plt.xticks(np.linspace(0, 9, 10))
    plt.ylim(0, 1)
    plt.show()


