import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model


def LoadData():
    #base_path = '/media/ymos/armbrusl/Projects/HCwPN/data/'
    base_path = '/home/ymos/Documents/coding/HCwPN_data/data/'
    InputName = 'n_TestInput'
    OutputName = 'n_TestOutput'
    
    INPUT, OUTPUT = [], []
    
    for i in range(2):
        INPUT.append(np.load(f'{base_path}{InputName}{i}.npy'))
        OUTPUT.append(np.load(f'{base_path}{OutputName}{i}.npy'))
        
    INPUT = np.concatenate(INPUT, axis=0)
    OUTPUT = np.concatenate(OUTPUT, axis=0)
    
    return INPUT, OUTPUT

def CheckTensorSize(TestInput, TestOutput):
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

def DecreasePointCloud(size, Input):

    n_trainInput = np.zeros((len(Input), size, 3))
    c = 0
    for input in Input:
        largest_indices = np.argsort(input[:, 2])[-size:]
        n_trainInput[c] = np.take(input, largest_indices, axis=0)
        c += 1
        
    return n_trainInput

def plotExamplePredictions(number=20):

    indices = np.arange(0, len(n_TestInput), 1)
    random_indices = np.random.choice(indices, number)

    plt.figure(figsize=(5, 20))
    for i, c in zip(random_indices, np.arange(1, len(random_indices)+1, 2)):
            
        plt.subplot(int(len(random_indices)/5), 5, c)
        plt.scatter(n_TestInput[i, :, 0], n_TestInput[i, :, 1], c= n_TestInput[i, :, 2], s=5, cmap='jet')
        plt.title(str(np.where(TestOutput[i]==1)[0]) + ' | ' + str(np.argmax(predictions[i])))
        plt.axis('scaled')
        plt.colorbar(shrink=0.5)
        plt.xticks([])
        plt.yticks([])

        #plt.subplot(int(len(random_indices)/5), 5, c+1)
        #plt.plot(np.linspace(0, 9, 10), predictions[i])
        #plt.xticks(np.linspace(0, 9, 10))
        #plt.ylim(0, 1)
        
    plt.show()
    
def predictAll(TestInput, predictions, TestOutput, path, n_TestInput):
    
    counter = 0
    counter2 = 0
    totalLength = len(TestOutput)

    for pred, truth, input, n_input in zip(predictions, TestOutput, TestInput, n_TestInput):
        
        truthval = np.where(truth==1)[0]
        predval = np.argmax(pred)
        
        if(truthval == predval):
            counter += 1
        else:
            counter2 += 1
            plt.figure(figsize=(10, 4))
            
            plt.subplot(1, 2, 1)
            plt.scatter(input[:, 0], input[:, 1], c=input[:, 2], s=5, cmap='jet')
            plt.colorbar()
            plt.title('Truth: ' + str(truthval[0]) + ' -|- Pred: ' + str(predval))
            
            plt.subplot(1, 2, 2)
            plt.scatter(n_input[:, 0], n_input[:, 1], c=n_input[:, 2], s=5, cmap='jet')
            plt.colorbar()
            
            
            plt.savefig(path + 'WrongPred/' + str(counter2) + '.png')
            plt.clf()
            plt.close()
            
    return counter/totalLength * 100
    

N_Points = 128
    
path = '/home/ymos/Documents/coding/HCwPN_data/'
TestInput, TestOutput= LoadData()
n_TestInput = DecreasePointCloud(N_Points, TestInput)
CheckTensorSize(n_TestInput, TestOutput)

# Load the model
model = load_model(path + 'SavedModels/20231028-165255.keras')
predictions = model.predict(n_TestInput)
            
    
truthRatio = predictAll(TestInput, predictions, TestOutput, path, n_TestInput)
print(str(truthRatio) + ' precent of all Test Inputs where predicted successfully.')
#plotExamplePredictions(20)

