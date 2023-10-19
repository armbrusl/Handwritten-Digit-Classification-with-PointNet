import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras.layers import Input, Convolution1D, MaxPooling1D, Lambda, concatenate, Dropout, Dense, Conv1D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
from tensorflow.keras import layers, models, callbacks
from tqdm.keras import TqdmCallback 


#Adding this for gpu compatibility
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print('There is/are ', len(physical_devices), ' GPUs available')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def check_tensor_sizes(TrainInput, TrainOutput, TestInput, TestOutput, ValInput, ValOutput):
    D = [TrainInput, TrainOutput, TestInput, TestOutput, ValInput, ValOutput]
    totaltrajectories = 0
    
    tensornames = ['train input', 'train output', 'test input', 'test output', 'validation input', 'validation output']
    print(f"{'Tensor Name': <30}{'Shape': <30}")
    print("_"*60)
    for _ in range(len(D)):
        totaltrajectories += D[_].shape[0]
        print(f"{tensornames[_]: <30}{str(D[_].shape): <30}")
        
        if(_ % 2 != 0):
            print('')
        
    print('')
    print("There are ", int(totaltrajectories/2), ' different images with their corresponding labels.')

def exp_dim(global_feature, N_Points):
    return tf.tile(global_feature, [1, N_Points, 1])

class custom_history_and_tqdm(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.tqdm_callback = TqdmCallback()

    def on_train_begin(self, logs=None):
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        for key, value in logs.items():
            self.history.setdefault(key, []).append(value)
        self.tqdm_callback.on_epoch_end(epoch, logs)

def BUILD_NEURAL_NETWORK(N_Points, N_Dimensions, N_variables, NetworkScale):

    input_layer = Input(shape=(N_Points, N_Dimensions))

    # Convolutional layers
    x = Conv1D(int(NetworkScale*16), N_Dimensions, activation='relu')(input_layer)
    x = Conv1D(int(NetworkScale*32), N_Dimensions, activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(int(NetworkScale*32), N_Dimensions, activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = Flatten()(x)

    # Fully connected layers
    x = Dense(int(NetworkScale*32), activation='relu')(x)
    x = Dense(int(NetworkScale*16), activation='relu')(x)
    output_layer = Dense(N_variables, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output_layer)


    return model

def print_model_summary(model):

    totalparams = 0
    print(f"{'Layer (type)': <35}{'Output Shape': <30}{'Param #': <15}")
    print("="*100)
    for layer in model.layers:
        totalparams += layer.count_params()
        print(f"{layer.name: <35}{str(layer.output_shape): <30}{layer.count_params(): <15}")
    print('')
    print('Total Parameters: ', totalparams)

def load_data_from_npy():
    InputNames = ['n_TrainInput', 'n_TestInput', 'n_ValInput']
    OutputNames = ['n_TrainOutput', 'n_TestOutput', 'n_ValOutput']

    INPUT = []
    OUTPUT = []

    for inputname, outputname in zip(InputNames, OutputNames):
        input1 = np.load('/media/ymos/armbrusl/Projects/HCwPN/data/' + inputname + str(0) + '.npy')
        input2 = np.load('/media/ymos/armbrusl/Projects/HCwPN/data/' + inputname + str(1) + '.npy')
        input3 = np.load('/media/ymos/armbrusl/Projects/HCwPN/data/' + inputname + str(2) + '.npy')
        input4 = np.load('/media/ymos/armbrusl/Projects/HCwPN/data/' + inputname + str(3) + '.npy')
        
        output1 = np.load('/media/ymos/armbrusl/Projects/HCwPN/data/' + outputname + str(0) + '.npy')
        output2 = np.load('/media/ymos/armbrusl/Projects/HCwPN/data/' + outputname + str(1) + '.npy')
        output3 = np.load('/media/ymos/armbrusl/Projects/HCwPN/data/' + outputname + str(2) + '.npy')
        output4 = np.load('/media/ymos/armbrusl/Projects/HCwPN/data/' + outputname + str(3) + '.npy')
        
        INPUT.append(np.concatenate((input1, input2, input3, input4), axis=0))
        OUTPUT.append(np.concatenate((output1, output2, output3, output4), axis=0))
          
    return INPUT[0], OUTPUT[0], INPUT[1], OUTPUT[1],  INPUT[2], OUTPUT[2] 

def decreasePointCloud(size, Input):

    n_trainInput = np.zeros((len(Input), size, 3))
    c = 0
    for input in Input:
        largest_indices = np.argsort(input[:, 2])[-size:]
        n_trainInput[c] = np.take(input, largest_indices, axis=0)
        c += 1
        
    return n_trainInput

TrainInput, TrainOutput, TestInput, TestOutput, ValInput, ValOutput = load_data_from_npy()
check_tensor_sizes(TrainInput, TrainOutput, TestInput, TestOutput, ValInput, ValOutput)


# Parameters
size = 2304
N_Points        = size
N_Dimensions    = 3
N_variables     = 10
NetworkScale    = 1
n_epochs        = 2000
initial_lr      = 1e-3
batchSize = 960


TrainInput = decreasePointCloud(size, TrainInput)
TestInput = decreasePointCloud(size, TestInput)
ValInput = decreasePointCloud(size, ValInput)



model = BUILD_NEURAL_NETWORK(N_Points, N_Dimensions, N_variables, NetworkScale)


print_model_summary(model)

learning_rate_piecewise = tf.keras.optimizers.schedules.PiecewiseConstantDecay([int(n_epochs*0.3), int(n_epochs*0.6)],[initial_lr, initial_lr/10, initial_lr/100])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate_piecewise), loss='categorical_crossentropy', metrics=['accuracy'])
checkpoint = callbacks.ModelCheckpoint('Models.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=0)
#tensorboard_callback = callbacks.TensorBoard(log_dir="logs/", histogram_freq=int(n_epochs/100))
history = model.fit(TrainInput, TrainOutput,epochs=n_epochs,batch_size=batchSize,validation_data=(ValInput, ValOutput),verbose=1)


model.save('moddel.keras')