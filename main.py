import numpy as np
import datetime
import json
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Convolution1D, MaxPooling1D, Lambda, concatenate, Dropout, Dense, Conv1D, Flatten, GlobalMaxPooling1D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import layers, models, callbacks


# Getting the current data and time
now = datetime.datetime.now()
current_datetime_str = now.strftime("%Y%m%d-%H%M%S")


#Adding this for gpu compatibility
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print('There is/are ', len(physical_devices), ' GPUs available')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def CheckTensorSize(TrainInput, TrainOutput, TestInput, TestOutput, ValInput, ValOutput):
    
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
        
def BuildNN(N_Points, N_Dimensions, N_variables, NetworkScale):

    input_layer = Input(shape=(N_Points, N_Dimensions))

    # Convolutional layers
    x = Conv1D(int(NetworkScale*64), 1, activation='relu')(input_layer)
    x = Conv1D(int(NetworkScale*128), 1, activation='relu')(x)
    x = Conv1D(int(NetworkScale*256), 1, activation='relu')(x)
    
    x = GlobalMaxPooling1D()(x)
    
    # Fully connected layers
    x = Dense(int(NetworkScale*512), activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(int(NetworkScale*256), activation='relu')(x)
    x = Dropout(0.3)(x)

    output_layer = Dense(N_variables, activation='softmax')(x)


    return Model(inputs=input_layer, outputs=output_layer)

def create_dense_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def LoadData(path):
    base_path = path + 'data/'
    datasets = ['Train', 'Test', 'Val']
    
    results = []
    
    for dataset in datasets:
        inputs, outputs = [], []
        
        for i in range(2):
            inputs.append(np.load(f'{base_path}n_{dataset}Input{i}.npy'))
            outputs.append(np.load(f'{base_path}n_{dataset}Output{i}.npy'))
            
        results.extend([np.concatenate(inputs, axis=0), np.concatenate(outputs, axis=0)])
        
    return tuple(results)

def DecreasePointCloud(N_Points, Input):

    n_trainInput = np.zeros((len(Input), N_Points, 3))
    
    c = 0
    for input in Input:
        largest_indices = np.argsort(input[:, 2])[-N_Points:]
        n_trainInput[c] = np.take(input, largest_indices, axis=0)

        c += 1
        
    return n_trainInput

class TimeHistory(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        logs['epoch_duration'] = time.time() - self.epoch_start_time



path = '/home/ymos/Documents/coding/HCwPN_data/'

# Loading in the altered dataset
TrainInput, TrainOutput, TestInput, TestOutput, ValInput, ValOutput = LoadData(path)

# Checking if the tensors are the right N_Points
CheckTensorSize(TrainInput, TrainOutput, TestInput, TestOutput, ValInput, ValOutput)

# Parameters
N_Points        = 192               # how many points should be taken from the pointcloud ?
N_Dimensions    = 3                 # number of input dimensions
N_variables     = 10                # number of output dimensions
NetworkScale    = 0.5                 # scaling factor for the neural network
n_epochs        = 10              # number of epochs
initial_lr      = 2e-3              # initial learning rate
batchSize       = 240               

# Decreasing PointCloud N_Points
TrainInput = DecreasePointCloud(N_Points, TrainInput)
TestInput = DecreasePointCloud(N_Points, TestInput)
ValInput = DecreasePointCloud(N_Points, ValInput)

# Checking if the tensors are the right N_Points after decreasing the N_Points
CheckTensorSize(TrainInput, TrainOutput, TestInput, TestOutput, ValInput, ValOutput)


# Building the Neural network 
model = BuildNN(N_Points, N_Dimensions, N_variables, NetworkScale)
#model = load_model('/home/ymos/Documents/coding/HCwPN_data/models/MODEL_20231028-154233.keras')
model.summary()

# Initialize the custom callback
time_callback = TimeHistory()


learning_rate_piecewise = tf.keras.optimizers.schedules.PiecewiseConstantDecay([int(n_epochs*0.6)],
                                                                               [initial_lr, initial_lr/5])

model.compile(optimizer = keras.optimizers.Adam(learning_rate=learning_rate_piecewise), 
              loss      = 'categorical_crossentropy', 
              metrics   = ['accuracy'])

checkpoint = callbacks.ModelCheckpoint(path + '/Models', 
                                       monitor          = 'val_accuracy', 
                                       save_best_only   = True, 
                                       mode             = 'max', 
                                       verbose          = 0, 
                                       save_format      = 'tf')

tensorboard_callback = callbacks.TensorBoard(log_dir=path + 'logs/', 
                                             histogram_freq = int(n_epochs/100))

history = model.fit(TrainInput, TrainOutput,
                    epochs          = n_epochs,
                    batch_size      = batchSize,
                    validation_data = (ValInput, ValOutput),
                    verbose         = 1, 
                    callbacks       = [checkpoint, tensorboard_callback, time_callback])


model.save(path + 'SavedModels/' + current_datetime_str + '.keras')

with open(f'{path}History/{current_datetime_str}.json', 'w') as file:
    json.dump(history.history, file)
