
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate
from tqdm import tqdm
from tensorflow.keras.layers import Input, Convolution1D, MaxPooling1D, Lambda, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import initializers

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


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

def BUILD_NEURAL_NETWORK(N_Points, N_Dimensions, N_variables, NetworkScale):
    _tf_INPUT_TENSOR = Input(shape=(N_Points, N_Dimensions))
    
    g = Convolution1D(int(64*NetworkScale), 1, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros())(_tf_INPUT_TENSOR)
    g = Convolution1D(int(64*NetworkScale), 1, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros())(g)

    seg_part1 = g
    g = Convolution1D(int(64*NetworkScale), 1, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros())(g)
    g = Convolution1D(int(128*NetworkScale), 1, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros())(g)
    g = Convolution1D(int(1024*NetworkScale), 1, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros())(g)
    
    #global_feature = MaxPooling1D(pool_size=N_Points)(g)
    #global_feature = Lambda(exp_dim, arguments={'N_Points': N_Points})(global_feature)
    c = g
    #c = concatenate([seg_part1, global_feature])
    c = Convolution1D(int(512*NetworkScale), 1, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros())(c)
    
    c = Dropout(rate=0.5)(c)  # Adding dropout layer
    
    c = Convolution1D(int(256*NetworkScale), 1, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros())(c)
    c = Convolution1D(int(128*NetworkScale), 1, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros())(c)
    c = Convolution1D(int(128*NetworkScale), 1, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros())(c)
    prediction = Convolution1D(N_variables, 1, activation='sigmoid', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros())(c)

    model = Model(inputs=_tf_INPUT_TENSOR, outputs=prediction)
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
 
def transform_inputs(TrainInput, TrainOutput, TestInput, TestOutput, ValInput, ValOutput):
    n_TrainInput = []
    n_TestInput = []
    n_ValInput = []

    n_TrainOutput = []
    n_TestOutput = []
    n_ValOutput = []
    
    
    indices = np.arange(0, len(TrainInput), 1)
    for q in tqdm(range(3*len(TrainInput))):
        
        index = np.random.choice(indices)
        original_matrix = TrainInput[index]
        
        rotated_matrix = rotate(original_matrix, np.random.randint(-20, 20), reshape=False)
        expanded_matrix = expand_matrix(rotated_matrix, num_rows_to_add=10, num_cols_to_add=10, fill_value=0)
        shifted_matrix = shift_matrix(expanded_matrix)
        noisy_matrix = add_gaussian_noise(shifted_matrix, mean=0, std=np.random.randint(0, 80))
        scaled_noisy_matrix = scale_matrix(noisy_matrix, new_min=0, new_max=1)
        
        point = []
        for i in range(48):
            for j in range(48):
                point.append([j/48,  (48 - i)/48, scaled_noisy_matrix[i, j]])
            
        n_TrainInput.append(point)
        
        output = np.zeros(10)
        output[TrainOutput[q]] = 1
        n_TrainOutput.append(output.T)
  
    for _ in tqdm(range(len(TestInput) - 4500)):
        
        original_matrix = TestInput[_]
        
        rotated_matrix = rotate(original_matrix, np.random.randint(-20, 20), reshape=False)
        expanded_matrix = expand_matrix(rotated_matrix, num_rows_to_add=10, num_cols_to_add=10, fill_value=0)
        shifted_matrix = shift_matrix(expanded_matrix)
        noisy_matrix = add_gaussian_noise(shifted_matrix, mean=0, std=np.random.randint(0, 80))
        scaled_noisy_matrix = scale_matrix(noisy_matrix, new_min=0, new_max=1)
        

        point = []
        for i in range(48):
            for j in range(48):
                point.append([j/48,  (48 - i)/48, scaled_noisy_matrix[i, j]])
            
        n_TestInput.append(point)
        
        output = np.zeros(10)
        output[TestOutput[_]] = 1
        n_TestOutput.append(output.T)
        
    for _ in tqdm(range(len(ValInput) - 4500)):
        
        original_matrix = ValInput[_]
        
        rotated_matrix = rotate(original_matrix, np.random.randint(-20, 20), reshape=False)
        expanded_matrix = expand_matrix(rotated_matrix, num_rows_to_add=10, num_cols_to_add=10, fill_value=0)
        shifted_matrix = shift_matrix(expanded_matrix)
        noisy_matrix = add_gaussian_noise(shifted_matrix, mean=0, std=np.random.randint(0, 80))
        scaled_noisy_matrix = scale_matrix(noisy_matrix, new_min=0, new_max=1)
        
        point = []
        for i in range(48):
            for j in range(48):
                point.append([j/48, (48 - i)/48, scaled_noisy_matrix[i, j]])
            
        n_ValInput.append(point)
        
        output = np.zeros(10)
        output[ValOutput[_]] = 1
        n_ValOutput.append(output.T)

        
    return np.array(n_TrainInput), np.array(n_TestInput), np.array(n_ValInput), np.array(n_TrainOutput), np.array(n_TestOutput), np.array(n_ValOutput)
    
print('a')
N_Points = 48*48
N_Dimensions = 3
N_variables = 10
NetworkScale = 1
n_epochs=1000
initial_lr = 0.001


def load_data():
    INPUT = []
    OUTPUT = []
    for qq in range(4):
        






check_tensor_sizes(TrainInput, TrainOutput, TestInput, TestOutput, ValInput, ValOutput)

model = BUILD_NEURAL_NETWORK(N_Points, N_Dimensions, N_variables, NetworkScale)


n_TrainInput, n_TestInput, n_ValInput, n_TrainOutput, n_TestOutput, n_ValOutput = transform_inputs(TrainInput, TrainOutput, TestInput, TestOutput, ValInput, ValOutput)


model = BUILD_NEURAL_NETWORK(N_Points, N_Dimensions, N_variables, NetworkScale)
print_model_summary(model)

learning_rate_piecewise = tf.keras.optimizers.schedules.PiecewiseConstantDecay([int(n_epochs*0.3), int(n_epochs*0.6)],
                                                                               [initial_lr, initial_lr/10, initial_lr/100])
optimizer = keras.optimizers.Adam(learning_rate=learning_rate_piecewise)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


