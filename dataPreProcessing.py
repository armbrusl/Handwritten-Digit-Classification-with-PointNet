import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.ndimage import rotate
from tqdm import tqdm
import matplotlib.pyplot as plt


# Load the MNIST dataset
(TrainInput, TrainOutput), (TestValInput, TestValOutput) = mnist.load_data()

def create_points(original_matrix):
    dim = original_matrix.shape[0]
    x, y = np.meshgrid(np.arange(dim), np.arange(dim))
    x = x / dim * 10
    y = (dim - y) / dim * 10
    points = np.stack((x, y, original_matrix), axis=-1)
    
    return points.reshape(-1, 3)

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

def add_gaussian_noise(matrix, mean=0, std=0.1):
    noise = np.random.normal(mean, std, size=matrix.shape)
    noisy_matrix = matrix + noise
    return noisy_matrix

def scale_matrix(matrix, new_min=0, new_max=255):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    scaled_matrix = ((matrix - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min
    return scaled_matrix

def expand_matrix(matrix, num_rows_to_add, num_cols_to_add, fill_value=0):
    expanded_matrix = np.pad(matrix, ((num_rows_to_add, num_rows_to_add), (num_cols_to_add, num_cols_to_add)), mode='constant', constant_values=fill_value)
    return expanded_matrix

def shift_matrix(matrix):
    
    shift_amount = np.random.randint(-5, 5, size=2)  # Shifts can be -1, 0, or 1
    shifted_matrix = np.zeros_like(matrix)
    rows, cols = matrix.shape
    for r in range(rows):
        for c in range(cols):
            new_r = (r + shift_amount[0]) % rows
            new_c = (c + shift_amount[1]) % cols
            shifted_matrix[new_r, new_c] = matrix[r, c]
    return shifted_matrix
   
def transform_inputs(Input, Output, multiplier):
    new_Input = []
    new_Output = []

    indices = np.arange(0, len(Input), 1)
    random_indices = np.random.choice(indices, int(multiplier*len(Input)))

    for index in tqdm(range(len(random_indices))):
        
        ii = random_indices[index]
        original_matrix = Input[ii]
        
        rotated_matrix = rotate(original_matrix, np.random.randint(-20, 20), reshape=False)
        expanded_matrix = expand_matrix(rotated_matrix, num_rows_to_add=10, num_cols_to_add=10, fill_value=0)
        shifted_matrix = shift_matrix(expanded_matrix)
        noisy_matrix = add_gaussian_noise(shifted_matrix, mean=0, std=np.random.randint(0, 40))
        scaled_noisy_matrix = scale_matrix(noisy_matrix, new_min=0, new_max=1)
        
            
        new_Input.append(create_points((scaled_noisy_matrix + 1) ** 4))
        
        output = np.zeros(10)
        output[Output[ii]] = 1
        new_Output.append(output.T)
        
  
    return np.array(new_Input), np.array(new_Output)


ValInput, TestInput, ValOutput, TestOutput = train_test_split(TestValInput, TestValOutput, test_size = 0.5, random_state=31)

check_tensor_sizes(TrainInput, TrainOutput, TestInput, TestOutput, ValInput, ValOutput)

path = '/home/ymos/Documents/coding/HCwPN_data/'

for qq in range(2):
    print('iteration: ', qq)
    n_TrainInput, n_TrainOutput = transform_inputs(TrainInput, TrainOutput, multiplier=0.1)
    n_TestInput, n_TestOutput = transform_inputs(TestInput, TestOutput, multiplier=0.1)
    n_ValInput, n_ValOutput = transform_inputs(ValInput, ValOutput, multiplier=0.1)

    check_tensor_sizes(n_TrainInput, n_TrainOutput, n_TestInput, n_TestOutput, n_ValInput, n_ValOutput)

    np.save(path + 'data/n_TrainInput' + str(qq) + '.npy', n_TrainInput)
    np.save(path + 'data/n_TrainOutput' + str(qq) + '.npy', n_TrainOutput)

    np.save(path + 'data/n_TestInput' + str(qq) + '.npy', n_TestInput)
    np.save(path + 'data/n_TestOutput' + str(qq) + '.npy', n_TestOutput)

    np.save(path + 'data/n_ValInput' + str(qq) + '.npy', n_ValInput)
    np.save(path + 'data/n_ValOutput' + str(qq) + '.npy', n_ValOutput)

    del n_TrainInput, n_TrainOutput, n_TestInput, n_TestOutput, n_ValInput, n_ValOutput
