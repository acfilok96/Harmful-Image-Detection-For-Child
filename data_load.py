import numpy as np
import os

def load_data(non_nude_data_path_1,non_nude_data_path_2,nude_data_path):

    # non_nude_dataset 1
    loaded_non_nude_1 = np.load(non_nude_data_path_1)
    loaded_non_nude_1_temp = loaded_non_nude_1['arr_0']

    # non_nude_dataset 2
    loaded_non_nude_2 = np.load(non_nude_data_path_2)
    loaded_non_nude_2_temp = loaded_non_nude_2['arr_0']

    # nude_dataset
    loaded_nude = np.load(nude_data_path)
    loaded_nude_temp = loaded_nude['arr_0']

    #image data
    x_train=[]
    for i in loaded_non_nude_1_temp:
        x_train.append(i)
    for k in loaded_non_nude_2_temp:
        x_train.append(k)

    for j in loaded_nude_temp:
        x_train.append(j)
    x_train = np.array(x_train)


    # level data
    zero_y = np.zeros(len(loaded_non_nude_1_temp)+len(loaded_non_nude_2_temp))
    one_y = np.ones(len(loaded_nude_temp))
    y_train=[]
    for i in zero_y:
        y_train.append(i)
    for j in one_y:
        y_train.append(j)
    y_train = np.array(y_train)
    
    return x_train,y_train


