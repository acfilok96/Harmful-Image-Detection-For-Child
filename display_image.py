from matplotlib import pyplot
from tensorflow import keras
import numpy as np


def show(image_path):
    if(type(image_path) == np.ndarray):
        image_data = image_path.reshape((224,224,3))
        pyplot.imshow(image_data)
    if(type(image_path) == str):
        image_data = keras.preprocessing.image.load_img(image_path,target_size=(224,224))
        image_data = np.array(image_data)
        pyplot.imshow(image_data)
    pyplot.show()
