# load libraries
from keras.preprocessing.image import load_img
import numpy as np

def prediction_function(image_path,model):
    # load the image
    image_data_test = load_img(image_path,target_size=(224,224))
    # convert image into numpy array
    image_data_test = np.array(image_data_test)
    # re-shape the image for model
    image_data_test = image_data_test.reshape((1,224,224,3))
    # predict the image
    predict_test_image = model.predict(image_data_test)
    # round-up prediction
    # it will give either '0' or '1'
    predict_test_image = round(predict_test_image[0][0])
    
    # check for child safe
    if int(predict_test_image) == 0:
        print('This Photo is a Good Photo for a Child.')
    else:
        print('Warning! Its Bad Photo for a child.')