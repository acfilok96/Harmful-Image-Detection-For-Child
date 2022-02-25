import numpy as np
from tensorflow import keras

def train(model_path,data,epochs=20,batch_size=32,optimizer=keras.optimizers.Adam(1e-3),loss="binary_crossentropy"):

    callbacks = [keras.callbacks.ModelCheckpoint("model_{epoch}.h5")]
    model_path.compile(optimizer=optimizer,loss=loss,metrics=["accuracy"])
    print("model compilation complete!")

    x_train, y_train = np.array(data[0]),np.array(data[1])

    print('training started. . .')
    model_path.fit(x_train, y_train, epochs = epochs, batch_size = batch_size, callbacks=callbacks, validation_data=(x_train, y_train))