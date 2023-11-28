#%% import library
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import prepare_data as pd
import numpy
import math

#%% PSNR calculation
def psnr(target, ref):
    # assume RGB image
    target_data = numpy.array(target, dtype=float)
    ref_data = numpy.array(ref, dtype=float)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(numpy.mean(diff ** 2.))
    return 20 * math.log10(255. / rmse)

#%% SRCNN model 
def model():
    SRCNN = Sequential()
    SRCNN.add(Conv2D(filters=128, kernel_size=9, kernel_initializer='he_normal',
                     activation='relu', padding='valid', use_bias=True, input_shape=(32, 32, 1)))
    SRCNN.add(Conv2D(filters=64, kernel_size=3, kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))
    SRCNN.add(Conv2D(filters=1, kernel_size=5, kernel_initializer='glorot_uniform',
                     activation='linear', padding='valid', use_bias=True))
    adam = Adam(lr=0.003)
    SRCNN.compile(optimizer=adam, loss='mean_absolute_error', metrics=['mean_absolute_error'])
    return SRCNN

#%% SRCNN model training
srcnn_model = model()
#print(srcnn_model.summary())
data, label = pd.read_training_data("./crop_train.h5")
val_data, val_label = pd.read_training_data("./test.h5")
checkpoint = ModelCheckpoint("SRCNN_check.h5", monitor='val_loss', verbose=1, save_best_only=True,
                             save_weights_only=False, mode='min')
callbacks_list = [checkpoint]
srcnn_model.fit(data, label, batch_size=256, validation_data=(val_data, val_label),
                callbacks=callbacks_list, shuffle=True, epochs=3, verbose=1)
