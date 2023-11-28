#%% import library
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization
import numpy
import math
import cv2
import tensorflow as tf
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
def predict_model():
    SRCNN = Sequential()
    SRCNN.add(Conv2D(filters=128, kernel_size=9, kernel_initializer='he_normal',
                     activation='relu', padding='valid', use_bias=True, input_shape=(None, None, 1)))
    SRCNN.add(Conv2D(filters=64, kernel_size=3, kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))
    SRCNN.add(Conv2D(filters=1, kernel_size=5, kernel_initializer='glorot_uniform',
                     activation='linear', padding='valid', use_bias=True))
    return SRCNN
#%% SRCNN prediction
srcnn_model = predict_model()
srcnn_model.summary()
srcnn_model.load_weights("SRCNN_check.h5")
IMG_NAME = "img_50.png"
INPUT_NAME = IMG_NAME[:-4]+"bicubic.jpg"
OUTPUT_NAME = IMG_NAME[:-4]+"predictionSRCNN.jpg"
scale=2

img = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
shape = img.shape
Y_img1 = cv2.resize(img[:, :, 0], (int(shape[1] / scale), int(shape[0] / scale)))
Y_img1 = cv2.resize(Y_img1, (shape[1], shape[0]))
img1=img
img1[:, :, 0] = Y_img1
img1 = cv2.cvtColor(img1, cv2.COLOR_YCrCb2BGR)
cv2.imwrite('downscale_image.jpg', img1)

Y_img = cv2.resize(img[:, :, 0], (int(shape[1] / scale), int(shape[0] / scale)), cv2.INTER_CUBIC)
Y_img = cv2.resize(Y_img, (shape[1], shape[0]), cv2.INTER_CUBIC)
#img[:, :, 0] = Y_img
img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
cv2.imwrite(INPUT_NAME, img)

Y = numpy.zeros((1, img.shape[0], img.shape[1], 1), dtype=float)
Y[0, :, :, 0] = Y_img.astype(float) / 255.

pre = srcnn_model.predict(Y, batch_size=1) * 255.
pre[pre[:] > 255] = 255
pre[pre[:] < 0] = 0
pre = pre.astype(numpy.uint8)
img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
img[6: -6, 6: -6, 0] = pre[0, :, :, 0]
img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
cv2.imwrite(OUTPUT_NAME, img)

# psnr calculation:
im1 = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]
im2 = cv2.imread(INPUT_NAME, cv2.IMREAD_COLOR)
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]
im3 = cv2.imread(OUTPUT_NAME, cv2.IMREAD_COLOR)
im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]
#%%
print ("Bicubic:")
print (cv2.PSNR(im1, im2))

print ("SRCNN:")
print (cv2.PSNR(im1, im3))

#%%
im1 = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)

im2 = cv2.imread(INPUT_NAME, cv2.IMREAD_COLOR)

im3 = cv2.imread(OUTPUT_NAME, cv2.IMREAD_COLOR)

bicubic_psnr = tf.image.psnr(im1, im2, max_val=255)
print(bicubic_psnr)