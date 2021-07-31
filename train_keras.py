import tensorflow as tf
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Lambda
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.layers import Input,LeakyReLU, MaxPooling2D, AveragePooling2D, Conv2DTranspose, Conv2D, Conv1D, BatchNormalization, UpSampling2D, Add, Concatenate, SeparableConv2D, GlobalAveragePooling2D,GlobalMaxPooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras import regularizers

from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.losses import CategoricalCrossentropy

from tensorflow.keras import initializers

from tensorflow.keras.models import load_model,save_model

from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,LearningRateScheduler,TerminateOnNaN,ReduceLROnPlateau, TensorBoard

from tensorflow.keras import backend as K

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.efficientnet import EfficientNetB0,preprocess_input

import random as rnd
import numpy as np
import matplotlib.pyplot as plt
import math

from models import *

#https://github.com/mjkvaak/ImageDataAugmentor
from ImageDataAugmentor.image_data_augmentor import *

import albumentations

train_data_dir='./training/'
val_data_dir='./validation/'
test_data_dir='./validation/'
img_height=384
img_width=576
batch_size=128
num_classes = 30
nb_epochs = 200
lr_init = 5e-2
num_workers=1
dsplit=0.

pg = 0.2
pc = 0.1
AUGMENTATIONS = albumentations.Compose([

    albumentations.HorizontalFlip(p=0.5),
    albumentations.GridDistortion(p=pg, distort_limit=0.2),
    albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, interpolation=1, border_mode=4,
                                    always_apply=False, p=pg),
    albumentations.IAAPerspective(p=pg, scale=(0.01, 0.05)),
    albumentations.RandomResizedCrop(img_height, img_width, (0.8, 1.), p=pg),


    albumentations.ColorJitter(p=0.5, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    albumentations.CoarseDropout(p=pc, max_holes=10, max_height=50, max_width=50),
    albumentations.Blur(p=pc, blur_limit=20),

    albumentations.Resize(img_height, img_width, p=1.),
])

seed = 22
rnd.seed(seed)
np.random.seed(seed)

train_datagen = ImageDataAugmentor(
        rescale=1.,
        augment=AUGMENTATIONS,
        preprocess_input=None,
        )

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    )

validation_datagen = ImageDataGenerator(rescale=1.,
    preprocessing_function = None,validation_split=dsplit)

validation_generator = validation_datagen.flow_from_directory(
    val_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    )

test_datagen = ImageDataGenerator(rescale=1.,
    preprocessing_function = None)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

def cosine_decay(epochs_tot=500,initial_lrate=1e-1,warmup=False):
    def coside_decay_full(epoch,lr,epochs_tot=epochs_tot,initial_lrate=initial_lrate,warmup=warmup):

        lrate = 0.5 * (1 + math.cos(((epoch * math.pi) / (epochs_tot)))) * initial_lrate
        if(warmup and epoch <40):
            lrate = 1e-5
        if(lrate < 1e-6):
            lrate = 1e-6
        return lrate
    return coside_decay_full

def scheduler_step(epoch, lr,tot=100,limit=20):
    if epoch < (tot-limit):
        return lr
    else:
        return lr * tf.math.exp(-0.1)

def exp_decay(k=0.025,initial_rate=0.1):
    def ed(epoch,lr,k=k,initial_rate=initial_rate):
        t=epoch
        lrate = initial_rate*math.exp(-k*t)
        if(lrate < 1e-4):
            lrate
        return lrate
    return ed

SMOOTHING=0.1
loss = CategoricalCrossentropy(label_smoothing=SMOOTHING)

#MobileNetV2 0.75
input_shape = [img_height, img_width, 3]
inp = Input(shape=input_shape)
x=tf.keras.layers.experimental.preprocessing.Resizing(192//2, 288//2, interpolation='bilinear')(inp)
base_model = MobileNetV2(input_tensor=x, include_top=False, weights='imagenet',classes=num_classes,alpha=0.75)
x = base_model.get_layer('block_14_depthwise_relu').output
x = conv_block(x, channels=num_classes, kernel_size=1, dropout_rate=0.)
x = GlobalAveragePooling2D()(x)
x = Activation("softmax", name='softmax')(x)
model = Model(inputs=[inp], outputs=[x])


model.summary()
opt = SGD(lr=lr_init, decay=lr_init/nb_epochs, momentum=0.9, nesterov=True)

model.compile(optimizer=opt,loss=loss, metrics=['accuracy'])
scheduler = exp_decay(k=0.025,initial_rate=lr_init)

lrs = LearningRateScheduler(scheduler,verbose=1)

my_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:03d}-{val_accuracy:.3f}.h5',monitor='val_accuracy',save_best_only=True,verbose=1),
    tf.keras.callbacks.ModelCheckpoint(filepath='model_best.h5',monitor='val_accuracy',save_best_only=True,verbose=1),
    lrs,
]

history=model.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // batch_size,
    epochs = nb_epochs,
    workers=num_workers,
    callbacks=my_callbacks)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./acc.png')
plt.clf()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./loss.png')

score = model.evaluate(test_generator)
print(score)

