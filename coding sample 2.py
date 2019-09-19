import numpy as np
import read_data
import kaggle_format

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, Activation
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping


def get_batches(batch_size):
    with open("/home/spacebat/furniture_data/lizq/train_labels_224.txt", 'rb') as f:
        labels = np.load(f).astype(np.int32)
    labels = to_categorical(labels, num_classes=128)
    flag = 0
    counter = 0
    while True:
        fname = "/home/spacebat/furniture_data/lizq/train_data_224_" + str(counter+1) + '.txt'
        data = np.load(open(fname, 'rb')).astype(np.float16)
        if counter != 4:
            label = labels[flag:(flag+40000),:]
            flag += 40000
        else:
            label = labels[flag:,:]
            flag = 0
        counter = (counter + 1)%5
        for batch in range(0, data.shape[0], batch_size):
            yield (data[batch:(batch+batch_size),:,:,:]/255.0, label[batch:(batch+batch_size),:])
            


eval_data, eval_labels = read_data.get_eval()
eval_labels = to_categorical(eval_labels, num_classes=128)

# Set the CNN model
model = Sequential()
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 use_bias=False, input_shape = (224,224,3)))
#model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same',
                 use_bias=False))
#model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same',
                 use_bias=False))
#model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(filters = 512, kernel_size = (3,3),padding = 'Same',
                 use_bias=False))
#model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Conv2D(filters = 512, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(filters = 512, kernel_size = (3,3),padding = 'Same',
                 use_bias=False))
#model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Conv2D(filters = 512, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(4096, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dense(4096, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dense(1024, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.4))
model.add(Dense(128, activation = "softmax"))

# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# load weights
# model.load_weights('weights.best.hdf5.v2')

# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

# Set a learning rate annealer
# learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3,
#                                            verbose=1, factor=0.5,  min_lr=0.00001)

# Checkpoint
checkpoint = ModelCheckpoint('weights.best.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# Early stopping
# earlyStopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')

epochs = 10
batch_size = 64

# Without data augmentation
#history = model.fit(train_data, train_labels, batch_size = batch_size, epochs = epochs,
#              validation_data = (eval_data, eval_labels), verbose = 1, callbacks=[checkpoint])
'''
# With data augmentation to prevent overfitting
train_datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image
        shear_range=0.2,  # Randomly shear image
        width_shift_range=0.0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

#eval_datagen = ImageDataGenerator()

train_datagen.fit(train_data)
'''
# Fit the model
history = model.fit_generator(get_batches(batch_size),
                              validation_data = (eval_data, eval_labels),
                              epochs=epochs, verbose = 1, steps_per_epoch=190286 // batch_size
                              , callbacks=[checkpoint])
'''
print("==================================================")
print("Model already trained. Start predicting...")
print("==================================================")
# del train_data
test_data = read_data.get_test()

# predict results
results = model.predict(test_data)
# select the indix with the maximum probability
results = np.argmax(results,axis = 1).astype(np.int16)
results += 1
print(np.min(results))
print(np.max(results))

kaggle_format.run(results)
'''

