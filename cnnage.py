import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
image_size=180
#batch_size=64

#class_weight={'0, 2':0.222,'4, 6':0.114,'8, 12':0.048,'15, 20': 0.034,'25, 32': 0.374, '38, 43': 0.126, '48, 53': 0.049, '60, 100': 0.033}
#class_weight={'0, 2': 0.222, '15, 20': 0.034, '25, 32': 0.374, '38, 43': 0.126, '4, 6': 0.114, '48, 53': 0.049, '60, 100': 0.033, '8, 12': 0.048}
#class indices = {'0, 2': 0, '15, 20': 1, '25, 32': 2, '38, 43': 3, '4, 6': 4, '48, 53': 5, '60, 100': 6, '8, 12': 7}
#class_weight={0:  3, 1: 14, 2: 1, 3: 0.126, 4: 0.114 5, 5: 0.049 12, 6: 0.033 14, 7: 0.048 12}
#class_weight={0:  3, 1: 14, 2: 1, 3:  4, 4: 5, 5:  12, 6:  14, 7:  12}
model=Sequential()



model.add(Conv2D(32, kernel_size=(4, 4),input_shape=(image_size, image_size, 1),
                 activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(200, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.1, name='Dropout_Regularization'))
model.add(Dense(400, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(8, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['categorical_accuracy'])




# model.add(prior)
# model.add(Flatten())
# model.add(Dense(256, activation='relu', name='Dense_Intermediate'))
# model.add(Dropout(0.1, name='Dropout_Regularization'))
# model.add(Dense(12, activation='sigmoid', name='Output'))

datagen = ImageDataGenerator(rescale=1./255,horizontal_flip=True)
traindata=datagen.flow_from_directory('CLEANED/age',color_mode='grayscale', class_mode='categorical',target_size=(image_size, image_size), batch_size=32)
testdata=datagen.flow_from_directory('CLEANED/test_age', color_mode='grayscale',class_mode='categorical',target_size=(image_size, image_size), batch_size=32)
classes=traindata.classes
class_weights = class_weight.compute_class_weight('balanced', np.unique(classes),classes)
model.fit_generator(traindata, steps_per_epoch=60, epochs=5, validation_data=testdata, validation_steps=10)
model.save_weights('agetrain_model.h5')
batchX, batchy = traindata.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
print("class indices = " + str(traindata.class_indices))


predictions = model.predict_generator(testdata)
predictions= np.argmax(predictions, axis=1)

print('THE CONDUSION MATRIX is')
confuse= confusion_matrix(testdata.classes, predictions)
print(confuse)



classes = model.predict_classes(images, batch_size=batch_size)

print(classes)
