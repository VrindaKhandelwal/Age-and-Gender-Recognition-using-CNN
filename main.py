import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

img_size = 150 # set universal size for images -- images will also be grayscale
batch_size = 32  # set universal batch size -- images processed per step

# load a saved gender detection model
def load_gender_model ():
	model = make_gender_model()
	model.load_weights("gender_model.h5")
	return model

# create the CNN to detect gender
def make_gender_model ():

	model = Sequential()
	model.add(Conv2D(32, kernel_size=7, input_shape=(img_size, img_size, 1), activation="relu"))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(32, kernel_size=5, activation="relu"))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(32, kernel_size=3, activation="relu"))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))

	model.add(Dense(1, activation='sigmoid'))

	model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['mse', 'binary_accuracy'])
	return model

# train a gender detection CNN
def train_gender_model ():

	model = make_gender_model()	# create the skeleton for the CNN

	# we create an image data generator that randomly modifies an image
	# so that we have better & more noisy data to train on
	# we will get better generalisation performance
	train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.01,
        zoom_range=0.01,
        horizontal_flip=True)

	# generator will read pictures found in the specified directory
	# classes are detected based on the folder
	# wil create random batches of the data it finds
	train_generator = train_datagen.flow_from_directory(
        'CLEANED/gender',  # this is the target directory
        target_size=(img_size, img_size),  # all images will be resized to 150x150
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

	# this is a similar generator, for validation data
	validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
        'CLEANED/test_gender',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='binary')

	# finally, we train the model
	model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=5,
        validation_data=validation_generator,
        validation_steps=500 // batch_size)

	# print out what each class means
	print("class indices = " + str(train_generator.class_indices))
	
	# save the model
	model.save_weights('gender_model.h5')  # always save your weights after training or during training
	return model

def get_gender_confusion_matrix (model):
	validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
        'CLEANED/test_gender',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='binary',
        shuffle=False)


	y_pred = model.predict_generator(validation_generator)
	y_pred = [ (1 if y[0] > 0.5 else 0) for y in y_pred]

	print(sum(abs(validation_generator.classes-y_pred)))

	print('Confusion Matrix')
	cm = confusion_matrix(validation_generator.classes, y_pred)
	print(cm)

	plt.imshow(cm, cmap=plt.cm.Blues)
	plt.xlabel("Predicted labels")
	plt.ylabel("True labels")
	plt.xticks([], [])
	plt.yticks([], [])
	plt.title('Gender Confusion Matrix')
	plt.colorbar()
	plt.show()

def load_age_model ():
	model = make_age_model()
	model.load_weights("age_model.h5")
	return model

def make_age_model ():

	model = Sequential()
	model.add(Conv2D(32, kernel_size=3, input_shape=(img_size, img_size, 1), activation="relu"))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(32, kernel_size=3, activation="relu"))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(32, kernel_size=3, activation="relu"))
	model.add(MaxPooling2D(pool_size=(2, 2)))


	model.add(Flatten())
	model.add(Dense(500, activation='relu'))
	model.add(Dropout(0.1))

	model.add(Dense(400, activation='relu'))
	model.add(Dropout(0.1))

	model.add(Dense(8))
	model.add(Activation('sigmoid'))

	model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['categorical_accuracy'])
	return model

def train_age_model ():

	model = make_age_model()	

	train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.01,
        horizontal_flip=True)

	train_generator = train_datagen.flow_from_directory(
        'CLEANED/age',  # this is the target directory
        target_size=(img_size, img_size),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical')  

	validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
        'CLEANED/test_age',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical')

	model.fit_generator(
        train_generator,
        steps_per_epoch=5000 // batch_size,
        epochs=5,
        validation_data=validation_generator,
        validation_steps=500 // batch_size)

	print("class indices = " + str(train_generator.class_indices))
	
	model.save_weights('age_model.h5')  # always save your weights after training or during training
	return model

def get_age_confusion_matrix (model):
	validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
        'CLEANED/test_age',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=False)


	y_pred = model.predict_generator(validation_generator)
	y_pred = np.argmax(y_pred, axis=1)

	print('Confusion Matrix')
	cm = confusion_matrix(validation_generator.classes, y_pred)
	print(cm)

	plt.imshow(cm, cmap=plt.cm.Blues)
	plt.xlabel("Predicted labels")
	plt.ylabel("True labels")
	plt.xticks([], [])
	plt.yticks([], [])
	plt.title('Age Confusion Matrix')
	plt.colorbar()
	plt.show()


img = image.load_img('test5.png', target_size=(img_size, img_size), color_mode='grayscale')
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])


#gmodel = load_gender_model()
#get_gender_confusion_matrix(gmodel)

#classes = gmodel.predict_classes(images, batch_size=batch_size)

#if classes[0][0] == 1:
#	print("person in image is male")
#else:
#	print("person in image is female")

amodel = train_gender_model()
get_gender_confusion_matrix(amodel)

classes = amodel.predict_classes(images, batch_size=batch_size)

print(classes)