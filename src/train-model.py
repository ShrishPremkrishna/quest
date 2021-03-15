import os
import numpy as np
from tensorflow.keras.preprocessing import image
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D
from tensorflow.keras.metrics import Accuracy, Precision
from tensorflow.keras.losses import Loss
from tensorflow.keras.callbacks import History, Callback


image_shape = (21, 72, 3)

image_gen = ImageDataGenerator(rotation_range=0,
                               width_shift_range=0,
                               height_shift_range=0,
                               rescale=1 / 255,
                               shear_range=0,
                               zoom_range=0,
                               horizontal_flip=False,
                               fill_mode='nearest'
                               )

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(4, 4), input_shape=(21, 72, 3), activation='relu', ))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(4, 4), input_shape=(21, 72, 3), activation='relu', ))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(1024, activation='relu'))

model.add(Dense(512, activation='relu'))

model.add(Dense(128, activation='relu'))

model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=["accuracy"])
print("\n")
print("MODEL SUMMARY START")
print(model.summary())
print("MODEL SUMMARY END")
print("\n")


train_image_gen = image_gen.flow_from_directory('../data/train',
                                                target_size=image_shape[:2],
                                                class_mode='categorical',
                                                batch_size = 444)
#6660 train values last time i checked i think if i am reading the correct thing

test_image_gen = image_gen.flow_from_directory('../data/test',
                                               target_size=image_shape[:2],
                                               class_mode='categorical',
                                               batch_size = 557)
#2785 test values last time i checked i think if i am reading the correct thing

results = model.fit_generator(train_image_gen, epochs=10,
                              steps_per_epoch=15,
                              validation_data=test_image_gen,
                              validation_steps=5
                              )

#REMEMBER TO NAME THE MODEL
model_name = "NAME THIS MODEL"
#REMEMBER TO NAME THE MODEL

history = results.history
print("\n")
print("HISTORY IN THE MAKING")
print("\n")
print("Loss", history['loss'])
print("\n")
print("Accuracy", history['accuracy'])
print("\n")
print("Val_Loss", history['val_loss'])
print("\n")
print("Val_Accuracy", history['val_accuracy'])
print("\n")
print("All History", history)
print("\n")

model.save(model_name)
print("MODEL SAVED")

print("\n")

num_values_of_test_data = 50
print("Using Model Against", num_values_of_test_data, "values of Test Data")
testing_against_test_data = load_model(model_name)
score_control_model = testing_against_test_data.evaluate_generator(test_image_gen, num_values_of_test_data) #(dataset to test on, number of values to use)
print("control model - loss and accuracy", score_control_model)

print("\n")

sample_file = '../data/train/N/N-1-07-77.jpg'#4
sample_img = image.load_img(sample_file, target_size=image_shape[:2])
sample_img = image.img_to_array(sample_img)
sample_img = np.expand_dims(sample_img, axis=0)
the_model = load_model(model_name)
predictions = the_model.predict_classes(sample_img)
print("predictions", predictions, "should be 4(none)")