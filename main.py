import tensorflow as tf
import logging
import numpy as np
import matplotlib.pyplot as plt
import cv2

tf.get_logger().setLevel(logging.ERROR)

main_path = './dataset/'
img_size = (64, 64)
batch_size = 64 # broj elemenata koji ce proci kroz mrezu u isto vreme u toku treniranja

from keras.utils import image_dataset_from_directory

Xtrain = image_dataset_from_directory('./train/',
 image_size=img_size,
 batch_size=batch_size,
 seed=123)
Xval = image_dataset_from_directory('./valid/',
 image_size=img_size,
 batch_size=batch_size,
 seed=123)
Xtest = image_dataset_from_directory('./test/',
 image_size=img_size,
 batch_size=batch_size,
 seed=123)
classes = Xtrain.class_names
print(classes)

import os
import matplotlib.pyplot as plt

train_dir = './train/'
valid_dir = './valid/'
test_dir = './test/'

# Get the class labels
classes = sorted(os.listdir(train_dir))

# Initialize counts
train_counts = []
valid_counts = []
test_counts = []

# Count the number of samples per class in train dataset
for class_label in classes:
    class_path = os.path.join(train_dir, class_label)
    count = len(os.listdir(class_path))
    train_counts.append(count)

# Count the number of samples per class in validation dataset
for class_label in classes:
    class_path = os.path.join(valid_dir, class_label)
    count = len(os.listdir(class_path))
    valid_counts.append(count)

# Count the number of samples per class in test dataset
for class_label in classes:
    class_path = os.path.join(test_dir, class_label)
    count = len(os.listdir(class_path))
    test_counts.append(count)

# Calculate the total number of samples per class
total_counts = [train + valid + test for train, valid, test in zip(train_counts, valid_counts, test_counts)]

# Plotting the class distribution
plt.figure(figsize=(4, 6))
plt.bar(classes, total_counts)
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.title('Total Number of Cards per Class')
plt.xticks(rotation=90)
plt.show()

N = 10
plt.figure()
for img, lab in Xtrain.take(1):
  for i in range(N):
   plt.subplot(2, int(N/2), i+1)
   plt.imshow(img[i].numpy().astype('uint8'))
   plt.title(classes[lab[i]])
   plt.axis('off')

plt.show()

from keras import layers
from keras import Sequential
data_augmentation = Sequential(
 [
 layers.RandomFlip("horizontal", input_shape=(img_size[0], img_size[1], 3)),
 layers.RandomRotation(0.25),
 layers.RandomZoom(0.1),
 ]
)

N = 10
plt.figure()
for img, lab in Xtrain.take(1):
  plt.title(classes[lab[0]])
  for i in range(N):
   aug_img = data_augmentation(img)
   plt.subplot(2, int(N/2), i+1)
   plt.imshow(aug_img[0].numpy().astype('uint8'))
   plt.axis('off')


from keras import Sequential
from keras import layers
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
num_classes = len(classes)
model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape=(64, 64, 3)),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
model.summary()
model.compile(Adam(learning_rate=0.001),
 loss=SparseCategoricalCrossentropy(),
 metrics='accuracy')

history = model.fit(Xtrain,
 epochs=50,
 validation_data=Xval,
 verbose=0)

print("posle history")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
print("posle history 2")
plt.figure()
plt.subplot(121)
plt.plot(acc)
plt.plot(val_acc)
plt.title('Accuracy')
plt.subplot(122)
plt.plot(loss)
plt.plot(val_loss)
plt.title('Loss')
print("pre show")
plt.show()

# labels = np.array([])
# pred = np.array([])
# for img, lab in Xval:
#  labels = np.append(labels, lab)
#  pred = np.append(pred, np.argmax(model.predict(img, verbose=0), axis=1))
#

#
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# cm = confusion_matrix(labels, pred, normalize='true')
# cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
# plt.figure(figsize=(15, 15))  # Increase the figure size
# cmDisplay.plot()
# plt.xticks(rotation='vertical')  # Rotate x-axis tick labels
# plt.show()

# matrica za trening set
labels = np.array([])
pred = np.array([])
for img, lab in Xtrain:
     labels = np.append(labels, lab)
     pred = np.append(pred, np.argmax(model.predict(img, verbose=1), axis=1))


from sklearn.metrics import accuracy_score
print('Tačnost Xval modela je: ' + str(100*accuracy_score(labels, pred)) + '%')

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(labels, pred, normalize='true')
cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
cmDisplay.plot(xticks_rotation='vertical')
plt.show()


labels1  = np.array([])
pred1 = np.array([])

for img, lab in Xtest:
    labels1 = np.append(labels1, lab)
    pred1 = np.append(pred1, np.argmax(model.predict(img, verbose=1), axis=1))

cm1 = confusion_matrix(labels1, pred1, normalize='true')
cmDisplay1 = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=classes)
cmDisplay1.plot()
plt.show()

print('Tačnost Xtest modela je: ' + str(100*accuracy_score(labels1,pred1))+'%')
