import csv
import numpy as np
import time
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import BatchNormalization, Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.callbacks import TensorBoard

epoach = 15

NAME = 'CNN-{}'.format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/epoach-{}-{}'.format(epoach, NAME), histogram_freq=1)

labels = []
test_labels = []

data = np.load('NR-ER-train/names_onehots.npy')
test_data = np.load('NR-ER-test/names_onehots.npy')

with open('NR-ER-train/names_labels.csv', 'r') as csvfile:
    # data = np.load('../NR-ER-score/names_onehots.npy')
    # with open('../NR-ER-score/names_labels.csv', 'r') as csvfile:
    # Reading the csv file
    rows = csv.reader(csvfile)

    # Transfer the labels into list
    for row in rows:
        content = row[1]
        labels.append(int(content))

with open('NR-ER-test/names_labels.csv', 'r') as csvfile:
    # Reading the csv file
    rows = csv.reader(csvfile)

    # Transfer the labels into list
    for row in rows:
        content = row[1]
        test_labels.append(int(content))

    test_labels = np.array(test_labels)

data = data.item()
names = np.array(data['names'])
smiles = np.array(data['onehots']).reshape(-1, 72, 398, 1)

test_data = test_data.item()
test_names = np.array(test_data['names'])
test_smiles = np.array(test_data['onehots']).reshape(-1, 72, 398, 1)

# print(smiles)
model = Sequential()

model.add(Conv2D(8, (2, 2), input_shape=smiles.shape[1:], kernel_regularizer=regularizers.l2(0.02)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(8, (2, 2), kernel_regularizer=regularizers.l2(0.02)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(8, (2, 2), kernel_regularizer=regularizers.l2(0.02)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Use TensorBoard to visualize the TensorFlow graph
model.fit(smiles, labels, batch_size=32, epochs=epoach, validation_data=(test_smiles, test_labels),
          callbacks=[tensorboard])

test_results = model.evaluate(test_smiles, test_labels)
print(test_results)

model.save('{}.model'.format(NAME))
