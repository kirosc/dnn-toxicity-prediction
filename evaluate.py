import numpy as np
import tensorflow as tf
import csv
import glob

test_labels = []

test_data = np.load('NR-ER-test/names_onehots.npy')

with open('NR-ER-test/names_labels.csv', 'r') as csvfile:
    # Reading the csv file
    rows = csv.reader(csvfile)

    # Transfer the labels into list
    for row in rows:
        content = row[1]
        test_labels.append(int(content))

    test_labels = np.array(test_labels)

test_data = test_data.item()
test_names = np.array(test_data['names'])
test_smiles = np.array(test_data['onehots']).reshape(-1, 72, 398, 1)


for file in glob.glob(r'*.model'):
    model = tf.keras.models.load_model(file)
    print(file)
    test_results = model.evaluate(test_smiles, test_labels)
    print(test_results)
    print()
    print('---------------------------')
