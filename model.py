import tensorflow as tf
from random import shuffle
import numpy as np
import os

ONEHOTS = {'TBD': [], 'TBD': [],'TBD': [], 'TBD': []}
SPLIT = .9

def main():
    print('Reading data from /data')
    labeled_data = []
    for file in os.listdir('data'):
        arr = np.load('data/' + str(file))
        data_type = str(file)[0:str(file).find('-')] # Slice the filename to just extract the data type
        labeled_data.append(arr,ONEHOTS[data_type]) # Add the data with its appropriate one-hot encoding to the list

    shuffle(labeled_data)

    samples = []
    labels = []
    for sample, label in labeled_data:
        samples.append(sample)
        labels.append(label)

    cutoff = int(len(labeled_data)*SPLIT)
    train_data = np.asarray(samples[:cutoff])
    test_data = np.asarray(samples[cutoff:])
    train_labels = np.asarray(labels[:cutoff])
    test_labels = np.asarray(labels[cutoff:])


    print(str(len(labeled_data)/5) + ' seconds of data found.')

    #I'm gonna try using a convolutional structure here, I'm not sure how this is gonna play with the interaction between channel pulls however. Iterative design
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size = 3, input_shape = (20,60)),
        tf.keras.layers.MaxPooling1D(),
        tf.keras.layers.Conv1D(filters=32,kernel_size=3),
        tf.keras.layers.MaxPooling1D(),
        tf.keras.layers.FLatten(),
        tf.keras.layers.Dense(512,activation='relu'),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
    print(model.summary())
    #Maybe play around with adding some callbacks, but I don't think this is complex enough to need them. TODO?
    model.fit(x = train_data, y = train_labels, validation_data = (test_data,test_labels), epochs = 15)

    print('Example prediction: ')
    print(train_labels[0])
    print(model.predict(train_data[0]))

    print('Save model?')
    tosave = str(input).lower()
    if tosave == 'y':
        name = str(input('Input name:'))
        model.save('models/' + name + '.h5')
        print('Model saved')


if __name__ == '__main__':
    main()
