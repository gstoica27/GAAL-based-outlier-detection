from keras.layers import Input, Dense
from keras.models import Sequential, Model
from keras.optimizers import SGD
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import keras
import math
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Run MO-GAAL.")
    parser.add_argument('--path', nargs='?', default='Data/Annthyroid',
                        help='Input data path.')
    parser.add_argument('--data_dir', help='Input data directory', default='Data/EDA')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of sub_generator.')
    parser.add_argument('--stop_epochs', type=int, default=25,
                        help='Stop training generator after stop_epochs.')
    parser.add_argument('--lr_d', type=float, default=0.01,
                        help='Learning rate of discriminator.')
    parser.add_argument('--lr_g', type=float, default=0.0001,
                        help='Learning rate of generator.')
    parser.add_argument('--decay', type=float, default=1e-6,
                        help='Decay.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum.')
    return parser.parse_args()

# Generator
def create_generator(latent_size):
    gen = Sequential()
    gen.add(Dense(latent_size, input_dim=latent_size, activation='relu', kernel_initializer=keras.initializers.Identity(gain=1.0)))
    gen.add(Dense(latent_size, activation='relu', kernel_initializer=keras.initializers.Identity(gain=1.0)))
    latent = Input(shape=(latent_size,))
    fake_data = gen(latent)
    return Model(latent, fake_data)

# Discriminator
def create_discriminator():
    dis = Sequential()
    dis.add(Dense(math.ceil(math.sqrt(data_size)), input_dim=latent_size, activation='relu', kernel_initializer= keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
    dis.add(Dense(1, activation='sigmoid', kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
    data = Input(shape=(latent_size,))
    fake = dis(data)
    return Model(data, fake)

# Load data
def load_data():
    data = pd.read_table('{path}'.format(path = args.path), sep=',', header=None)
    data = data.sample(frac=1).reset_index(drop=True)
    id = data.pop(0)
    y = data.pop(1)
    data_x = data.as_matrix()
    data_id = id.values
    data_y = y.values
    return data_x, data_y, data_id

def load_train_data():
    train_file = os.path.join(args.data_dir, 'train_features.txt')
    data_x = np.loadtxt(train_file)
    data_y = np.array(['nor'] * len(data_x))
    data_id = np.arange(len(data_x))
    return data_x, data_y, data_id

def load_test_data(file_path):
    data_x = np.loadtxt(file_path)
    data_id = np.arange(data_x)
    return data_x, data_id

# Plot loss history
def plot(train_history, name):
    dy = train_history['discriminator_loss']
    gy = train_history['generator_loss']
    auc_y = train_history['auc']
    for i in range(k):
        names['gy_' + str(i)] = train_history['sub_generator{}_loss'.format(i)]
    x = np.linspace(1, len(dy), len(dy))
    fig, ax = plt.subplots()
    ax.plot(x, dy, color='blue')
    ax.plot(x, gy,color='red')
    ax.plot(x, auc_y, color='yellow', linewidth = '3')
    for i in range(k):
        ax.plot(x, names['gy_' + str(i)], color='green', linewidth='0.5')
    plt.show()

if __name__ == '__main__':
    train = True

    # initilize arguments
    args = parse_args()

    # initialize dataset
    data_x, data_y, data_id = load_train_data() #load_data()
    data_size = data_x.shape[0]
    latent_size = data_x.shape[1]
    print("The dimension of the training data :{}*{}".format(data_size, latent_size))

    if train:
        train_history = defaultdict(list)
        names = locals()
        epochs = args.stop_epochs * 3
        stop = 0
        k = args.k

        # Create discriminator
        discriminator = create_discriminator()
        discriminator.compile(optimizer=SGD(lr=args.lr_d, decay=args.decay, momentum=args.momentum), loss='binary_crossentropy')

        # Create k combine models
        for i in range(k):
            names['sub_generator' + str(i)] = create_generator(latent_size)
            latent = Input(shape=(latent_size,))
            names['fake' + str(i)] = names['sub_generator' + str(i)](latent)
            discriminator.trainable = False
            names['fake' + str(i)] = discriminator(names['fake' + str(i)])
            names['combine_model' + str(i)] = Model(latent, names['fake' + str(i)])
            names['combine_model' + str(i)].compile(optimizer=SGD(lr=args.lr_g, decay=args.decay, momentum=args.momentum), loss='binary_crossentropy')

        # Start iteration
        for epoch in range(epochs):
            print('Epoch {} of {}'.format(epoch + 1, epochs))
            batch_size = min(500, data_size)
            num_batches = int(data_size / batch_size)

            for index in range(num_batches):
                print('\nTesting for epoch {} index {}:'.format(epoch + 1, index + 1))

                # Generate noise
                noise_size = batch_size
                noise = np.random.uniform(0, 1, (int(noise_size), latent_size))

                # Get training data
                data_batch = data_x[index * batch_size: (index + 1) * batch_size]

                # Generate potential outliers
                block = ((1 + k) * k) // 2
                for i in range(k):
                    if i != (k-1):
                        noise_start = int((((k + (k - i + 1)) * i) / 2) * (noise_size // block))
                        noise_end = int((((k + (k - i)) * (i + 1)) / 2) * (noise_size // block))
                        names['noise' + str(i)] = noise[noise_start : noise_end ]
                        names['generated_data' + str(i)] = names['sub_generator' + str(i)].predict(names['noise' + str(i)], verbose=0)
                    else:
                        noise_start = int((((k + (k - i + 1)) * i) / 2) * (noise_size // block))
                        names['noise' + str(i)] = noise[noise_start : noise_size]
                        names['generated_data' + str(i)] = names['sub_generator' + str(i)].predict(names['noise' + str(i)], verbose=0)

                # Concatenate real data to generated data
                for i in range(k):
                    if i == 0:
                        X = np.concatenate((data_batch, names['generated_data' + str(i)]))
                    else:
                        X = np.concatenate((X, names['generated_data' + str(i)]))
                Y = np.array([1] * batch_size + [0] * int(noise_size))

                # Train discriminator
                discriminator_loss = discriminator.train_on_batch(X, Y)
                train_history['discriminator_loss'].append(discriminator_loss)

                # Get the target value of sub-generator
                p_value = discriminator.predict(data_x)
                p_value = pd.DataFrame(p_value)
                for i in range(k):
                    names['T' + str(i)] = p_value.quantile(i/k)
                    names['trick' + str(i)] = np.array([float(names['T' + str(i)])] * noise_size)

                # Train generator
                noise = np.random.uniform(0, 1, (int(noise_size), latent_size))
                if stop == 0:
                    for i in range(k):
                        names['sub_generator' + str(i) + '_loss'] = names['combine_model' + str(i)].train_on_batch(noise, names['trick' + str(i)])
                        train_history['sub_generator{}_loss'.format(i)].append(names['sub_generator' + str(i) + '_loss'])
                else:
                    for i in range(k):
                        names['sub_generator' + str(i) + '_loss'] = names['combine_model' + str(i)].evaluate(noise, names['trick' + str(i)])
                        train_history['sub_generator{}_loss'.format(i)].append(names['sub_generator' + str(i) + '_loss'])

                generator_loss = 0
                for i in range(k):
                    generator_loss = generator_loss + names['sub_generator' + str(i) + '_loss']
                generator_loss = generator_loss / k
                train_history['generator_loss'].append(generator_loss)

                # Stop training generator
                if epoch +1  > args.stop_epochs:
                    stop = 1

        # Detection result
        print('Obtaining test data predictions...')
        test_dir = os.path.join(args.data_dir, 'test_features')
        predictions_dir = os.path.join(args.data_dir, 'test_predictions')
        os.makedirs(predictions_dir, exist_ok=True)
        for filename in os.listdir(test_dir):
            test_file = os.path.join(test_dir, filename)
            data_x, data_id = load_test_data(test_file)
            p_value = discriminator.predict(data_x)
            # assign save path
            basename = os.path.splitext(filename)[0]
            save_file = os.path.join(predictions_dir, basename + '.txt')
            print('Saving predictions to {}....'.format(save_file))
            np.savetxt(save_file, p_value)