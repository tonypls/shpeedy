import h5py
import tensorflow as tf
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import createbatches
import os
import numpy as np

sports_1m = h5py.File('data/c3d-sports1M_weights.h5', mode='r')

def train(model, TIMESTEPS):
    for i in range(len(model.layers)):
        layer = model.layers[i]
        layer_name = 'layer_' + str(i)

        weights = sports_1m[layer_name].values()
        weights = [weight.value for weight in weights]
        weights = [weight if len(weight.shape) < 4 else weight.transpose(2, 3, 4, 1, 0) for weight in weights]

        layer.set_weights(weights)
        # ignore the last 2 layer, 1 dropout and 1 dense
        if i > len(model.layers) - 3:
            break

    weight_path = "data/weight_c3d.h5"
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=2, mode='min', verbose=1)
    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

    callbacks_list = [checkpoint]
    model.save_weights(weight_path)

    data_folder = 'data/train/'
    split_ratio = 0.90

    indices = list(range(TIMESTEPS-1, len(os.listdir(data_folder + 'images/')), TIMESTEPS))
    np.random.shuffle(indices)

    train_indices = list(indices[0:int(len(indices)*split_ratio)])
    valid_indices = list(indices[int(len(indices)*split_ratio):])

    gen_train = createbatches.BatchGenerator(data_folder, train_indices, batch_size=4, timesteps=TIMESTEPS)
    gen_valid = createbatches.BatchGenerator(data_folder, valid_indices, batch_size=4, timesteps=TIMESTEPS, jitter = False)


    def custom_loss(y_true, y_pred):
        loss = tf.squared_difference(y_true, y_pred)
        loss = tf.reduce_mean(loss)

        return loss

    model.load_weights(weight_path)

    tb_counter  = max([int(num) for num in os.listdir('../logs/speed/')] or [0]) + 1
    tensorboard = TensorBoard(log_dir='../logs/speed/' + str(tb_counter), histogram_freq=0, write_graph=True, write_images=False)

    sgd = SGD(lr=1e-5, decay=0.001, momentum=0.9)
    model.compile(loss=custom_loss, optimizer=sgd)

    #adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #model.compile(loss=custom_loss, optimizer=adam)

    #rms = RMSprop(lr=1e-3, rho=0.9, epsilon=1e-08, decay=0.0)
    #model.compile(loss=custom_loss, optimizer=rms)

    model.fit_generator(generator = gen_train.get_gen(),
                        steps_per_epoch = gen_train.get_size(),
                        epochs  = 10,
                        verbose = 1,
                        validation_data = gen_valid.get_gen(),
                        validation_steps = gen_valid.get_size(),
                        callbacks = [early_stop, checkpoint, tensorboard],
                        max_q_size = 8)

    return model

