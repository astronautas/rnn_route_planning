import numpy as np
import tensorflow.keras
from random import shuffle
import utils

class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, X, Y, batch_size=32, shuffle=False, model=None, graph=None):

        'Initialization'
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.model = model
        self.graph = graph

        # self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X) / self.batch_size))

    def shuffled(self, X, Y):
        zipped = list(zip(X, Y))
        shuffle(zipped)
        return zip(*zipped)

    def shuffled_groups(self, X, Y, group_size):
        groups_X = utils.chunks(X, group_size)
        groups_Y = utils.chunks(Y, group_size)
        
        x, y = self.shuffled(groups_X, groups_Y)

        x = utils.flatten(x)
        y = utils.flatten(y)

        return x, y

    # Generate a batch item
    # TODO: try imitation learning (https://arxiv.org/pdf/1607.05241.pdf)
    def __getitem__(self, index):
        # X = self.X[index * self.batch_size:index * self.batch_size + self.batch_size]
        # Y = self.Y[index * self.batch_size:index * self.batch_size + self.batch_size]

        if self.shuffle:
            x, y = self.shuffled(self.X[index * self.batch_size:index * self.batch_size + self.batch_size],
                            self.Y[index * self.batch_size:index * self.batch_size + self.batch_size])

            X = np.asarray(x)
            Y = np.asarray(y)
        else:
            X = np.asarray(self.X[index * self.batch_size:index * self.batch_size + self.batch_size])
            Y = np.asarray(self.Y[index * self.batch_size:index * self.batch_size + self.batch_size])

        # X_upd = np.empty_like(X)
        # Y_upd = np.empty_like(Y)

        # for ex_i in range(0, len(X)):
        #     curr_episode_X = np.empty_like(X[ex_i])
        #     curr_episode_Y = np.empty_like(Y[ex_i])

        #     curr_episode_X[0] = X[ex_i][0]
        #     curr_episode_Y[0] = Y[ex_i][0]

        #     for t in range(1, len(Y[ex_i])):
        #         if np.random.rand() < 0.5:
        #             curr_episode_X[t] = X[ex_i][t]
        #             curr_episode_Y[t] = Y[ex_i][t]
        #         else:
        #             curr_episode_X[t] = X[ex_i][t]
        #             curr_episode_Y[t] = Y[ex_i][t]
        #             # Predict X at t
        #             curr_node = X[ex_i][t-1]
        #             goal_node = X[ex_i][-1]

        #             next_node_positional_index = self.model.predict_classes(X[ex_i][t-1], batch_size=1).tolist()[0][0]
        #             x0, ng_ids = osmnx_utils.get_ng_data(self.graph, curr_node, goal_node)
        #             next_node_id = ng_ids[next_node_index]

        #             curr_episode_X[t] = osmnx_utils.get_ng_data(self.graph, next_node_id, X[ex_i][len(Y[ex_i])-1])
        #             curr_episode_Y[t] = Y[ex_i][t]
            
        #     X_upd[ex_i] = curr_episode_X
        #     Y_upd[ex_i] = curr_episode_Y

        return X, Y

    def on_epoch_begin(self):
        'Updates indexes after each epoch'

        if self.shuffle:
            print("Begin epoch shuffle...")
            
            x, y = self.shuffled_groups(self.X, self.Y, self.batch_size)

            self.X = x
            self.Y = y

    # def __data_generation(self, list_IDs_temp):
    #     'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    #     # Initialization
    #     X = np.empty((self.batch_size, *self.dim, self.n_channels))
    #     y = np.empty((self.batch_size), dtype=int)

    #     # Generate data
    #     for i, ID in enumerate(list_IDs_temp):
    #         # Store sample
    #         X[i,] = np.load('data/' + ID + '.npy')

    #         # Store class
    #         y[i] = self.labels[ID]

    #     return X, tensorflow.keras.utils.to_categorical(y, num_classes=self.n_classes)