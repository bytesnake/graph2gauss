import os
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Lambda

from .utils import *

class MyLayer(keras.layers.Layer):
    def __init__(self, L, **kwargs):
        self.L = L
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.mu = Dense(self.L, input_shape=input_shape, activation=None)
        
        self.sigma2 = Dense(self.L, input_shape=input_shape, activation=tf.nn.elu)
        self.sigma = Lambda(lambda x: x + 1 + 1e-14)

        super(MyLayer, self).build(input_shape)

    def call(self,x):
        mu = self.mu(x)
        sigma = self.sigma(self.sigma2(x))

        return tf.concat([mu,sigma],2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.L)
    
    def get_config(self):
        base_config = super(MyLayer, self).get_config()
        base_config['L'] = self.L
        return base_config

class TrainingSequence(keras.utils.Sequence):
    def __init__(self,X, ones, K, batch_size):
        self.X = X
        self.batch_size = batch_size

        A_train = edges_to_sparse(ones, self.X.shape[0])
        self.hops = get_hops(A_train, K)

        self.scale_terms = {h if h != -1 else max(self.hops.keys()) + 1:
                           self.hops[h].sum(1).A1 if h != -1 else self.hops[1].shape[0] - self.hops[h].sum(1).A1
                       for h in self.hops}

        self.on_epoch_end()

    def on_epoch_end(self):
        self.items, self.weights = to_triplets(sample_all_hops(self.hops), self.scale_terms)
        self.size = self.items.shape[0]

    def __getitem__(self, i):
        X_training = self.X[self.items[self.batch_size*i:self.batch_size*(i+1)]]
        weights_training = self.weights[self.batch_size*i:self.batch_size*(i+1)]

        return X_training, weights_training

    def __len__(self):
        return math.floor(self.size / self.batch_size)

class LossFunction:
    def __init__(self, L, scale=False):
        self.L = L
        self.scale = scale

    def custom_loss(self, scale_terms, params):
        pos = tf.stack([params[:,0], params[:,1]],1)
        neg = tf.stack([params[:,0], params[:,2]],1)

        eng_pos = self.energy_kl(pos)
        eng_neg = self.energy_kl(neg)
        energy = tf.square(eng_pos) + tf.exp(-eng_neg)

        if self.scale:
            return tf.reduce_mean(energy * scale_terms)
        else:
            return tf.reduce_mean(energy)

    def energy_kl(self, data):
        """
        Computes the energy of a set of node pairs as the KL divergence between their respective Gaussian embeddings.

        Parameters
        ----------
        pairs : array-like, shape [?, 2]
            The edges/non-edges for which the energy is calculated

        Returns
        -------
        energy : array-like, shape [?]
            The energy of each pair given the currently learned model
        """
        
        sigma_ratio = data[:, 0, self.L:] / data[:, 1, self.L:]

        trace_fac = tf.reduce_sum(sigma_ratio, 1)
        log_det = tf.reduce_sum(tf.math.log(sigma_ratio + 1e-14), 1)

        mu_diff_sq = tf.reduce_sum(tf.square(data[:, 0, 0:self.L] - data[:, 1, 0:self.L]) / data[:, 0, self.L:], 1)

        return 0.5 * (trace_fac + mu_diff_sq - self.L - log_det)

class RocCallback(keras.callbacks.Callback):
    def __init__(self, X, ones, zeros, model,loss):
        X_access = np.concatenate((ones, zeros), axis=0)

        self.model = model
        self.loss = loss
        self.x_input = X[X_access]
        self.y_truth = np.concatenate((np.ones(ones.shape[0]), np.zeros(zeros.shape[0])))
        self.last_val = 0.0

        self.variances = []

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x_input)
        mean_vars = np.mean(y_pred[:,0,:],axis=0)
        self.variances.append(mean_vars)

        np.save("variances.npy", self.variances);

        y_pred = -self.loss.energy_kl(y_pred)
        auc, prec = roc_auc_score(self.y_truth, y_pred), average_precision_score(self.y_truth, y_pred)

        if auc > self.last_val:
            self.last_val = auc
            self.tolerance = 500
            self.model.save_weights('model.h5')
        else:
            self.tolerance -= 1

        print(' ROC auc: {} - ROC precision: {}'.format(auc, prec))

        if self.tolerance == 0:
            self.model.load_weights('model.h5')
            self.model.stop_training = True
            print("ROC auc has not increased for 50 epoch! Stopping ..")

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

    def get(self):
        return (self.x_input, self.y_truth)

class Graph2Gauss:
    """
    Implementation of the method proposed in the paper:
    'Deep Gaussian Embedding of Graphs: Unsupervised Inductive Learning via Ranking'
    by Aleksandar Bojchevski and Stephan Günnemann,
    published at the 6th International Conference on Learning Representations (ICLR), 2018.

    Copyright (C) 2018
    Aleksandar Bojchevski
    Technical University of Munich
    """

    def __init__(self, A, X, L, K=1, p_val=0.002, p_test=0.001, hidden_layers=[512],
                 max_iter=2000, tolerance=100, scale=False, seed=0, verbose=True, model_path=None):
        """
        Parameters
        ----------
        A : scipy.sparse.spmatrix
            Sparse unweighted adjacency matrix
        X : scipy.sparse.spmatrix
            Sparse attribute matirx
        L : int
            Dimensionality of the node embeddings
        K : int
            Maximum distance to consider
        p_val : float
            Percent of edges in the validation set, 0 <= p_val < 1
        p_test : float
            Percent of edges in the test set, 0 <= p_test < 1
        p_nodes : float
            Percent of nodes to hide (inductive learning), 0 <= p_nodes < 1
        hidden_layers : list(int)
            A list specifying the size of each hidden layer, default n_hidden=[512]
        max_iter :  int
            Maximum number of epoch for which to run gradient descent
        tolerance : int
            Used for early stopping. Number of epoch to wait for the score to improve on the validation set
        scale : bool
            Whether to apply the up-scaling terms.
        seed : int
            Random seed used to split the edges into train-val-test set
        verbose : bool
            Verbosity.
        """
        tf.random.set_seed(seed)
        np.random.seed(seed)

        # ensure that the attribute matrix constists of f32
        #X = X.astype(np.float32)

        #self.X = X.toarray()
        self.X = X

        self.N, D = X.shape
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.scale = scale
        self.verbose = verbose

        layer_sizes = [D] + hidden_layers + [L]
        self.loss = LossFunction(L, scale)


        if model_path:
            self.model = tf.keras.models.load_model(model_path, custom_objects={'MyLayer': MyLayer,'custom_loss': self.loss.custom_loss})
        else:
            self.model = Sequential()
            self.model.add(Dense(layer_sizes[1], input_shape=(None, layer_sizes[0]), activation=tf.nn.relu))

            for size in layer_sizes[2:-1]:
                self.model.add(Dense(size, activation=tf.nn.relu))

            self.model.add(MyLayer(layer_sizes[-1]))

        self.model.compile(loss=self.loss.custom_loss, optimizer='adam', metrics=[])

        # hold out some validation and/or test edges
        # pre-compute the hops for each node for more efficient sampling
        if p_val + p_test > 0:
            train_ones, val_ones, val_zeros, self.test_ones, self.test_zeros = train_val_test_split_adjacency(
                A=A, p_val=p_val, p_test=p_test, seed=seed, neg_mul=1, every_node=True, connected=False,
                undirected=(A != A.T).nnz == 0)
            
            print("Generated {} training samples, {} validation samples, {} test samples".format(train_ones.shape[0], val_ones.shape[0] + val_zeros.shape[0], self.test_ones.shape[0] + self.test_zeros.shape[0]))

            self.training_generator = TrainingSequence(self.X, train_ones, K, 128)
            self.roc = RocCallback(self.X, val_ones, val_zeros, self.model, self.loss)

        else:
            pass

    def train(self, gpu_list='0'):
        """
        Trains the model.

        Parameters
        ----------
        gpu_list : string
            A list of available GPU devices.

        Returns
        -------
        sess : tf.Session
            Tensorflow session that can be used to obtain the trained embeddings

        """

        history = self.model.fit_generator(self.training_generator, epochs=self.max_iter, callbacks=[self.roc])
        self.model.summary()

    def test(self):
        X_access = np.concatenate((self.test_ones, self.test_zeros), axis=0)
        y_truth = np.concatenate((np.ones(self.test_ones.shape[0]), np.zeros(self.test_zeros.shape[0])))

        y_pred = self.model.predict(self.X[X_access])
        import pdb; pdb.set_trace()
        mean_vars = np.mean(y_pred[:,0,1],axis=0)
        print(mean_vars)

        y_pred = -self.loss.energy_kl(y_pred)
        fpr, tpr, threshold = roc_curve(y_truth, y_pred)
        roc_auc = roc_auc_score(y_truth, y_pred)

        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    def save(self, out):
        self.model.save(out, save_format="tf")
        self.model.save(out + "/model.h5", save_format='h5')
