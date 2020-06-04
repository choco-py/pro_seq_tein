import os
import tensorflow as tf
import numpy as np

class NeuralNetworkModel(object):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def input(self, *args, **kwargs):
        pass

    def build(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        pass


class ProteinSeq(NeuralNetworkModel):

    def __init__(
        self,
        input_shape=(None, 410, 528),
        batch_size=128,
        buffer_size=1000,
        dropout=0.7,
    ):

        super().__init__()


        self.input_shape = input_shape

        self.EPOCH_NUM = 0
        self.BUFFER_SIZE = buffer_size
        self.BATCH_SIZE = batch_size
        self.DROPOUT = dropout

        self._x_tensor = None
        self._x_batch_tensor = None

        self.next_batch = None
        self.data_init_op = None
        self.data_init_op_eval = None

        self.variable_init_op = None
        self.train_op = None
        self.loss = None
        self.accuracy = None
        self.global_step = None

        # Build Input and network

        self.input(
            input_shape=self.input_shape,
        )

        self.build()

        
