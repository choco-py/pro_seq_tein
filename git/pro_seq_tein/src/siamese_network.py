import os
import sys
import shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import importlib
from math import ceil
from math import ceil
# from keras.regularizers import l2,l1

class NoValidationException(Exception):
    def __init__(self, valid_value):
        self.valid_value = str(valid_value)
    def __str__(self):
        msg = "Training with NO-VALIDATION: 'VALIDATION_RATIO=%s'"
        return repr(msg % self.valid_value)

tf.reset_default_graph()


class NeuralNetworkModel(object):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def input(self, *args, **kwargs):
        pass

    def build(self, *args, **kwargs):
        pass

    def train_SiamNet(self, *args, **kwargs):
        pass

# Model Classes --------------------------------------------------------------
class SiameseNet(NeuralNetworkModel):

    def __init__(
        self,
        input_finger1_shape=(None, 1024),
        input_finger2_shape=(None, 1024),
        input_finger3_shape=(None, 1024),
        input_finger4_shape=(None, 1024),
        input_finger5_shape=(None, 1024),
        input_finger6_shape=(None, 1024),
        input_finger7_shape=(None, 166),
        input_protein_shape=(None, 20),
        batch_size=128,
        buffer_size=1000,
        dropout=0.7,
        ):

        super().__init__()

        self.input_finger1_shape = input_finger1_shape
        self.input_finger2_shape = input_finger2_shape
        self.input_finger3_shape = input_finger3_shape
        self.input_finger4_shape = input_finger4_shape
        self.input_finger5_shape = input_finger5_shape
        self.input_finger6_shape = input_finger6_shape
        self.input_finger7_shape = input_finger7_shape
        self.input_protein_shape = input_protein_shape


        self.EPOCH_NUM = 0
        self.BUFFER_SIZE = buffer_size            # For tf.Dataset.suffle(buffer_size)
        self.BATCH_SIZE = batch_size
        self.DROPOUT = dropout

        self._x_fp1_tensor = None
        self._x_fp2_tensor = None
        self._x_fp3_tensor = None
        self._x_fp4_tensor = None
        self._x_fp5_tensor = None
        self._x_fp6_tensor = None
        self._x_fp7_tensor = None
        self._x_protein_tensor = None
        self._x_label_tensor = None
        #
        # self._y_fp1_tensor = None
        # self._y_fp2_tensor = None
        # self._y_fp3_tensor = None
        # self._y_fp4_tensor = None
        # self._y_fp5_tensor = None
        # self._y_fp6_tensor = None
        # self._y_fp7_tensor = None
        # self._y_protein_tensor = None
        # self._y_label_tensor = None

        self._x_fp1_batch_tensor = None
        self._x_fp2_batch_tensor = None
        self._x_fp3_batch_tensor = None
        self._x_fp4_batch_tensor = None
        self._x_fp5_batch_tensor = None
        self._x_fp6_batch_tensor = None
        self._x_fp7_batch_tensor = None
        self._x_protein_batch_tensor = None

        self._y_fp1_batch_tensor = None
        self._y_fp2_batch_tensor = None
        self._y_fp3_batch_tensor = None
        self._y_fp4_batch_tensor = None
        self._y_fp5_batch_tensor = None
        self._y_fp6_batch_tensor = None
        self._y_fp7_batch_tensor = None
        self._y_protein_batch_tensor = None
        self._matching_batch_tensor = None


        self.next_batch = None
        self.data_init_op = None
        self.data_init_op = None
        self.data_init_op_eval = None

        self.variable_init_op = None
        self.train_op = None
        self.train_loss_history = None
        self.train_valid_history = None

        self.prediction = None
        self.loss = None
        self.accuracy = None
        self.global_step = None

        self.metrics_init_op = None
        self.metrics = None
        self.update_metrics_op = None

        self.summary_op = None
        self.history = {}

        # Build input/network

        self.input(
            input_finger1_shape=self.input_finger1_shape,
            input_finger2_shape=self.input_finger2_shape,
            input_finger3_shape=self.input_finger3_shape,
            input_finger4_shape=self.input_finger4_shape,
            input_finger5_shape=self.input_finger5_shape,
            input_finger6_shape=self.input_finger6_shape,
            input_finger7_shape=self.input_finger7_shape,
            input_protein_shape=self.input_protein_shape,
        )

        self.build_SiamNet()

    def _print_parameter(
        self,
        parameter_dict,
        ):

        parameter_str = "\n".join(
            f"| {prmt_key:18s}: {str(prmt_value):>8s} |"
            for prmt_key, prmt_value in parameter_dict.items()
        )
        print(
            "=" * 7 + " Given Parameters " +  "=" * 7,
            parameter_str,
            "=" * 32,
            sep='\n',
        )

    def _print_layer(
        self,
        name,
        input_shape=None,
        output_shape=None,
        ):

        string_format = " ".join(
            [
                "{name:15s}\t|",
                "{input_shape:20s}",
                "-> ",
                "{output_shape:20s}",
                "|",
            ]
        )
        print(
            string_format.format(
                name=name,
                input_shape=str(input_shape),
                output_shape=str(output_shape),
            )
        )




    def input(
        self,
        input_finger1_shape=(None, 1024),
        input_finger2_shape=(None, 1024),
        input_finger3_shape=(None, 1024),
        input_finger4_shape=(None, 1024),
        input_finger5_shape=(None, 1024),
        input_finger6_shape=(None, 1024),
        input_finger7_shape=(None, 166),
        input_protein_shape=(None, 20),
        ):

        buffer_size = self.BUFFER_SIZE
        with tf.name_scope('input'):


            # First Input
            X_fp1_t = tf.placeholder(
                tf.float32,
                input_finger1_shape,
                name='fingerprint1_tensor_interface',
            )
            X_fp2_t = tf.placeholder(
                tf.float32,
                input_finger2_shape,
                name='fingerprint2_tensor_interface',
            )
            X_fp3_t = tf.placeholder(
                tf.float32,
                input_finger3_shape,
                name='fingerprint3_tensor_interface',
            )
            X_fp4_t = tf.placeholder(
                tf.float32,
                input_finger4_shape,
                name='fingerprint4_tensor_interface',
            )
            X_fp5_t = tf.placeholder(
                tf.float32,
                input_finger5_shape,
                name='fingerprint5_tensor_interface',
            )
            X_fp6_t = tf.placeholder(
                tf.float32,
                input_finger6_shape,
                name='fingerprint6_tensor_interface',
            )
            X_fp7_t = tf.placeholder(
                tf.float32,
                input_finger7_shape,
                name='fingerprint7_tensor_interface',
            )
            X_protein_t = tf.placeholder(
                tf.float32,
                input_protein_shape,
                name='protein_tensor_interface',
            )
            X_label_t = tf.placeholder(
                tf.float32,
                shape=[None],
                name='label_tensor_interface',
            )

            dataset_x = tf.data.Dataset.from_tensor_slices((
                X_fp1_t, X_fp2_t, X_fp3_t, X_fp4_t, X_fp5_t, X_fp6_t, X_fp7_t,
                X_protein_t, X_label_t))
            dataset_x = dataset_x.shuffle(
                buffer_size=buffer_size,
            )
            dataset_x = dataset_x.prefetch(
                buffer_size=3*self.BATCH_SIZE,
            )
            dataset_y = dataset_x.shuffle(
                buffer_size=buffer_size,
            )

            dataset_x = dataset_x.batch(
                batch_size=self.BATCH_SIZE,
            )
            # dataset_y = tf.data.Dataset.from_tensor_slices((
            #     X_fp1_t, X_fp2_t, X_fp3_t, X_fp4_t, X_fp5_t, X_fp6_t, X_fp7_t,
            #     X_protein_t, X_label_t))
            dataset_y = dataset_y.batch(
                batch_size=self.BATCH_SIZE,
            )
            # dataset_y = tf.data.Dataset.from_tensor_slices((
            #     Y_fp1_t, Y_fp2_t, Y_fp3_t, Y_fp4_t, Y_fp5_t, Y_fp6_t, Y_fp7_t,
            #     Y_protein_t, Y_label_t))
            # dataset_y = dataset_y.shuffle(
            #     buffer_size=buffer_size,
            # )
            # dataset_y = dataset_y.batch(
            #     batch_size=self.BATCH_SIZE,
            # )

            dataset = tf.data.Dataset.zip((dataset_x, dataset_y))

            data_op = dataset.make_initializable_iterator()
            data_init_op = data_op.initializer
            next_batch = ((X_fp1_batch, X_fp2_batch,  X_fp3_batch,  X_fp4_batch,
                X_fp5_batch, X_fp6_batch,  X_fp7_batch, X_protein_batch, X_label_batch),
                (Y_fp1_batch, Y_fp2_batch, Y_fp3_batch,  Y_fp4_batch,
                Y_fp5_batch, Y_fp6_batch,  Y_fp7_batch, Y_protein_batch, Y_label_batch)) = data_op.get_next()


        print('[dtype] fingerprint1: %s , fingerprint2: %s, fingerprint3: %s, fingerprint4: %s, fingerprint5: %s, fingerprint6: %s , fingerprint7: %s, protein: %s' % (X_fp1_batch.dtype, X_fp2_batch.dtype, X_fp3_batch.dtype, X_fp4_batch.dtype, X_fp5_batch.dtype, X_fp6_batch.dtype, X_fp7_batch.dtype, X_protein_batch.dtype))
        print('[shape] fingerprint1: %s , fingerprint2: %s, fingerprint3: %s, fingerprint4: %s, fingerprint5: %s, fingerprint6: %s , fingerprint7: %s, protein: %s' % (Y_fp1_batch.get_shape(), Y_fp2_batch.get_shape(),X_fp3_batch.get_shape(), X_fp4_batch.get_shape(), X_fp5_batch.get_shape(), Y_fp6_batch.get_shape(), X_fp7_batch.get_shape(), X_protein_batch.get_shape()))


        self._x_fp1_tensor = X_fp1_t
        self._x_fp2_tensor = X_fp2_t
        self._x_fp3_tensor = X_fp3_t
        self._x_fp4_tensor = X_fp4_t
        self._x_fp5_tensor = X_fp5_t
        self._x_fp6_tensor = X_fp6_t
        self._x_fp7_tensor = X_fp7_t
        self._x_protein_tensor = X_protein_t
        self._x_label_tensor = X_label_t

        # self._y_fp1_tensor = X_fp1_t
        # self._y_fp2_tensor = X_fp2_t
        # self._y_fp3_tensor = X_fp3_t
        # self._y_fp4_tensor = X_fp4_t
        # self._y_fp5_tensor = X_fp5_t
        # self._y_fp6_tensor = X_fp6_t
        # self._y_fp7_tensor = X_fp7_t
        # self._y_protein_tensor = X_protein_t
        # self._y_label_tensor = X_label_t


        self._x_fp1_batch_tensor = X_fp1_batch
        self._x_fp2_batch_tensor = X_fp2_batch
        self._x_fp3_batch_tensor = X_fp3_batch
        self._x_fp4_batch_tensor = X_fp4_batch
        self._x_fp5_batch_tensor = X_fp5_batch
        self._x_fp6_batch_tensor = X_fp6_batch
        self._x_fp7_batch_tensor = X_fp7_batch
        self._x_protein_batch_tensor = X_protein_batch
        self._x_label_batch_tensor = X_label_batch

        self._y_fp1_batch_tensor = Y_fp1_batch
        self._y_fp2_batch_tensor = Y_fp2_batch
        self._y_fp3_batch_tensor = Y_fp3_batch
        self._y_fp4_batch_tensor = Y_fp4_batch
        self._y_fp5_batch_tensor = Y_fp5_batch
        self._y_fp6_batch_tensor = Y_fp6_batch
        self._y_fp7_batch_tensor = Y_fp7_batch
        self._y_protein_batch_tensor = Y_protein_batch
        self._y_label_batch_tensor = Y_label_batch


        # self._matching_batch_tensor = match_result

        self.next_batch = next_batch
        self.data_init_op = data_init_op

    def _player_conv2d(
        self,
        _input,
        out_channels,
        reuse=tf.AUTO_REUSE,
        name='p_conv2d',
    ):
        # print(name + ' ' + '-'*20)
        # print(_input.get_shape())
        with tf.variable_scope(name, reuse=reuse):

            prev_length = _input.get_shape()[-2]
            prev_channels = _input.get_shape()[-1]

            initializer = tf.random_normal_initializer(
                mean=.0,
                stddev=0.1,
            )
            filter_weight = tf.get_variable(
                'protein_filter_weight',
                shape=[
                    1,
                    3,
                    prev_channels,
                    out_channels,
                ],
                dtype=tf.float32,
                initializer=initializer,
            )

            conv2d = tf.nn.conv2d(
                _input,
                filter_weight,
                strides=[1, 1, 1, 1],
                padding="SAME",
                name=name,
            )

            self._print_layer(
                name=name,
                input_shape=_input.get_shape(),
                output_shape=conv2d.get_shape(),
            )

            batch_norm = tf.layers.batch_normalization(
                inputs=conv2d,
                name='batch_norm',
            )

            activation = tf.nn.relu(
                batch_norm,
                name='activation',
            )

            pooling = tf.nn.max_pool(
                activation,
                ksize=[1, 1, 2, 2],
                strides=[1, 1, 2, 2],
                padding='SAME',
            )

            self._print_layer(
                name='max_pooling',
                input_shape=activation.get_shape(),
                output_shape=pooling.get_shape(),
            )

        return pooling

    def _layer_conv2d(
        self,
        input_,
        present_weight,
        name="conv2d",
        reuse=tf.AUTO_REUSE,
        ):

        input_dim = len(input_.get_shape())
        input_channels = input_.get_shape()[-1]
        # stride_channels = tf.div(input_channels, 4)

        with tf.variable_scope(name, reuse=reuse):

            prev_weight = input_.get_shape()[-1]

            initializer = tf.random_normal_initializer(
                mean=.0,
                stddev=0.1,
            )
            filter_weight = tf.get_variable(
                'conv2d_filter_weight',
                shape=[
                    3,
                    1,
                    prev_weight,
                    present_weight,
                ],
                dtype=tf.float32,
                initializer=initializer,
            )
            biases = tf.get_variable(
                'conv2d_biases',
                shape=[present_weight],
                dtype=tf.float32,
                initializer=initializer,
            )
            conv2d = tf.add(
                tf.nn.conv2d(
                    input_,
                    filter_weight,
                    strides=[1, 1, 1, 1],
                    padding="SAME",
                ),
                biases,
                name=name,
            )
        self._print_layer(
            name=name,
            input_shape=input_.get_shape(),
            output_shape=conv2d.get_shape(),
        )

        return conv2d

    def protein_bert_network(
        self,
        proteins,
        name='protein_bert_network',
        reuse=tf.AUTO_REUSE,
        # dropout=0.7,
        ):
        print(name + ' ' + '-'*20)

        with tf.variable_scope(name, reuse=reuse):
            f, g, t = proteins.get_shape()
            proteins = tf.reshape(
                proteins,
                [-1, 1, g, t],
            )
            conv_layer1 = self._player_conv2d(
                _input=proteins,
                out_channels=256,
                name='conv2d_p1',
            )
            conv_layer2 = self._player_conv2d(
                _input=conv_layer1,
                out_channels=64,
                name='conv2d_p2',
            )
            a = conv_layer2.get_shape()[-1]
            b = conv_layer2.get_shape()[-2]

            reshape = tf.reshape(
                conv_layer2,
                [-1,a*b],
            )
            dense_layer = tf.layers.dense(
                tf.nn.elu(reshape),
                256,
                name="dense_layer",
                reuse=reuse,
            )
            self._print_layer(
                name='dense_layer',
                input_shape=conv_layer2.get_shape(),
                output_shape=dense_layer.get_shape(),
            )

            end_layer = tf.layers.dense(
                tf.nn.elu(dense_layer),
                64,
                name="end_layer",
                reuse=reuse,
            )
            self._print_layer(
                name="end_layer",
                input_shape=dense_layer.get_shape(),
                output_shape=end_layer.get_shape(),
            )

        return end_layer

    def separate_network(
        self,
        separate_input,
        name='separate_network',
        reuse=tf.AUTO_REUSE,
        # dropout=0.7,
        ):
        print(name + ' ' + '-'*20)

        with tf.variable_scope(name, reuse=reuse):
            input_channels = separate_input.get_shape()[-1]
            separate_input = tf.reshape(
                separate_input,
                [-1, input_channels, 1, 1],
            )

            # first layer
            conv2d_0 = self._layer_conv2d(
                input_= tf.nn.elu(separate_input),
                present_weight=4,
                name="conv2d_0",
                reuse=reuse,
            )
            # print(conv2d_0)
            conv2d_0 = tf.nn.elu(conv2d_0)
            # second layer
            conv2d_1 = self._layer_conv2d(
                input_=conv2d_0,
                present_weight=8,
                name="conv2d_1",
                reuse=reuse,
            )
            # print(conv2d_1)

            # third layer
            conv2d_2 = tf.nn.dropout(
                tf.nn.max_pool(
                    conv2d_1,
                    ksize=[1,2,1,1],
                    strides=[1,2,1,1],
                    padding="SAME",
                ),
                keep_prob=self.DROPOUT,
            )
            vector = tf.reshape(
                conv2d_2,
                [-1, int(ceil(int(input_channels)/2)*conv2d_1.get_shape()[-1])],
                name='output_vector',
            )
            print(vector.get_shape())
            return vector

    def _layer_linear(
        self,
        input_x,
        output_size,
        is_training=True,
        stddev=0.02,
        name=None,
        ):
        with tf.variable_scope(name or 'linear'):

            input_shape = input_x.get_shape().as_list()

            weight = tf.get_variable(
                "weight",
                shape=[input_shape[1], output_size],
                dtype=tf.float32,
                initializer=tf.random_normal_initializer(
                    stddev=stddev,
                ),
            )
            bias = tf.get_variable(
                "bias",
                shape=[output_size],
                initializer=tf.constant_initializer(0.0),
            )

            lineared = tf.matmul(input_x, weight) + bias
            print(name, lineared.get_shape())

        return lineared


    def total_network(
        self,
        input_total,
        input_protein,
        name='end_network',
        reuse=tf.AUTO_REUSE,
        ):
        print(name + ' ' + '-'*20)
        with tf.variable_scope(name, reuse=reuse):

            # first_layer = self._layer_linear

            first_layer = tf.layers.dense(
                tf.nn.elu(input_total),
                2048,
                name="first_layer",
                reuse=reuse,
            )
            self._print_layer(
                name="first_layer",
                input_shape=input_total.get_shape(),
                output_shape=first_layer.get_shape(),
            )
            second_layer = tf.layers.dense(
                tf.nn.elu(input_total),
                512,
                name="second_layer",
                reuse=reuse,
            )
            self._print_layer(
                name="second_layer",
                input_shape=first_layer.get_shape(),
                output_shape=second_layer.get_shape(),
            )
            third_layer = tf.layers.dense(
                tf.nn.elu(first_layer),
                128,
                name="third_layer",
                reuse=reuse,
            )
            self._print_layer(
                name="third_layer",
                input_shape=second_layer.get_shape(),
                output_shape=third_layer.get_shape(),
            )
            end_layer = tf.concat(
                (third_layer, input_protein),
                axis=1,
                name='end_layer',
            )
            self._print_layer(
                name='end_layer',
                input_shape=third_layer.get_shape(),
                output_shape=end_layer.get_shape(),
            )

        return end_layer

    def protein_network(
        self,
        proteins,
        name='protein_network',
        reuse=tf.AUTO_REUSE,
        ):

        print(name + ' ' + '-'*20)

        with tf.variable_scope(name, reuse=reuse):

            first_layer = tf.layers.dense(
                tf.nn.elu(proteins),
                2048,
                name="first_layer",
                reuse=reuse,
            )
            self._print_layer(
                name=name,
                input_shape=proteins.get_shape(),
                output_shape=first_layer.get_shape(),
            )
            second_layer = tf.layers.dense(
                tf.nn.elu(first_layer),
                512,
                name="second_layer",
                reuse=reuse,
            )
            self._print_layer(
                name=name,
                input_shape=first_layer.get_shape(),
                output_shape=second_layer.get_shape(),
            )
            third_layer = tf.layers.dense(
                tf.nn.elu(second_layer),
                128,
                name="third_layer",
                reuse=reuse,
            )
            self._print_layer(
                name=name,
                input_shape=second_layer.get_shape(),
                output_shape=third_layer.get_shape(),
            )
            end_layer = tf.layers.dense(
                tf.nn.elu(third_layer),
                64,
                name="end_layer",
                reuse=reuse,
            )
            self._print_layer(
                name=name,
                input_shape=third_layer.get_shape(),
                output_shape=end_layer.get_shape(),
            )

        return end_layer

    def build_SiamNet(
        self,
        reuse=tf.AUTO_REUSE,
        ):

        print('\n[SiameseNet_model]: ' + '='*30)

        with tf.variable_scope('SiameseNet_model', reuse=reuse):
            # First Input
            Input_X_fp1 = tf.placeholder(
                tf.float32,
                self.input_finger1_shape,
                name='Input_X_fingerprint1',
            )
            Input_X_fp2 = tf.placeholder(
                tf.float32,
                self.input_finger2_shape,
                name='Input_X_fingerprint2',
            )
            Input_X_fp3 = tf.placeholder(
                tf.float32,
                self.input_finger3_shape,
                name='Input_X_fingerprint3',
            )
            Input_X_fp4 = tf.placeholder(
                tf.float32,
                self.input_finger4_shape,
                name='Input_X_fingerprint4',
            )
            Input_X_fp5 = tf.placeholder(
                tf.float32,
                self.input_finger5_shape,
                name='Input_X_fingerprint5',
            )
            Input_X_fp6 = tf.placeholder(
                tf.float32,
                self.input_finger6_shape,
                name='Input_X_fingerprint6',
            )
            Input_X_fp7 = tf.placeholder(
                tf.float32,
                self.input_finger7_shape,
                name='Input_X_fingerprint7',
            )
            Input_X_protein = tf.placeholder(
                tf.float32,
                self.input_protein_shape,
                name='Input_X_protein',
            )
            Input_X_label = tf.placeholder(
                tf.float32,
                shape=[None],
                name='Input_X_label',
            )
            # Second Input
            Input_Y_fp1 = tf.placeholder(
                tf.float32,
                self.input_finger1_shape,
                name='Input_Y_fingerprint1',
            )
            Input_Y_fp2 = tf.placeholder(
                tf.float32,
                self.input_finger2_shape,
                name='Input_Y_fingerprint2',
            )
            Input_Y_fp3 = tf.placeholder(
                tf.float32,
                self.input_finger3_shape,
                name='Input_Y_fingerprint3',
            )
            Input_Y_fp4 = tf.placeholder(
                tf.float32,
                self.input_finger4_shape,
                name='Input_Y_fingerprint4',
            )
            Input_Y_fp5 = tf.placeholder(
                tf.float32,
                self.input_finger5_shape,
                name='Input_Y_fingerprint5',
            )
            Input_Y_fp6 = tf.placeholder(
                tf.float32,
                self.input_finger6_shape,
                name='Input_Y_fingerprint6',
            )
            Input_Y_fp7 = tf.placeholder(
                tf.float32,
                self.input_finger7_shape,
                name='Input_Y_fingerprint7',
            )
            Input_Y_protein = tf.placeholder(
                tf.float32,
                self.input_protein_shape,
                name='Input_Y_protein',
            )
            Input_Y_label = tf.placeholder(
                tf.float32,
                shape=[None],
                name='Input_Y_label',
            )
            match_result = tf.cast(
                x=tf.equal(
                    Input_X_label,
                    Input_Y_label,
                ),
                dtype=tf.float32,
                name='x_y_label_match',
            )

            # Fingerprint1_network
            X_fp1_network = self.separate_network(
                separate_input=Input_X_fp1,
                name='fp1_network',
            )
            Y_fp1_network = self.separate_network(
                separate_input=Input_Y_fp1,
                name='fp1_network',
                reuse=True,
            )

            # Fingerprint2_network
            X_fp2_network = self.separate_network(
                separate_input=Input_X_fp2,
                name='fp2_network',
            )
            Y_fp2_network = self.separate_network(
                separate_input=Input_Y_fp2,
                name='fp2_network',
                reuse=True,
            )

            # Fingerprint3_network
            X_fp3_network = self.separate_network(
                separate_input=Input_X_fp3,
                name='fp3_network',
            )
            Y_fp3_network = self.separate_network(
                separate_input=Input_Y_fp3,
                name='fp3_network',
                reuse=True,
            )
            # Fingerprint4_network
            X_fp4_network = self.separate_network(
                separate_input=Input_X_fp4,
                name='fp4_network',
            )
            Y_fp4_network = self.separate_network(
                separate_input=Input_Y_fp4,
                name='fp4_network',
                reuse=True,
            )
            # Fingerprint5_network
            X_fp5_network = self.separate_network(
                separate_input=Input_X_fp5,
                name='fp5_network',
            )
            Y_fp5_network = self.separate_network(
                separate_input=Input_Y_fp5,
                name='fp5_network',
                reuse=True,
            )
            # Fingerprint6_network
            X_fp6_network = self.separate_network(
                separate_input=Input_X_fp6,
                name='fp6_network',
            )
            Y_fp6_network = self.separate_network(
                separate_input=Input_Y_fp6,
                name='fp6_network',
                reuse=True,
            )
            # Fingerprint7_network
            X_fp7_network = self.separate_network(
                separate_input=Input_X_fp7,
                name='fp7_network',
            )
            Y_fp7_network = self.separate_network(
                separate_input=Input_Y_fp7,
                name='fp7_network',
                reuse=True,
            )
            # protein_network
            # X_protein_network = self.protein_network(
            #     proteins=Input_X_protein,
            #     name='protein_network',
            #     reuse=False,
            # )
            # Y_protein_network = self.protein_network(
            #     proteins=Input_Y_protein,
            #     name='protein_network',
            #     reuse=True,
            # )

            X_protein_network = self.protein_bert_network(
                proteins=Input_X_protein,
                name='protein_network',
                reuse=False,
            )
            Y_protein_network = self.protein_bert_network(
                proteins=Input_Y_protein,
                name='protein_network',
                reuse=True,
            )
            X_total_vector = tf.concat(
                (X_fp1_network, X_fp2_network, X_fp3_network, X_fp4_network,
                X_fp5_network, X_fp6_network, X_fp7_network),
                axis=1,
                name='X_total_network')

            Y_total_vector = tf.concat(
                (Y_fp1_network, Y_fp2_network, Y_fp3_network, Y_fp4_network,
                Y_fp5_network, Y_fp6_network, Y_fp7_network),
                axis=1,
                name='Y_total_network')


            X_network = self.total_network(
                input_total=X_total_vector,
                input_protein=X_protein_network,
                name='total_network',
                reuse=False,
            )
            Y_network = self.total_network(
                input_total=Y_total_vector,
                input_protein=Y_protein_network,
                name='total_network',
                reuse=True,
            )

            summary_X = tf.summary.histogram('X', Input_X_fp1)
            summary_Y = tf.summary.histogram('Y', Input_Y_fp1)
            summary_X_net = tf.summary.histogram('X_network', X_network)
            summary_Y_net = tf.summary.histogram('Y_network', Y_network)

            with tf.name_scope('spring_loss_scope'):

                margin = 50.0
                true_labels = match_result
                false_labels = tf.subtract(
                    1.0,
                    match_result,
                    name="false_labels",
                )

                eucl_distance_0 = tf.reduce_sum(
                    tf.pow(
                        tf.subtract(
                            X_network,
                            Y_network,
                        ),
                        2,
                    ),
                    1,
                )
                eucl_distance = tf.sqrt(
                    eucl_distance_0+1e-6,
                    name='eucl_distance',
                )
                Constant = tf.constant(
                    margin,
                    name="Constant",
                )
                positive = tf.multiply(
                    true_labels,
                    eucl_distance_0,
                    name="x_y_eucl_distance_0",
                )
                negative = tf.multiply(
                    false_labels,
                    tf.pow(
                        tf.maximum(
                            0.0,
                            tf.subtract(
                                Constant,
                                eucl_distance,
                            ),
                        ),
                        2,
                    ),
                    name="Ny_c_eucl_distance",
                )

                losses = tf.add(
                    positive,
                    negative,
                    name="losses",
                )
                print('losses', losses)
                spring_loss = tf.reduce_mean(
                    losses,
                    name="spring_loss",
                )
                print('total_losses', losses)
            summary_loss_spring = tf.summary.scalar('spring_loss', spring_loss)

            # Summaries for training
            summary_op_input = tf.summary.merge([
                summary_X,
                summary_Y,
                summary_X_net,
                summary_Y_net,
            ])
            summary_op_loss = tf.summary.merge([
                summary_loss_spring,
            ])
        # Optimization =======================================================

        with tf.variable_scope('optimization'):

            optimizer = tf.train.AdamOptimizer(
                learning_rate=.01,
                name='optimizer_Adam',
            )
            train_op = optimizer.minimize(
                spring_loss,
                var_list=tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES,
                    scope='SiameseNet_model',
                )
            )
        variable_network = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope='SiameseNet_model',
        )
        var_init_op = tf.variables_initializer(
            var_list = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope='SiameseNet_model',
            )
        )

        opt_init_op = tf.variables_initializer(
            var_list = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope='optimization',
            )
        )
        variable_init_op = tf.group(
            *[var_init_op, opt_init_op]
        )

        # ====================================================================

        with tf.variable_scope("network_metrics", reuse=reuse):
            metrics_train = {
                "Train_loss_spring": tf.metrics.mean(spring_loss),
            }

        # Group the update ops for the tf.metrics
        update_metrics_op_train = tf.group(
            *[op for _, op in metrics_train.values()]
        )

        # Get the op to reset the local variables used in tf.metrics
        metrics_init_op = tf.variables_initializer(
            var_list=tf.get_collection(
                tf.GraphKeys.LOCAL_VARIABLES,
                scope="network_metrics",
            ),
            name='metrics_init_op',
        )

        # Return
        self.variable = variable_network
        self.var_init_op = var_init_op
        self.opt_init_op = opt_init_op
        self.variable_init_op = variable_init_op

        self.train_op = train_op

        self.metrics_train = metrics_train
        self.update_metrics_op_train = update_metrics_op_train
        self.metrics_init_op = metrics_init_op
        self.summary_op_input = summary_op_input
        self.summary_op_loss = summary_op_loss

        self.Input_X_fp1 = Input_X_fp1
        self.Input_X_fp2 = Input_X_fp2
        self.Input_X_fp3 = Input_X_fp3
        self.Input_X_fp4 = Input_X_fp4
        self.Input_X_fp5 = Input_X_fp5
        self.Input_X_fp6 = Input_X_fp6
        self.Input_X_fp7 = Input_X_fp7

        self.Input_Y_fp1 = Input_Y_fp1
        self.Input_Y_fp2 = Input_Y_fp2
        self.Input_Y_fp3 = Input_Y_fp3
        self.Input_Y_fp4 = Input_Y_fp4
        self.Input_Y_fp5 = Input_Y_fp5
        self.Input_Y_fp6 = Input_Y_fp6
        self.Input_Y_fp7 = Input_Y_fp7

        self.Input_X_label = Input_X_label
        self.Input_Y_label = Input_Y_label

        self.Input_X_protein = Input_X_protein
        self.Input_Y_protein = Input_Y_protein
        self.Input_match_result = match_result

        self.X_network = X_network
        self.Y_network = Y_network

        self.loss_spring = spring_loss
        self.eucl_distance = eucl_distance
        # self.loss_step = step_loss
        print('='*50)


    def train_SiamNet(
        self,
        input_finger1=None,
        input_finger2=None,
        input_finger3=None,
        input_finger4=None,
        input_finger5=None,
        input_finger6=None,
        input_finger7=None,
        input_protein=None,
        input_label=None,
        batch_size=128,
        epoch_num=25,
        dropout=0.7,
        model_save_dir='./model_save',
        pre_trained_path=None,
        reuse=tf.AUTO_REUSE,
        ):

        metrics_train = self.metrics_train
        self.BATCH_SIZE = batch_size
        self.DROPOUT = dropout

        if batch_size is None:
            batch_size_int = self.BATCH_SIZE
        else:
            batch_size_int = int(batch_size)


        parameter_dict = {
            'BATCH_SIZE': batch_size_int,
            'EPOCH_NUM': epoch_num,
            'DROPOUT': dropout,
        }
        self._print_parameter(parameter_dict)

        self.input(
            input_finger1_shape=self.input_finger1_shape,
            input_finger2_shape=self.input_finger2_shape,
            input_finger3_shape=self.input_finger3_shape,
            input_finger4_shape=self.input_finger4_shape,
            input_finger5_shape=self.input_finger5_shape,
            input_finger6_shape=self.input_finger6_shape,
            input_finger7_shape=self.input_finger7_shape,
            input_protein_shape=self.input_protein_shape,
        )

        if pre_trained_path is None:
            if os.path.isdir(model_save_dir):
                shutil.rmtree(model_save_dir)
                os.makedirs(model_save_dir, exist_ok=True)

        # Initialize tf.Saver instances to save weights during training
        last_saver = tf.train.Saver(
            var_list=self.variable,
            max_to_keep=2,  # will keep last 5 epochs as default
        )
        begin_at_epoch = 0

        with tf.Session() as sess:

            # For TensorBoard (takes care of writing summaries to files)
            train_writer = tf.summary.FileWriter(
                logdir=os.path.join(
                    model_save_dir,
                    'train_summaries',
                ),
                graph=sess.graph,
            )
            optimizer = tf.train.AdamOptimizer(
                learning_rate=.0001,
                name='optimizer_Adam',
            )
            train_op = optimizer.minimize(
                self.loss_spring,
            )
            tf.initialize_all_variables().run()
            # Reload weights from directory if specified
            if pre_trained_path is not None:
                #logging.info("Restoring parameters from {}".format(restore_from))
                if os.path.isdir(pre_trained_path):
                    last_save_path = os.path.join(
                        pre_trained_path,
                        'last_weights',
                    )
                    saved_model = tf.train.latest_checkpoint(last_save_path)
                    begin_at_epoch = int(saved_model.split('-')[-1])
                    epoch_num = begin_at_epoch + epoch_num

                last_saver.restore(sess, saved_model)
                print("Pre-trained model loaded")

            # Initialize model variables
            sess.run(self.variable_init_op)


            for epoch in range(begin_at_epoch, epoch_num):
                # Load the training dataset into the pipeline
                # and initialize the metrics local variables
                sess.run(
                    self.data_init_op,
                    feed_dict={
                        self._x_fp1_tensor: input_finger1,
                        self._x_fp2_tensor: input_finger2,
                        self._x_fp3_tensor: input_finger3,
                        self._x_fp4_tensor: input_finger4,
                        self._x_fp5_tensor: input_finger5,
                        self._x_fp6_tensor: input_finger6,
                        self._x_fp7_tensor: input_finger7,
                        self._x_protein_tensor: input_protein,
                        self._x_label_tensor: input_label,
                    }
                )
                sess.run(self.metrics_init_op)
                epoch_msg = "Epoch %d/%d\n" % (epoch + 1, epoch_num)
                sys.stdout.write(epoch_msg)

                # BATCH : Optimized by each chunk
                batch_num = 0
                batch_len = int(np.ceil(len(input_finger1) / batch_size_int))

                train_len = batch_len

                batch_remains_ok = True
                while batch_remains_ok and (batch_num <= batch_len):
                    try:
                        for batch in range(train_len):
                            ((X_fp1_batch, X_fp2_batch,  X_fp3_batch,  X_fp4_batch,
                                X_fp5_batch, X_fp6_batch,  X_fp7_batch, X_protein_batch, X_label_batch),
                                (Y_fp1_batch, Y_fp2_batch, Y_fp3_batch,  Y_fp4_batch,
                                Y_fp5_batch, Y_fp6_batch,  Y_fp7_batch, Y_protein_batch, Y_label_batch)) = sess.run(self.next_batch)

                            (summary_train_op,
                             summary_input,
                             summary_loss,
                             err_rate,) = sess.run(
                                [
                                    train_op,
                                    self.summary_op_input,
                                    self.summary_op_loss,
                                    self.loss_spring,
                                ],
                                feed_dict={
                                    self.Input_X_fp1: X_fp1_batch,
                                    self.Input_X_fp2: X_fp2_batch,
                                    self.Input_X_fp3: X_fp3_batch,
                                    self.Input_X_fp4: X_fp4_batch,
                                    self.Input_X_fp5: X_fp5_batch,
                                    self.Input_X_fp6: X_fp6_batch,
                                    self.Input_X_fp7: X_fp7_batch,

                                    self.Input_Y_fp1: Y_fp1_batch,
                                    self.Input_Y_fp2: Y_fp2_batch,
                                    self.Input_Y_fp3: Y_fp3_batch,
                                    self.Input_Y_fp4: Y_fp4_batch,
                                    self.Input_Y_fp5: Y_fp5_batch,
                                    self.Input_Y_fp6: Y_fp6_batch,
                                    self.Input_Y_fp7: Y_fp7_batch,

                                    self.Input_X_protein: X_protein_batch,
                                    self.Input_Y_protein: Y_protein_batch,
                                    self.Input_X_label: X_label_batch,
                                    self.Input_Y_label: Y_label_batch,
                                },
                            )
                            # err_rate = self.loss_spring.eval(
                            #     {
                            #         self.Input_Y: ,
                            #     }
                            # )
                            #
                            # sess.run(
                            #     [
                            #         self.update_metrics_op_train,
                            #     ],
                            #     feed_dict={
                            #         self.Input_X: X_fp1_batch,
                            #         self.Input_Y: Y_fp1_batch,
                            #
                            #     },
                            # )
                            # -----------------------------------------------

                            # Write summaries for tensorboard
                            batch_num += 1
                            batch_pct = int(20 * batch_num / train_len)
                            batch_bar = "[%s] " % (("#" * batch_pct) + ("-" * (20 - batch_pct)))
                            batch_msg = "\rBatch [%s/%s] " % (batch_num, train_len)
                            batch_err = "err_rate: %.5f" % (err_rate)
                            # batch_dist = f'distance: {np.round(distance, 2)}'
                            batch_msg = batch_msg + batch_bar + batch_err# +batch_dist

                            sys.stdout.flush()
                            sys.stdout.write(batch_msg)

                            # -----------------------------------------------

                    except tf.errors.OutOfRangeError:
                        batch_remains_ok = False
                        result_msg = "\n"
                        # result_msg = "\n finished.\n"
                        sys.stdout.write(result_msg)
                        continue

                train_writer.add_summary(
                    summary_input,
                    epoch,
                )


                # Metrics
                # metrics_values_train = {k: value[0]
                #     for k, value in metrics_train.items()}
                #
                # metrics_res_train = sess.run([
                #     metrics_values_train,
                # ])
                # metrics_res = {
                #     **metrics_res_train,
                # }
                # metrics_str = "\n".join(
                #     "{metric_key}: {metric_value:.8f}".format(
                #             metric_key=k,
                #             metric_value=value,
                #     ) for k, value in metrics_res.items()
                # )
                # print(
                #     "-- Metrics -- ",
                #     metrics_str,
                #     sep='\n',
                # )

                # Save weights
                if (epoch%10 == 0):

                    last_save_path = os.path.join(
                        model_save_dir,
                        'last_weights',
                        'after-epoch',
                    )
                    last_saver.save(
                        sess,
                        last_save_path,
                        global_step=epoch + 1,
                    )
                    print('Model Saved: %s' % last_save_path)

            self.EPOCH_NUM = epoch_num
            print("Training Finished!")

    def evaluate_SiamNet(
        self,
        input_finger1=None,
        input_finger2=None,
        input_finger3=None,
        input_finger4=None,
        input_finger5=None,
        input_finger6=None,
        input_finger7=None,
        input_protein=None,
        input_label=None,
        dropout=1.0,
        pre_trained_path='./model_save',
        target_epoch=None,
        ):
        self.DROPOUT = dropout
        print(dropout, self.DROPOUT)
        assert pre_trained_path is not None, "`pre_trained_path` is mandatory."
        #global_step = tf.train.get_global_step()

        if self.X_network is None:
            tf.reset_default_graph()
            self.build_SiamNet()
            print('build_SiamNetwork again,')
        with tf.device('/cpu:0'):
            c = tf.ConfigProto(log_device_placement=True)


            with tf.Session(config=c) as sess:
                ## Initialize model variables
                sess.run(self.variable_init_op)

                # Reload weights from directory if specified
                if pre_trained_path is not None:
                    #logging.info("Restoring parameters from {}".format(restore_from))
                    if os.path.isdir(pre_trained_path):
                        last_save_path = os.path.join(pre_trained_path, 'last_weights')

                        if target_epoch:
                            saved_model = ''.join(
                                [
                                    last_save_path + '/',
                                    'after-epoch-',
                                    str(target_epoch),
                                ],
                            )
                            last_saver = tf.train.import_meta_graph(
                                saved_model + '.meta'
                            )
                        else:
                            last_saver = tf.train.Saver(
                                self.variable,
                            )
                            saved_model = tf.train.latest_checkpoint(
                                last_save_path,
                            )
                        #begin_at_epoch = int(saved_model.split('-')[-1])
                        last_saver.restore(sess, saved_model)

                pred = self.X_network.eval(
                    feed_dict={
                        self.Input_X_fp1: input_finger1,
                        self.Input_X_fp2: input_finger2,
                        self.Input_X_fp3: input_finger3,
                        self.Input_X_fp4: input_finger4,
                        self.Input_X_fp5: input_finger5,
                        self.Input_X_fp6: input_finger6,
                        self.Input_X_fp7: input_finger7,

                        self.Input_X_protein: input_protein,
                        self.Input_X_label: input_label,
                    },
                )

        return pred
