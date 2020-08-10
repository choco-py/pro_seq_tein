import shutil
import os
import tensorflow as tf
import numpy as np
import sys

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
        label_shape=(None, 488),
        batch_size=128,
        buffer_size=1000,
        dropout=1.0,
    ):

        super().__init__()


        self.input_shape = input_shape
        self.label_shape = label_shape

        self.EPOCH_NUM = 0
        self.BUFFER_SIZE = buffer_size
        self.BATCH_SIZE = batch_size
        self.DROPOUT = dropout

        self._x_tensor = None
        self._x_batch_tensor = None
        self._x_label_tensor = None
        self._x_batch_label_tensor = None

        self.next_batch = None
        self.data_init_op = None
        self.data_init_op_eval = None
        self.reduction = None

        self.variable_init_op = None
        self.train_op = None
        self.loss = None
        self.accuracy = None
        self.global_step = None

        # Build Input and network

        self.input(
            input_shape=self.input_shape,
            label_shape=self.label_shape,
        )

        self.build()

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
        input_shape=(None, 410, 528),
        label_shape=(None, 488),
    ):

        X_t = tf.placeholder(
            tf.float32,
            self.input_shape,
            name='protein_tensor_interface',
        )

        X_label_t = tf.placeholder(
            tf.float32,
            self.label_shape,
            name='label_tensor_interface',
        )

        dataset = tf.data.Dataset.from_tensor_slices((
            X_t,
            X_label_t,
        ))

        dataset = dataset.shuffle(
            buffer_size= 3*self.BATCH_SIZE,
        )
        dataset = dataset.prefetch(
            buffer_size=3*self.BATCH_SIZE,
        )
        dataset = dataset.batch(
            batch_size=self.BATCH_SIZE,
        )
        dataset = dataset.flat_map(
            lambda data_x, data_z: tf.data.Dataset.zip(
                (
                    tf.data.Dataset.from_tensors(data_x),
                    tf.data.Dataset.from_tensors(data_z),
                )
            )
        )
        data_op = dataset.make_initializable_iterator()

        data_init_op = data_op.initializer

        next_batch = X_t_batch, X_label_batch = data_op.get_next()

        print(f'[shape] protein: {X_t_batch.get_shape()}')

        self._x_tensor = X_t
        self._x_batch_tensor = X_t_batch
        self._x_label_tensor = X_label_t
        self._x_batch_label_tensor = X_label_batch
        self.next_batch = next_batch
        self.data_init_op = data_init_op

    def _layer_conv2d(
        self,
        _input,
        out_channels,
        dropout=1.0,
        reuse=tf.AUTO_REUSE,
        name='conv2d',
    ):
        # print(name + ' ' + '-'*20)
        # print(_input.get_shape())
        with tf.variable_scope(name, reuse=reuse) as scope:

            prev_length = _input.get_shape()[-2]
            prev_channels = _input.get_shape()[-1]

            initializer = tf.random_normal_initializer(
                mean=.0,
                stddev=0.1,
            )
            filter_weight = tf.get_variable(
                'filter_weight',
                shape=[
                    3,
                    3,
                    prev_channels,
                    out_channels,
                ],
                dtype=tf.float32,
                initializer=initializer,
            )

            conv2d = tf.nn.conv2d(
                _input,
                filter=filter_weight,
                strides=[1, 1, 1, 1],
                padding="SAME",
                data_format='NHWC',
                name='conv2d',
            )
            bias = tf.get_variable(
                'biases',
                shape=[out_channels],
                initializer=initializer,
            )
            conv2d = tf.nn.bias_add(conv2d, bias)

            self._print_layer(
                name=name,
                input_shape=_input.get_shape(),
                output_shape=conv2d.get_shape(),
            )
            #
            # batch_norm = tf.layers.batch_normalization(
            #     inputs=conv2d,
            #     name='batch_norm',
            # )

            activation = tf.nn.relu(
                conv2d,
                name='activation',
            )

            pooling = tf.nn.max_pool(
                activation,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
                name='pooling',
            )
            self._print_layer(
                name='max_pooling',
                input_shape=activation.get_shape(),
                output_shape=pooling.get_shape(),
            )

            output = tf.nn.dropout(pooling, dropout)

        layer_name = conv2d.name
        layer_shape = conv2d.get_shape()
        layer_info = f'[shape] {layer_name}: {layer_shape}'

        print(layer_info)

        return output

    def protein_network(
        self,
        proteins,
        name='protein_network',
        reuse=tf.AUTO_REUSE,
        dropout=1.0,
        ):
        print(name + ' ' + '-'*20)

        with tf.variable_scope(name, reuse=reuse):
            proteins = tf.reshape(
                proteins,
                [-1, 403, 768, 1],
            )
            conv_layer1 = self._layer_conv2d(
                _input=proteins,
                out_channels=16,
                dropout=self.DROPOUT,
                name='conv2d_1',
                reuse=reuse,
            )
            conv_layer2 = self._layer_conv2d(
                _input=conv_layer1,
                out_channels=32,
                dropout=self.DROPOUT,
                name='conv2d_2',
                reuse=reuse,

            )
            conv_layer3 = self._layer_conv2d(
                _input=conv_layer2,
                out_channels=32,
                dropout=self.DROPOUT,
                name='conv2d_3',
                reuse=reuse,
            )
            conv_layer4 = self._layer_conv2d(
                _input=conv_layer3,
                out_channels=16,
                dropout=self.DROPOUT,
                name='conv2d_4',
                reuse=reuse,
            )
            reshape = tf.layers.flatten(conv_layer4)

            dense_layer1 = tf.layers.dense(
                tf.nn.elu(reshape),
                2048,
                name="dense_layer1",
                reuse=reuse,
            )
            self._print_layer(
                name='dense_layer1',
                input_shape=conv_layer4.get_shape(),
                output_shape=dense_layer1.get_shape(),
            )
            dense_layer2 = tf.layers.dense(
                tf.nn.elu(dense_layer1),
                488,
                name="dense_layer2",
                reuse=reuse,
            )
            self._print_layer(
                name='dense_layer2',
                input_shape=dense_layer1.get_shape(),
                output_shape=dense_layer2.get_shape(),
            )

            self.reduction = conv_layer4
            # activation = tf.nn.softmax(
            #     logis=dense_layer2,
            # )
            # self._print_layer(
            #     name='activation',
            #     input_shape=dense_layer2.get_shape(),
            #     output_shape=activation.get_shape(),
            # )
        print(conv_layer3.name)

        print(dense_layer1.name)
        return dense_layer2


    def build(
        self,
        reuse=tf.AUTO_REUSE,
    ):
        print('\n[Protein_model]: '+ '='*30)

        with tf.variable_scope('Protein_model', reuse=tf.AUTO_REUSE):

            Input_X = tf.placeholder(
                tf.float32,
                self.input_shape,
                name='Input_X',
            )
            Input_X_label = tf.placeholder(
                tf.float32,
                shape=self.label_shape,
                name='Input_X_label',
            )

            output = self.protein_network(
                proteins=Input_X,
                name='protein_network',
                dropout=0.7,
                )
            print(output.name)

            with tf.name_scope('loss_scope'):

                loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=output,
                        labels=Input_X_label,
                    ),
                    name='loss',
                )
            summary_loss = tf.summary.scalar('loss', loss)
            print(tf.GraphKeys.TRAINABLE_VARIABLES,)
            print(tf.GraphKeys.GLOBAL_VARIABLES,)

        with tf.variable_scope('optimization'):
            optimizer = tf.train.AdamOptimizer(
                learning_rate=.01,
                name='optimizer_Adam',
            )
            train_op = optimizer.minimize(
                loss,
                var_list=tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES,
                    scope="Protein_model/protein_network",
                )
            )
        variable_network = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope='Protein_model',
        )
        var_init_op = tf.variables_initializer(
            var_list = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope='Protein_model',
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

        with tf.variable_scope("network_metrics", reuse=reuse):
            metrics_train = {
                "Train_loss": tf.metrics.mean(loss),
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
        # with tf.name_scope('prediction'):
        #
        #     predict_op = tf.argmax(
        #         output,
        #     )
        #

        # Return
        self.variable = variable_network
        self.var_init_op = var_init_op
        self.opt_init_op = opt_init_op
        self.variable_init_op = variable_init_op

        self.train_op = train_op

        self.metrics_train = metrics_train
        self.update_metrics_op_train = update_metrics_op_train
        self.metrics_init_op = metrics_init_op

        self.Input_X = Input_X
        self.Input_X_label = Input_X_label
        self.loss = loss

        self.output = output


    def train(
        self,
        input_=None,
        input_label=None,
        batch_size=128,
        epoch_num=5,
        dropout=0.7,
        model_save_dir='./model_save',
        pre_trained_path=None,
        reuse=tf.AUTO_REUSE,
    ):

        self.BATCH_SIZE = batch_size
        parameter_dict = {
            'BATCH_SIZE': batch_size,
            'EPOCH_NUM': epoch_num,
            'DROPOUT': dropout,
        }
        self._print_parameter(parameter_dict)

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
                self.loss,
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
                sess.run(
                    self.data_init_op,
                    feed_dict={
                        self._x_tensor: input_,
                        self._x_label_tensor: input_label,
                    }
                )

                sess.run(self.metrics_init_op)
                epoch_msg = "Epoch %d/%d\n" % (epoch + 1, epoch_num)
                sys.stdout.write(epoch_msg)

                # BATCH : Optimized by each chunk
                batch_num = 0
                batch_len = int(np.ceil(len(input_) / batch_size))

                train_len = batch_len

                batch_remains_ok = True
                while batch_remains_ok and (batch_num <= batch_len):
                    try:
                        for batch in range(train_len):
                            (X_t_batch, X_label_batch) = sess.run(self.next_batch)

                            (summary_train_op,
                             summary_loss,
                             ) = sess.run(
                                [
                                    train_op,
                                     self.loss,
                                ],
                                feed_dict={
                                    self.Input_X: X_t_batch,
                                    self.Input_X_label: X_label_batch,
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
                            batch_err = "err_rate: %.5f" % (summary_loss)
                            # batch_dist = f'distance: {np.round(distance, 2)}'
                            batch_msg = batch_msg + batch_bar + batch_err# +batch_dist

                            sys.stdout.flush()
                            sys.stdout.write(batch_msg)

                    except tf.errors.OutOfRangeError:
                        batch_remains_ok = False
                        result_msg = "\n"
                        # result_msg = "\n finished.\n"
                        sys.stdout.write(result_msg)
                        continue



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

    def evaluate(
        self,
        input_protein=None,
        dropout=1.0,
        pre_trained_path='./model_save',
        target_epoch=None,
        ):
        self.DROPOUT = dropout
        print(dropout, self.DROPOUT)
        assert pre_trained_path is not None, "`pre_trained_path` is mandatory."
        #global_step = tf.train.get_global_step()

        if self.protein_network is None:
            tf.reset_default_graph()
            self.build()
            print('build again,')
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

                pred = self.output.eval(
                    feed_dict={
                        self.Input_X: input_protein,
                        # self.Input_X_label: input_label,
                    },
                )

        return pred
