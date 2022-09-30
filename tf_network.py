import tensorflow as tf



class LeNet:

    def __init__(self):
        """
        Define some basic parameters here
        """
        self.network = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid',
                                   padding='same'),
            tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
            tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                                   activation='sigmoid'),
            tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(120, activation='sigmoid'),
            tf.keras.layers.Dense(84, activation='sigmoid'),
            tf.keras.layers.Dense(10)])

        pass

    def net(self, feats):
        """
        Define network.
        You can use init_weight() and init_bias() function to init weight matrix,
        for example:
            conv1_W = self.init_weight((3, 3, 1, 6))
            conv1_b = self.init_bias(6)
        :param feats: input features
        :return: logits
        """
        # layer 1
        # TODO: construct the conv1
        # layer 2
        # TODO: construct the pool1
        # layer 3
        # TODO: construct the conv2
        # layer 4
        # TODO: construct the pool2
        # layer 5
        # TODO: construct the fc1
        # layer 2
        # TODO: construct the fc2


        for layer in self.network.layers:
            feats = layer(feats)
        return feats


    def forward(self, feats):
        """
        Forward the network
        """
        return self.net(feats)

    @staticmethod
    def init_weight(shape):
        """
        Init weight parameter.
        """
        w = tf.truncated_normal(shape=shape, mean=0, stddev=0.1)
        return tf.Variable(w)

    @staticmethod
    def init_bias(shape):
        """
        Init bias parameter.
        """
        b = tf.zeros(shape)
        return tf.Variable(b)
