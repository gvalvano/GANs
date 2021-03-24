"""
Implementation of spectral normalization layer.
Based on the implementations from:
   https://github.com/thisisiron/spectral_normalization-tf2
   https://github.com/taki0112/Spectral_Normalization-Tensorflow
   https://medium.com/@FloydHsiu0618/spectral-normalization-implementation-of-tensorflow-2-0-keras-api-d9060d26de77
"""
import tensorflow as tf  # TF 2.0


class SpectralNormalization(tf.keras.layers.Wrapper):

    def __init__(self, layer, n_iterations=1, training=True, **kwargs):

        self.num_iterations = n_iterations  # one iteration is usually enough
        self.do_power_iteration = training

        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError(
                'Please initialize `TimeDistributed` layer with a '
                '`Layer` instance. You passed: {input}'.format(input=layer))

        super(SpectralNormalization, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        self.layer.build(input_shape)

        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()

        self.v = self.add_weight(shape=(1, self.w_shape[0] * self.w_shape[1] * self.w_shape[2]),
                                 initializer=tf.initializers.RandomNormal(),
                                 trainable=False, name='sn_v', dtype=tf.float32)

        self.u = self.add_weight(shape=(1, self.w_shape[-1]),
                                 initializer=tf.initializers.RandomNormal(),
                                 trainable=False, name='sn_u', dtype=tf.float32)

        super(SpectralNormalization, self).build()

    def call(self, inputs):
        self.update_weights()
        output = self.layer(inputs)
        # self.restore_weights()  # Restore weights because of this formula "W = W - alpha * W_SN`"
        return output

    def update_weights(self):
        """Generate normalized weights.
        This method will update the value of self.layer.kernel with the
        normalized value, so that the layer is ready for call().
        """

        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])

        u_hat = self.u
        v_hat = self.v  # init v vector

        if self.do_power_iteration:
            # power iteration: one iteration is usually enough
            for _ in range(self.num_iterations):
                v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
                # v_hat = v_ / (tf.reduce_sum(v_ ** 2) ** 0.5 + self.eps)
                v_hat = tf.nn.l2_normalize(v_)

                u_ = tf.matmul(v_hat, w_reshaped)
                # u_hat = u_ / (tf.reduce_sum(u_ ** 2) ** 0.5 + self.eps)
                u_hat = tf.nn.l2_normalize(u_)

        u_hat = tf.stop_gradient(u_hat)
        v_hat = tf.stop_gradient(v_hat)

        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
        self.u.assign(u_hat)
        self.v.assign(v_hat)

        self.layer.kernel.assign(self.w / sigma)

    def restore_weights(self):
        self.layer.kernel.assign(self.w)
