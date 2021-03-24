from typing import List, Any, Tuple
import tensorflow as tf

ortho_init = tf.keras.initializers.Orthogonal()


class UNetEncoder(tf.keras.Model):

    def __init__(self, num_filters: List[int], **kwargs: Any):
        super().__init__()
        assert len(num_filters) >= 1

        self.blocks = [
            self.build_block(f, name=f'block_{i}', **kwargs)
            for i, f in enumerate(num_filters)]
        self.downsample_ops = [
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='same', name=f'downsample_{i}')
            for i in range(len(num_filters))]

    @staticmethod
    def build_block(num_filters, name, **kwargs) -> tf.keras.Sequential:
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=num_filters, kernel_size=3, padding='same',
                                   kernel_initializer=ortho_init, **kwargs),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),

            tf.keras.layers.Conv2D(filters=num_filters, kernel_size=3, padding='same',
                                   kernel_initializer=ortho_init,  **kwargs),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
        ], name=name)

    def call(self, x: tf.Tensor) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """
        Call the encoder
        """
        feature_maps = []
        y = x
        for block, downsample in zip(self.blocks, self.downsample_ops):
            y = block(y)
            feature_maps.append(y)
            y = downsample(y)
        return y, feature_maps


class UNetDecoder(tf.keras.Model):

    def __init__(self, num_filters: List[int], **kwargs: Any):
        super().__init__()
        assert len(num_filters) >= 1
        self.blocks = [
            self.build_block(f, name=f'block_{i}', **kwargs)
            for i, f in enumerate(num_filters)]

        self.upsample_ops = [
            tf.keras.layers.UpSampling2D(size=(2, 2), name=f'upsample_{i}', interpolation='nearest')
            for i in range(len(num_filters) - 1)]
        self.upsample_ops += [lambda x: x]

    @staticmethod
    def build_block(num_filters: int, name: str, **kwargs: Any) -> tf.keras.Sequential:
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=num_filters, kernel_size=3, padding='same',
                                   kernel_initializer=ortho_init, **kwargs),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),

            tf.keras.layers.Conv2D(filters=num_filters, kernel_size=3, padding='same',
                                   kernel_initializer=ortho_init,  **kwargs),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu')
        ], name=name)

    def call(self, initial: tf.Tensor, feature_maps: List[tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        """
        assert len(feature_maps) == len(self.blocks) == len(self.upsample_ops)

        y_u = initial
        for feature_maps, block, upsampler in zip(feature_maps, self.blocks, self.upsample_ops):
            x = tf.keras.backend.concatenate((feature_maps, y_u))
            y = block(x)
            y_u = upsampler(y)

        return y  # == y_u


class UNet(tf.keras.Model):

    def __init__(self, output_channels, num_filters: List[int], **kwargs):
        super().__init__()

        *enc_features, bottleneck_features = num_filters
        self.encoder = UNetEncoder(list(enc_features), **kwargs)
        self.decoder = UNetDecoder(list(reversed(enc_features)), **kwargs)
        self.bottleneck = self.build_bottleneck(bottleneck_features, **kwargs)
        self.tail = self.build_tail(enc_features[0], output_channels, **kwargs)

    @staticmethod
    def build_bottleneck(features: int, **kwargs: Any) -> tf.keras.Sequential:
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=features, kernel_size=3, padding='same',
                                   kernel_initializer=ortho_init,  **kwargs),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),

            tf.keras.layers.Conv2D(filters=features, kernel_size=3, padding='same',
                                   kernel_initializer=ortho_init,  **kwargs),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),

            tf.keras.layers.UpSampling2D()
        ], name='bottleneck')

    @staticmethod
    def build_tail(features: int, output_channels: int, **kwargs: Any) -> tf.keras.Sequential:
        layers = [
            tf.keras.layers.Conv2D(filters=features, kernel_size=3, padding='same',
                                   kernel_initializer=ortho_init, **kwargs),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),

            tf.keras.layers.Conv2D(filters=features, kernel_size=3, padding='same',
                                   kernel_initializer=ortho_init, **kwargs),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu')
        ]

        # output layer:
        layers += [tf.keras.layers.Conv2D(filters=output_channels, kernel_size=3, padding='same',
                                          kernel_initializer=ortho_init, **kwargs)]

        return tf.keras.Sequential(layers, name='tail')

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        """
        y, feature_maps = self.encoder(x)
        y = self.bottleneck(y)
        y = self.decoder(y, feature_maps[::-1])
        return self.tail(y)
