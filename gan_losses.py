import tensorflow as tf


class LeastSquareGAN(object):
    """ Least squares GAN losses.
    See `Least Squares Generative Adversarial Networks` (https://arxiv.org/abs/1611.04076) for more details.
    """
    def __init__(self):
        super(LeastSquareGAN, self).__init__()
        self.real_label = 1.0
        self.fake_label = -1.0

    @staticmethod
    def generator_loss(disc_pred_fake, real_label=1.0):
        loss = 0.5 * tf.reduce_mean(input_tensor=tf.math.squared_difference(disc_pred_fake, real_label))
        return loss

    @staticmethod
    def discriminator_loss(disc_pred_real, disc_pred_fake, real_label=1.0, fake_label=0.0):
        loss = 0.5 * tf.reduce_mean(input_tensor=tf.math.squared_difference(disc_pred_real, real_label)) + \
               0.5 * tf.reduce_mean(input_tensor=tf.math.squared_difference(disc_pred_fake, fake_label))
        return loss

    @staticmethod
    def discriminator_fake_loss(disc_pred_fake, fake_label=0.0):
        loss = 0.5 * tf.reduce_mean(input_tensor=tf.math.squared_difference(disc_pred_fake, fake_label))
        return loss

    @staticmethod
    def discriminator_real_loss(disc_pred_real, real_label=1.0):
        loss = 0.5 * tf.reduce_mean(input_tensor=tf.math.squared_difference(disc_pred_real, real_label))
        return loss


class NonSaturatingGAN(object):
    """ Modified GAN losses.
    See `Generative Adversarial Nets` (https://arxiv.org/abs/1406.2661) for more details.
    If modified==True, it uses the modified loss suggested by the authors; otherwise, it will compute the losses as in
    the vanilla GAN (not recommended).
    """
    def __init__(self):
        super(NonSaturatingGAN, self).__init__()
        self.real_label = 1.0
        self.fake_label = 0.0

    @staticmethod
    def generator_loss(disc_pred_fake):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_pred_fake, labels=tf.ones_like(disc_pred_fake))
        return tf.reduce_mean(loss)

    @staticmethod
    def discriminator_loss(disc_pred_real, disc_pred_fake):
        loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_pred_real, labels=tf.ones_like(disc_pred_real))
        loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_pred_fake, labels=tf.zeros_like(disc_pred_fake))
        return tf.reduce_mean(loss_real) + tf.reduce_mean(loss_fake)

    @staticmethod
    def discriminator_fake_loss(disc_pred_fake):
        loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_pred_fake, labels=tf.zeros_like(disc_pred_fake))
        return tf.reduce_mean(loss_fake)

    @staticmethod
    def discriminator_real_loss(disc_pred_real):
        loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_pred_real, labels=tf.ones_like(disc_pred_real))
        return tf.reduce_mean(loss_real)


class WassersteinGAN(object):
    """ Wasserstein GAN losses.
    See `Wasserstein GAN` (https://arxiv.org/abs/1701.07875) for more details."""
    def __init__(self):
        super(WassersteinGAN, self).__init__()
        self.real_label = 0.0
        self.fake_label = 1e8
        raise NotImplementedError  # todo: double check generator loss
        # self.default_real_label =

    # @staticmethod
    # def generator_loss(disc_pred_fake, real_label=1.0):
    #     loss = 0.5 * tf.reduce_mean(input_tensor=tf.math.squared_difference(disc_pred_fake, real_label))
    #     return loss

    @staticmethod
    def discriminator_loss(disc_pred_real, disc_pred_fake):
        loss = tf.reduce_mean(disc_pred_real) - tf.reduce_mean(disc_pred_fake)
        return loss

    @staticmethod
    def gradient_penalty(discriminator_model, x_real, x_gen, gp_weight=10.0):
        epsilon = tf.random.uniform([x_real.shape[0], 1, 1, 1], 0.0, 1.0)
        x_hat = epsilon * x_real + (1 - epsilon) * x_gen

        with tf.GradientTape() as tape:
            tape.watch(x_hat)
            d_hat = discriminator_model(x_hat)
        gradients = tape.gradient(d_hat, x_hat)
        ddx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2]))

        penalty = tf.reduce_mean((ddx - 1.0) ** 2)

        return gp_weight * penalty
