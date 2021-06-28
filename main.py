import argparse
import tensorflow as tf
import utils
import models
import numpy as np
import os
import pickle
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import sys

tf.executing_eagerly()


def user_input():
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--epochs', help='Number of training epochs', type=int, required=False, default=300)
    parser.add_argument('-bs', '--batch_size', help='Number of samples in a mini-batch', type=int, required=False,
                        default=64)
    parser.add_argument('-tc', '--training_checkpoins', help='checkpoint directory',
                        type=str, required=False, default='training_checkpoints')
    parser.add_argument('-d', '--dest_dir', help='destination directory',
                        type=str, required=False, default='.')
    parser.add_argument('-r', '--restore_chkpt', help='continue with trained model',
                        type=int, required=False, default=0, choices=[0, 1])
    parser.add_argument('-gg', '--gen_gif', help='gen gif',
                        type=int, required=False, default=0, choices=[0, 1])
    parser.add_argument('-m', '--mode', help='whether to perform train or test',
                        type=str, required=False, default='train')

    args = parser.parse_args()
    arguments = vars(args)

    return arguments


class clusterGAN():

    def __init__(self, real_data, n_classes, epochs, batch_size, checkpoint_prefix, save_dir, seed=None):

        self.d_iter = 5
        self.save_dir = save_dir
        self.real_data = real_data
        self.n_classes = n_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.checkpoint_prefix = checkpoint_prefix
        self.generator = models.generator()
        self.discriminator = models.discriminator()
        self.encoder = models.encoder()
        self.ge_history = []
        self.d_history = []
        self.zn_cycle = []
        self.zc_cycle = []
        self.x_cycle = []
        self.ge_opt = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
        self.d_opt = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
        self.checkpoint = tf.train.Checkpoint(ge_optimizer=self.ge_opt,
                                              discriminator_optimizer=self.d_opt,
                                              generator=self.generator,
                                              discriminator=self.discriminator,
                                              encoder=self.encoder)
        self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_prefix, max_to_keep=2)

        self.scale = 10
        self.seed = seed

    # @tf.function
    def train_step(self, real_img):

        zn, zc = utils.z_sampler(real_img.shape[0])

        with tf.GradientTape() as ge_tape, tf.GradientTape() as d_tape:

            fake_img = self.generator([zn, zc], training=True)
            z_enc = self.encoder(fake_img, training=True)

            real_pred = self.discriminator(real_img, training=True)
            fake_pred = self.discriminator(fake_img, training=True)

            if self.iter == self.d_iter:
                # for every d_iter iterations of the discriminator, iterate generator_encoder once
                self.ge_loss = -tf.reduce_mean(fake_pred) + utils.e_loss([zn, zc], z_enc)
                ge_grad = ge_tape.gradient(self.ge_loss,
                                           self.generator.trainable_variables + self.encoder.trainable_variables)
                self.ge_opt.apply_gradients(zip(ge_grad,
                                                self.generator.trainable_variables + self.encoder.trainable_variables))
                self.iter = 0
            else:
                self.d_loss = tf.reduce_mean(fake_pred) - \
                              tf.reduce_mean(real_pred) + \
                              self.calc_penalty(real_img, fake_img)
                d_grad = d_tape.gradient(self.d_loss, self.discriminator.trainable_variables)
                self.d_opt.apply_gradients(zip(d_grad, self.discriminator.trainable_variables))

    def train(self, restore=False):

        if restore:
            self.checkpoint.restore(self.manager.latest_checkpoint)

        for epoch in range(self.epochs):

            tf.print(f'Epoch {epoch + 1}')
            self.iter = 0
            for batch in self.real_data:
                self.train_step(batch)
                self.iter += 1
            if (epoch + 1) % 3 == 0:
                tf.print(f'ge loss {float(self.ge_loss)}')
                tf.print(f'discriminator loss {float(self.d_loss)}')
                self.manager.save()

            self.cycle_loss(batch)
            self.ge_history.append(self.ge_loss)
            self.d_history.append(self.d_loss)

        return {'ge_loss': self.ge_history, 'd_loss': self.d_history}

    def calc_penalty(self, real_img, fake_img):  # gradient panelty for WGAN-GP

        eps = tf.random.uniform([real_img.shape[0], 1, 1, 1], 0.0, 0.1)
        x_hat = eps * real_img + (1 - eps) * fake_img
        with tf.GradientTape() as penalty_tape:
            penalty_tape.watch(x_hat)
            pred = self.discriminator(x_hat)

        ddx = penalty_tape.gradient(pred, x_hat)
        ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1) + 1e-8)
        ddx = tf.reduce_mean(tf.square(ddx - 1.0) * self.scale)

        return ddx

    def cycle_loss(self, real_img):

        zn, zc = utils.z_sampler(real_img.shape[0])

        latent = self.encoder(real_img, training=False)
        recon = self.generator([latent[:, :-self.n_classes], latent[:, -self.n_classes:]], training=False)

        recon_loss = tf.reduce_mean(tf.square(real_img - recon))
        self.x_cycle.append(recon_loss.numpy())

        fake_img = self.generator([zn, zc], training=False)
        latent_recon = self.encoder(fake_img, training=False)

        z_loss = utils.e_loss([zn, zc], latent_recon, combined_loss=False)

        zn_recon_loss = z_loss[1]
        self.zn_cycle.append(zn_recon_loss.numpy())

        zc_recon_loss = z_loss[0]
        self.zc_cycle.append(zc_recon_loss.numpy())

    def test(self, clip_val=0.6):

        self.clip_val = clip_val

        self.checkpoint.restore(self.manager.latest_checkpoint)
        y_list = []
        z_list = []

        for X, y in self.real_data:
            z = self.encoder(X, training=False)
            z_list.append(z)
            y_list.append(y)

        all_z = np.concatenate(z_list, axis=0)
        all_y = np.concatenate(y_list, axis=0)

        utils.cluster_viz(all_z, all_y, self.save_dir)

        acc_c, acc, ari, nmi = utils.cluster_latent(all_z, all_y)
        print(f'cluster eval- ARI: {ari}, NMI: {nmi}, ACC: {acc}, class ACC: {acc_c}')
        zc_acc, zc_acc_c, ari, nmi = utils.zc_acc(all_z[:, -self.n_classes:], all_y)
        print(f'zc classification acc - ACC: {zc_acc}, class ACC: {zc_acc_c}, NMI: {nmi}, ARI: {ari}')


def main(args):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    if args['mode'] == 'train':
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
        train_images = train_images / 255
        BUFFER_SIZE = 60000
        BATCH_SIZE = args['batch_size']
        train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        if not os.path.exists(args['dest_dir']):
            os.mkdir(args['dest_dir'])

        if args['gen_gif']:
            seed = utils.z_sampler(batch_size=16)
            if not os.path.exists(args['dest_dir'] + '/gif_ims'):
                os.mkdir(args['dest_dir'] + '/gif_ims')
        else:
            seed = None

        checkpoint_dir = args['dest_dir'] + '/' + args['training_checkpoins']
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

        cluster_gan = clusterGAN(real_data=train_dataset,
                                 n_classes=len(np.unique(train_labels)),
                                 epochs=args['epochs'],
                                 batch_size=BATCH_SIZE,
                                 checkpoint_prefix=checkpoint_prefix, save_dir=args['dest_dir'], seed=seed)

        history = cluster_gan.train(restore=args['restore_chkpt'])
        pickle.dump(history, open(args['dest_dir'] + '/history.pkl', 'wb'))

        fig, ax = plt.subplots(3, 1, sharex=True)
        ax[0].plot(range(len(cluster_gan.zn_cycle)), cluster_gan.zn_cycle)
        ax[0].set_title('||z_n-E(G(z_n))||')
        ax[1].plot(range(len(cluster_gan.zc_cycle)), cluster_gan.zc_cycle)
        ax[1].set_title('H(z_c,E(G(z_c)))')
        ax[2].plot(range(len(cluster_gan.x_cycle)), cluster_gan.x_cycle)
        ax[2].set_title('||x-G(E(x))||')
        ax[2].set_xlabel('Epoch')
        plt.tight_layout()
        plt.savefig(args['dest_dir'] + '/cycle_loss.png')

    elif args['mode'] == 'test':

        checkpoint_dir = args['dest_dir'] + '/' + args['training_checkpoins']
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
        test_images = test_images / 255
        BUFFER_SIZE = 60000
        BATCH_SIZE = args['batch_size']
        test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).shuffle(BUFFER_SIZE).batch(
            BATCH_SIZE)

        cluster_gan = clusterGAN(real_data=test_dataset,
                                 n_classes=len(np.unique(train_labels)),
                                 epochs=args['epochs'],
                                 batch_size=BATCH_SIZE,
                                 checkpoint_prefix=checkpoint_prefix, save_dir=args['dest_dir'])

        cluster_gan.test()


if __name__ == '__main__':
    args = user_input()
    main(args)
