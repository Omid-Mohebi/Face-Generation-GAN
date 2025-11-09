import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import kagglehub
from kagglehub import KaggleDatasetAdapter
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Layer, LeakyReLU, Conv2DTranspose, Add, Conv2D, MaxPool2D, Flatten, InputLayer, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

path = kagglehub.dataset_download("jessicali9530/celeba-dataset")
print("Downloaded to:", path)


image_dir = os.path.join(path, "img_align_celeba/img_align_celeba")

image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")]

num_images = 9
sample_files = random.sample(image_files, num_images)

plt.figure(figsize=(10, 10))
for i, file in enumerate(sample_files):
    img_path = os.path.join(image_dir, file)
    img = Image.open(img_path)

    plt.subplot(3, 3, i + 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(file)

plt.tight_layout()
plt.show()



batch_size = 128
image_shape = (64,64, 3)
lr = 2e-4
l_dim = 100


raw_data = tf.keras.preprocessing.image_dataset_from_directory(
    image_dir,
    label_mode=None,
    batch_size=batch_size,
    image_size=(image_shape[0], image_shape[1])
)


def preprocess_image(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1.0
    return image


train_data = (
    raw_data
    .map(preprocess_image)
    .unbatch()
    .shuffle(buffer_size=1024, reshuffle_each_iteration = True)
    .batch(batch_size, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)


plt.figure(figsize=(10, 10))
n = 6

for images in train_data.take(1):
    for i in range(n):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow((images[i].numpy() * 0.5 + 0.5))
        plt.axis("off")

plt.tight_layout()
plt.show()


generator = tf.keras.Sequential([
    Input(shape = (l_dim,)),
    Dense(4*4*l_dim),
    Reshape((4,4,l_dim)),

    Conv2DTranspose(512, kernel_size = 4, strides = 2, padding = "same"),
    BatchNormalization(),
    LeakyReLU(0.2),

    Conv2DTranspose(256, kernel_size = 4, strides = 2, padding = "same"),
    BatchNormalization(),
    LeakyReLU(0.2),

    Conv2DTranspose(128, kernel_size = 4, strides = 2, padding = "same"),
    BatchNormalization(),
    LeakyReLU(0.2),

    Conv2DTranspose(3, kernel_size = 4, strides = 2, padding="same", activation = tf.keras.activations.tanh)
], name = 'generator')


generator.summary()


discriminator = tf.keras.Sequential([
    Input(shape = (image_shape[0], image_shape[1], 3)),

    Conv2D(64, kernel_size = 4, strides = 2, padding = "same"),
    LeakyReLU(0.2),

    Conv2D(128, kernel_size = 4, strides = 2, padding = "same"),
    BatchNormalization(),
    LeakyReLU(0.2),

    Conv2D(256, kernel_size = 4, strides = 2, padding = "same"),
    BatchNormalization(),
    LeakyReLU(0.2),

    Conv2D(1, kernel_size = 4, strides = 2, padding = "valid"),
    Flatten(),
    Dense(1, activation = 'sigmoid')
], name = 'discriminator')


discriminator.summary()


class GAN(tf.keras.Model):
  def __init__(self, discriminator, generator):
    super(GAN, self).__init__()
    self.discriminator = discriminator
    self.generator = generator

  def compile(self, d_optimizer, g_optimizer, loss):
    super(GAN, self).compile()
    self.d_optimizer = d_optimizer
    self.g_optimizer = g_optimizer
    self.loss = loss
    self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
    self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")

  @property
  def metrics(self):
    return [self.d_loss_metric, self.g_loss_metric]

  def train_step(self, real_images):
    batch_size = tf.shape(real_images)[0]
    noise = tf.random.normal(shape = (batch_size, l_dim))

    fake_images = self.generator(noise)
    real_label = tf.ones((batch_size, 1)) + 0.25 * tf.random.uniform((batch_size, 1), minval=-1, maxval=1)
    fake_label = tf.zeros((batch_size, 1))  + 0.25 * tf.random.uniform((batch_size, 1))

    with tf.GradientTape() as recorder:
      real_output = self.discriminator(real_images)
      d_loss_real = self.loss(real_label, real_output)
      fake_output = self.discriminator(fake_images)
      d_loss_fake = self.loss(fake_label, fake_output)
      d_loss = d_loss_real + d_loss_fake

    grads = recorder.gradient(d_loss, self.discriminator.trainable_weights)
    self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))



    noise = tf.random.normal(shape = (batch_size, l_dim))
    misleading_labels = tf.ones((batch_size, 1))

    with tf.GradientTape() as recorder:
      # fake_images = self.generator(noise, training=True)
      fake_output = self.discriminator(self.generator(noise))
      g_loss = self.loss(misleading_labels, fake_output)

    grads = recorder.gradient(g_loss, self.generator.trainable_weights)
    self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))


    self.d_loss_metric.update_state(d_loss)
    self.g_loss_metric.update_state(g_loss)

    return {
        "d_loss": self.d_loss_metric.result(),
        "g_loss": self.g_loss_metric.result(),
    }





gan = GAN(discriminator=discriminator, generator=generator)
gan.compile(
    d_optimizer=Adam(learning_rate=lr, beta_1=0.5),
    g_optimizer=Adam(learning_rate=lr, beta_1=0.5),
    loss=tf.keras.losses.BinaryCrossentropy(),
)


class ShowImage(tf.keras.callbacks.Callback):
  def __init__(self, latent_dim = 100):
    self.latent_dim = latent_dim

  def on_epoch_end(self, epoch, logs=None):
    n = 6
    k = 0
    out = self.model.generator(tf.random.normal(shape = (36, self.latent_dim)))
    plt.figure(figsize=(16, 16))
    for i in range(n):
      for j in range(n):
        ax = plt.subplot(n, n, k+1)
        plt.imshow((out[k] + 1) / 2,)
        plt.axis("off")
        k += 1
    os.makedirs('gen', exist_ok=True)
    plt.savefig('gen/image_in_epoch_{}.png'.format(epoch))
    plt.show()


history = gan.fit(train_data, epochs=100, callbacks=[ShowImage(latent_dim=l_dim)])