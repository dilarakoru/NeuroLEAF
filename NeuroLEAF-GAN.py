# Adım 1: Kütüphaneleri ve veri setini yüklemek

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Veri setini yükleyin veya veri setinizin yükleme kodlarını buraya ekleyin

# Adım 2: Veri setini hazırlamak

# Veri setinizi GAN modeline uygun hale getirmek için gerekli dönüşümleri uygulayın.

# Veri setinizi GAN modeline uygun şekilde yükleyin.

# Adım 3: Discriminator (Ayırt Edici) modelini oluşturmak

def make_discriminator_model():
    model = keras.Sequential()
    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.5))

    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

discriminator = make_discriminator_model()


# Adım 4: Generator (Üretici) modelini oluşturmak

def make_generator_model():
    model = keras.Sequential()
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model

generator = make_generator_model()



# Adım 5: Discriminator (Ayırt Edici) modelini derlemek


cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

@tf.function
def train_discriminator(images):
    with tf.GradientTape() as disc_tape:
        generated_images = generator(tf.random.normal([batch_size, 100]), training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return disc_loss


# Adım 6: Generator (Üretici) modelini derlemek


generator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

@tf.function
def train_generator():
    with tf.GradientTape() as gen_tape:
        generated_images = generator(tf.random.normal([batch_size, 100]), training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return gen_loss

# Adım 7: Eğitim döngüsünü tanımlamak

def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            disc_loss = train_discriminator(image_batch)
            gen_loss = train_generator()

        # Her epoch sonunda generatör tarafından üretilen birkaç örneği görselleştirin
        generate_and_save_images(generator, epoch + 1, seed)

train(dataset, epochs=50)
