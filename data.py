import tensorflow as tf
import tensorflow_datasets as tfds
from constants import BATCH_SIZE, BUFFER_SIZE


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


@tf.function
def load_image_train(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)
    gray_image = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(input_image))

    return tf.image.rgb_to_yuv(gray_image), { "mask_output": input_mask, "color_output": tf.image.rgb_to_yuv(input_image) }


def load_image_test(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    input_image, input_mask = normalize(input_image, input_mask)
    gray_image = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(input_image))

    return tf.image.rgb_to_yuv(gray_image), { "mask_output": input_mask, "color_output": tf.image.rgb_to_yuv(input_image) }


dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(load_image_test)

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)