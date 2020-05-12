from IPython.display import clear_output
import matplotlib.pyplot as plt
import tensorflow as tf


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True', 'Predicted']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


from data import train
for image, mask in train.take(2):
    sample_image, sample_mask, sample_color = image, mask['mask_output'], mask['color_output']


def display_sample():
    display([sample_image, sample_mask])
    display([sample_image, tf.image.yuv_to_rgb(sample_color)])


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0][0]


def create_color(pred_mask):
    return pred_mask[1][0]


def show_predictions(model, dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask["mask_output"][0], create_mask(pred_mask)])
            display([image[0], mask["color_output"][0], create_color(pred_mask)])
            display([image[0], tf.image.yuv_to_rgb(mask["color_output"][0]), tf.image.yuv_to_rgb(create_color(pred_mask))])
    else:
        display([sample_image,
                 sample_mask,
                 create_mask(model.predict(sample_image[tf.newaxis, ...]))
                 ])
        display([sample_image,
                 sample_color,
                 create_color(model.predict(sample_image[tf.newaxis, ...]))
                 ])
        display([sample_image,
                 tf.image.yuv_to_rgb(sample_color),
                 tf.image.yuv_to_rgb(create_color(model.predict(sample_image[tf.newaxis, ...])))
                 ])


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))