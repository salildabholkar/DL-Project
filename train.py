from model import unet_model
from constants import *
from data import test_dataset, train_dataset
from display import show_predictions

import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import matplotlib.pyplot as plt


if __name__ == "__main__":
    model = unet_model(OUTPUT_CHANNELS)

    losses = {
        "mask_output": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        "color_output": "mse",
    }

    model.compile(optimizer='adam',
                  loss=losses,
                  metrics=['accuracy'])

    model.summary()

    model_history = model.fit(train_dataset, epochs=EPOCHS,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              validation_steps=VALIDATION_STEPS,
                              validation_data=test_dataset,
                              callbacks=[
                                  #  DisplayCallback(),
                                  tf.keras.callbacks.TensorBoard(
                                      log_dir='logs',
                                      histogram_freq=1,
                                      write_graph=True,
                                      write_images=True
                                  )
                              ])

    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    epochs = range(EPOCHS)

    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'bo', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()

    show_predictions(model, test_dataset, 3)
