import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard, ModelCheckpoint
import time
import numpy as np
import tensorflow_datasets as tfds

from model import *
from config import *


def save_chart(history, path, initial_epoch, epochs):
    acc = history.history["categorical_accuracy"]
    val_acc = history.history["val_categorical_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(initial_epoch, epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.savefig(path)


def save_train_fine_model_chart(historys, path, initial_epoch, epochs):
    epochs_range = range(initial_epoch, epochs)
    plt.figure(figsize=(32, 32))

    plt.subplot(4, 1, 1)
    count = 0
    for history in historys:
        acc = history.history["categorical_accuracy"]

        plt.plot(epochs_range, acc, label="Fine {}".format(count))

        count += 1
    plt.legend(loc="lower right")
    plt.title("Training Accuracy")

    plt.subplot(4, 1, 2)
    count = 0
    for history in historys:
        val_acc = history.history["val_categorical_accuracy"]

        plt.plot(epochs_range, val_acc, label="Fine {}".format(count))

        count += 1
    plt.legend(loc="lower right")
    plt.title("Validation Accuracy")

    plt.subplot(4, 1, 3)
    count = 0
    for history in historys:
        loss = history.history["loss"]

        plt.plot(epochs_range, loss, label="Fine {}".format(count))

        count += 1
    plt.legend(loc="lower right")
    plt.title("Training Loss")

    plt.subplot(4, 1, 4)
    count = 0
    for history in historys:
        val_loss = history.history["val_loss"]

        plt.plot(epochs_range, val_loss, label="Fine {}".format(count))

        count += 1
    plt.legend(loc="lower right")
    plt.title("Validation Loss")

    plt.savefig(path)


def train_single_classifier(initial_epoch, epochs):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        FINE_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=123,
        image_size=(HD_CNN_IMG_WIDTH, HD_CNN_IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        FINE_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=123,
        image_size=(HD_CNN_IMG_WIDTH, HD_CNN_IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    model = single_classifier_model(
        SINGLE_CLASSIFIER_MODEL_LEARNING_RATE,
        HD_CNN_IMG_WIDTH,
        HD_CNN_IMG_HEIGHT,
        HD_CNN_IMG_CHANNEL,
        FINE_CLASS_NUM,
    )

    current_time = str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    log_dir = os.path.join(LOG_DIR, current_time)
    tensorboard = TensorBoard(
        log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True
    )

    checkpoint = ModelCheckpoint(
        filepath=SINGLE_CLASSIFIER_MODEL_WEIGHTS_PATH,
        save_weights_only=True,
        save_best_only=True,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=initial_epoch,
        epochs=epochs,
        callbacks=[tensorboard, checkpoint],
    )

    save_chart(
        history,
        os.path.join(WORKING_DIR, "single_classifier_train_chart.png"),
        initial_epoch,
        epochs,
    )


def train_coarse_classifier(initial_epoch, epochs):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        COARSE_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=123,
        image_size=(HD_CNN_IMG_WIDTH, HD_CNN_IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        COARSE_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=123,
        image_size=(HD_CNN_IMG_WIDTH, HD_CNN_IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    model = coarse_classifier_model(COARSE_CLASSIFIER_MODEL_LEARNING_RATE)

    current_time = str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    log_dir = os.path.join(LOG_DIR, current_time)
    tensorboard = TensorBoard(
        log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True
    )

    checkpoint = ModelCheckpoint(
        filepath=COARSE_CLASSIFIER_MODEL_WEIGHTS_PATH,
        save_weights_only=True,
        save_best_only=True,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=initial_epoch,
        epochs=epochs,
        callbacks=[tensorboard, checkpoint],
    )

    save_chart(
        history,
        os.path.join(WORKING_DIR, "coarse_classifier_train_chart.png"),
        initial_epoch,
        epochs,
    )


def train_fine_classifier(initial_epoch, epochs):
    historys = []

    for i in range(COARSE_CLASS_NUM):
        train_ds = tf.keras.utils.image_dataset_from_directory(
            os.path.join(COARSE_FINE_DIR, str(i)),
            validation_split=VALIDATION_SPLIT,
            subset="training",
            seed=123,
            image_size=(HD_CNN_IMG_WIDTH, HD_CNN_IMG_HEIGHT),
            batch_size=BATCH_SIZE,
            label_mode="categorical",
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            os.path.join(COARSE_FINE_DIR, str(i)),
            validation_split=VALIDATION_SPLIT,
            subset="validation",
            seed=123,
            image_size=(HD_CNN_IMG_WIDTH, HD_CNN_IMG_HEIGHT),
            batch_size=BATCH_SIZE,
            label_mode="categorical",
        )

        model = fine_classifier_model(FINE_CLASSIFIER_MODEL_LEARNING_RATE)

        current_time = str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
        log_dir = os.path.join(LOG_DIR, current_time)
        tensorboard = TensorBoard(
            log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True
        )

        filepath = FINE_CLASSIFIER_MODEL_WEIGHTS_PATH.format(str(i))
        checkpoint = ModelCheckpoint(
            filepath=filepath,
            save_weights_only=True,
            save_best_only=True,
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=initial_epoch,
            epochs=epochs,
            callbacks=[tensorboard, checkpoint],
        )

        historys.append(history)

    save_train_fine_model_chart(
        historys,
        os.path.join(WORKING_DIR, "fine_classifier_train_chart.png"),
        initial_epoch,
        epochs,
    )


def train_vgg16(initial_epoch, epochs, batch_size, validation_split):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        FINE_DIR,
        validation_split=validation_split,
        subset="training",
        seed=123,
        image_size=(HD_CNN_IMG_WIDTH, HD_CNN_IMG_HEIGHT),
        batch_size=batch_size,
        label_mode="categorical",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        FINE_DIR,
        validation_split=validation_split,
        subset="validation",
        seed=123,
        image_size=(HD_CNN_IMG_WIDTH, HD_CNN_IMG_HEIGHT),
        batch_size=batch_size,
        label_mode="categorical",
    )

    model = vgg16_model(learning_rate=0.0001, dropout_rate=0.5)

    current_time = str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    log_dir = os.path.join(LOG_DIR, current_time)
    tensorboard = TensorBoard(
        log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True
    )

    checkpoint = ModelCheckpoint(
        filepath=os.path.join(WEIGHTS_DIR, "vgg16_model", "cp.ckpt"),
        save_weights_only=True,
        save_best_only=True,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=initial_epoch,
        epochs=epochs,
        callbacks=[tensorboard, checkpoint],
    )

    save_chart(
        history,
        os.path.join(WORKING_DIR, "single_classifier_train_chart.png"),
        initial_epoch,
        epochs,
    )


def get_error(y, yh):
    # Threshold
    yht = np.zeros(np.shape(yh))
    yht[np.arange(len(yh)), yh.argmax(1)] = 1
    # Evaluate Error
    error = np.count_nonzero(np.count_nonzero(y - yht, 1)) / len(y)
    return error


def get_probabilistic_averaging_result(x_test):
    coarse_predictions = []
    fine_predictions = []
    final_predictions = []

    coarse_layer = coarse_classifier_model(
        COARSE_CLASSIFIER_MODEL_LEARNING_RATE, load_weight=True
    )
    for x in x_test:
        prediction = coarse_layer.predict(tf.expand_dims(x, 0))
        coarse_predictions.append(prediction[0])

    for i in range(COARSE_CLASS_NUM):
        fine_layer = fine_classifier_model(
            FINE_CLASSIFIER_MODEL_LEARNING_RATE, load_weight=True, class_idx=i
        )
        predictions = []

        for x in x_test:
            prediction = fine_layer.predict(tf.expand_dims(x, 0))
            predictions.append(prediction[0])

        fine_predictions.append(predictions)

    prediction_size = len(coarse_predictions)
    for img in range(prediction_size):
        proba = [0] * FINE_CLASS_NUM
        for finec in range(FINE_CLASS_NUM):
            for coarsec in range(COARSE_CLASS_NUM):
                proba[finec] += (
                    coarse_predictions[img][coarsec]
                    * fine_predictions[coarsec][img][finec]
                )
        final_predictions.append(proba)

    return np.array(final_predictions)


if __name__ == "__main__":
    if os.path.exists(LOG_DIR) == False:
        os.mkdir(LOG_DIR)
    if os.path.exists(WEIGHTS_DIR) == False:
        os.mkdir(WEIGHTS_DIR)

    train_single_classifier(initial_epoch=0, epochs=50)
    train_coarse_classifier(initial_epoch=50, epochs=100)
    train_fine_classifier(initial_epoch=0, epochs=50)
