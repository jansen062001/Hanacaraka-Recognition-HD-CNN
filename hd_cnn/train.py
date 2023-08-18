import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard, ModelCheckpoint
import time
import numpy as np
import argparse
import os

from . import model, config


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

    plt.close()


def save_train_fine_model_chart(historys, path, initial_epoch, epochs):
    plt_loc = "lower right"
    plt_label = "Fine {}"

    epochs_range = range(initial_epoch, epochs)
    plt.figure(figsize=(32, 32))

    plt.subplot(4, 1, 1)
    count = 0
    for history in historys:
        acc = history.history["categorical_accuracy"]

        plt.plot(epochs_range, acc, label=plt_label.format(count))

        count += 1
    plt.legend(loc=plt_loc)
    plt.title("Training Accuracy")

    plt.subplot(4, 1, 2)
    count = 0
    for history in historys:
        val_acc = history.history["val_categorical_accuracy"]

        plt.plot(epochs_range, val_acc, label=plt_label.format(count))

        count += 1
    plt.legend(loc=plt_loc)
    plt.title("Validation Accuracy")

    plt.subplot(4, 1, 3)
    count = 0
    for history in historys:
        loss = history.history["loss"]

        plt.plot(epochs_range, loss, label=plt_label.format(count))

        count += 1
    plt.legend(loc=plt_loc)
    plt.title("Training Loss")

    plt.subplot(4, 1, 4)
    count = 0
    for history in historys:
        val_loss = history.history["val_loss"]

        plt.plot(epochs_range, val_loss, label=plt_label.format(count))

        count += 1
    plt.legend(loc=plt_loc)
    plt.title("Validation Loss")

    plt.savefig(path)


def train_single_classifier(initial_epoch, epochs):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(config.TRAIN_DATASET_DIR, "fine"),
        seed=123,
        image_size=(config.HD_CNN_IMG_WIDTH, config.HD_CNN_IMG_HEIGHT),
        batch_size=config.BATCH_SIZE,
        label_mode="categorical",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(config.VALID_DATASET_DIR, "fine"),
        seed=123,
        image_size=(config.HD_CNN_IMG_WIDTH, config.HD_CNN_IMG_HEIGHT),
        batch_size=config.BATCH_SIZE,
        label_mode="categorical",
    )

    single_classifier_model = model.single_classifier_model(
        config.SINGLE_CLASSIFIER_MODEL_LEARNING_RATE,
        config.HD_CNN_IMG_WIDTH,
        config.HD_CNN_IMG_HEIGHT,
        config.HD_CNN_IMG_CHANNEL,
        config.FINE_CLASS_NUM,
    )

    current_time = str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    log_dir = os.path.join(config.LOG_DIR, current_time)
    tensorboard = TensorBoard(
        log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True
    )

    checkpoint = ModelCheckpoint(
        filepath=config.SINGLE_CLASSIFIER_MODEL_WEIGHTS_PATH,
        save_weights_only=True,
        save_best_only=True,
    )

    history = single_classifier_model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=initial_epoch,
        epochs=epochs,
        callbacks=[tensorboard, checkpoint],
    )

    save_chart(
        history,
        os.path.join(config.HD_CNN_DIR, "data", "single_classifier_train_chart.png"),
        initial_epoch,
        epochs,
    )


def train_coarse_classifier(initial_epoch, epochs):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(config.TRAIN_DATASET_DIR, "coarse"),
        seed=123,
        image_size=(config.HD_CNN_IMG_WIDTH, config.HD_CNN_IMG_HEIGHT),
        batch_size=config.BATCH_SIZE,
        label_mode="categorical",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(config.VALID_DATASET_DIR, "coarse"),
        seed=123,
        image_size=(config.HD_CNN_IMG_WIDTH, config.HD_CNN_IMG_HEIGHT),
        batch_size=config.BATCH_SIZE,
        label_mode="categorical",
    )

    coarse_classifier_model = model.coarse_classifier_model(
        config.COARSE_CLASSIFIER_MODEL_LEARNING_RATE
    )

    current_time = str(time.strftime(config.LOG_TIME_FORMAT, time.localtime()))
    log_dir = os.path.join(config.LOG_DIR, current_time)
    tensorboard = TensorBoard(
        log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True
    )

    checkpoint = ModelCheckpoint(
        filepath=config.COARSE_CLASSIFIER_MODEL_WEIGHTS_PATH,
        save_weights_only=True,
        save_best_only=True,
    )

    history = coarse_classifier_model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=initial_epoch,
        epochs=epochs,
        callbacks=[tensorboard, checkpoint],
    )

    save_chart(
        history,
        os.path.join(config.HD_CNN_DIR, "data", "coarse_classifier_train_chart.png"),
        initial_epoch,
        epochs,
    )


def train_fine_classifier(initial_epoch, epochs, class_idx):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(config.TRAIN_DATASET_DIR, "coarse_fine", str(class_idx)),
        seed=123,
        image_size=(config.HD_CNN_IMG_WIDTH, config.HD_CNN_IMG_HEIGHT),
        batch_size=config.BATCH_SIZE,
        label_mode="categorical",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(config.VALID_DATASET_DIR, "coarse_fine", str(class_idx)),
        seed=123,
        image_size=(config.HD_CNN_IMG_WIDTH, config.HD_CNN_IMG_HEIGHT),
        batch_size=config.BATCH_SIZE,
        label_mode="categorical",
    )

    fine_classifier_model = model.fine_classifier_model(
        config.FINE_CLASSIFIER_MODEL_LEARNING_RATE
    )

    current_time = str(time.strftime(config.LOG_TIME_FORMAT, time.localtime()))
    log_dir = os.path.join(config.LOG_DIR, current_time)
    tensorboard = TensorBoard(
        log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True
    )

    filepath = config.FINE_CLASSIFIER_MODEL_WEIGHTS_PATH.format(num=str(class_idx))
    checkpoint = ModelCheckpoint(
        filepath=filepath,
        save_weights_only=True,
        save_best_only=True,
    )

    history = fine_classifier_model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=initial_epoch,
        epochs=epochs,
        callbacks=[tensorboard, checkpoint],
    )

    return history


def train_vgg16(initial_epoch, epochs, batch_size):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(config.TRAIN_DATASET_DIR, "fine"),
        seed=123,
        image_size=(config.HD_CNN_IMG_WIDTH, config.HD_CNN_IMG_HEIGHT),
        batch_size=batch_size,
        label_mode="categorical",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(config.VALID_DATASET_DIR, "fine"),
        seed=123,
        image_size=(config.HD_CNN_IMG_WIDTH, config.HD_CNN_IMG_HEIGHT),
        batch_size=batch_size,
        label_mode="categorical",
    )

    vgg16_model = model.vgg16_model(learning_rate=0.001)

    current_time = str(time.strftime(config.LOG_TIME_FORMAT, time.localtime()))
    log_dir = os.path.join(config.LOG_DIR, current_time)
    tensorboard = TensorBoard(
        log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True
    )

    checkpoint = ModelCheckpoint(
        filepath=os.path.join(config.WEIGHTS_DIR, "vgg16_model", "cp.ckpt"),
        save_weights_only=True,
        save_best_only=True,
    )

    history = vgg16_model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=initial_epoch,
        epochs=epochs,
        callbacks=[tensorboard, checkpoint],
    )

    save_chart(
        history,
        os.path.join(config.HD_CNN_DIR, "data", "vgg16_train_chart.png"),
        initial_epoch,
        epochs,
    )


def train_vgg19(initial_epoch, epochs, batch_size):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(config.TRAIN_DATASET_DIR, "fine"),
        seed=123,
        image_size=(config.HD_CNN_IMG_WIDTH, config.HD_CNN_IMG_HEIGHT),
        batch_size=batch_size,
        label_mode="categorical",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(config.VALID_DATASET_DIR, "fine"),
        seed=123,
        image_size=(config.HD_CNN_IMG_WIDTH, config.HD_CNN_IMG_HEIGHT),
        batch_size=batch_size,
        label_mode="categorical",
    )

    vgg19_model = model.vgg19_model(learning_rate=0.001)

    current_time = str(time.strftime(config.LOG_TIME_FORMAT, time.localtime()))
    log_dir = os.path.join(config.LOG_DIR, current_time)
    tensorboard = TensorBoard(
        log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True
    )

    checkpoint = ModelCheckpoint(
        filepath=os.path.join(config.WEIGHTS_DIR, "vgg19_model", "cp.ckpt"),
        save_weights_only=True,
        save_best_only=True,
    )

    history = vgg19_model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=initial_epoch,
        epochs=epochs,
        callbacks=[tensorboard, checkpoint],
    )

    save_chart(
        history,
        os.path.join(config.HD_CNN_DIR, "data", "vgg19_train_chart.png"),
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

    coarse_layer = model.coarse_classifier_model(
        config.COARSE_CLASSIFIER_MODEL_LEARNING_RATE, load_weight=True
    )
    predictions = coarse_layer.predict(x_test, batch_size=1)
    for i in range(len(predictions)):
        coarse_predictions.append(predictions[i])

    coarse_idx = []
    for i in range(config.COARSE_CLASS_NUM):
        coarse_idx.append(str(i))
    coarse_idx.sort()
    for i in range(len(coarse_idx)):
        coarse_idx[i] = int(coarse_idx[i])

    for i in coarse_idx:
        fine_layer = model.fine_classifier_model(
            config.FINE_CLASSIFIER_MODEL_LEARNING_RATE, load_weight=True, class_idx=i
        )

        arr_predictions = []
        predictions = fine_layer.predict(x_test, batch_size=1)
        for j in range(len(predictions)):
            arr_predictions.append(predictions[j])

        fine_predictions.append(arr_predictions)

    prediction_size = len(coarse_predictions)
    for img in range(prediction_size):
        proba = [0] * config.FINE_CLASS_NUM
        for finec in range(config.FINE_CLASS_NUM):
            for coarsec in range(config.COARSE_CLASS_NUM):
                proba[finec] += (
                    coarse_predictions[img][coarsec]
                    * fine_predictions[coarsec][img][finec]
                )
        final_predictions.append(proba)

    return np.array(final_predictions)


def main(args):
    if not os.path.exists(config.LOG_DIR):
        os.mkdir(config.LOG_DIR)
    if not os.path.exists(config.WEIGHTS_DIR):
        os.mkdir(config.WEIGHTS_DIR)

    if args.model == "hd_cnn":
        train_single_classifier(initial_epoch=0, epochs=50)
        train_coarse_classifier(initial_epoch=50, epochs=100)

        for i in range(config.COARSE_CLASS_NUM):
            initial_epoch = 0
            epochs = 50

            history = train_fine_classifier(
                initial_epoch=initial_epoch, epochs=epochs, class_idx=i
            )
            save_chart(
                history,
                os.path.join(
                    config.HD_CNN_DIR,
                    "data",
                    "fine_classifier_{}_train_chart.png".format(str(i)),
                ),
                initial_epoch,
                epochs,
            )
    elif args.model == "vgg16":  # Benchmark
        train_vgg16(
            initial_epoch=0,
            epochs=100,
            batch_size=config.BATCH_SIZE,
        )
    elif args.model == "vgg19":  # Benchmark
        train_vgg19(
            initial_epoch=0,
            epochs=100,
            batch_size=config.BATCH_SIZE,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    main(args)
