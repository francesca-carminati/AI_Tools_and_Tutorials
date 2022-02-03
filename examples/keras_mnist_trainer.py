import neptune.new as neptune
import tensorflow as tf
from decouple import config
from neptune.new.integrations.tensorflow_keras import NeptuneCallback

NUM_LABELS = 10

BATCH_SIZE = config("BATCH_SIZE", cast=int, default=1)
MODEL_DIR = config("MODEL_DIR", default="/workspace/model")
DATA_DIR = config("DATA_DIR", default="/workspace/data")
CACHE_DIR = config("CACHE_DIR", default="./cache/huggingface")
LOG_DIR = config("LOG_DIR", default="/workspace/logs")

# Neptune API
NEPTUNE_API_TOKEN = config("NEPTUNE_API_TOKEN")
NEPTUNE_PROJECT = config("NEPTUNE_PROJECT")

run = neptune.init(
    api_token=NEPTUNE_API_TOKEN,
    project=NEPTUNE_PROJECT,
    name="keras-mnist-classifier",
    description="Very simple keras classifer",
    tags=["keras", "image", "MNIST_dataset"],
)


if __name__ == "__main__":
    neptune_logger = NeptuneCallback(run=run, base_namespace="metrics")

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation=tf.keras.activations.relu),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax),
        ]
    )

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.4,)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.fit(x_train, y_train, epochs=5, batch_size=BATCH_SIZE, callbacks=[neptune_logger])
    print(model.summary())

    model.save(f"{MODEL_DIR}/keras_MNIST_classifier")
    run.stop()
