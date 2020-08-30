import tensorflow as tf
import numpy as np
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation
import time

def swish(x, beta = 1):
    return (x * sigmoid(beta * x))

def process_dataset():
    # Import the data
    (x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Reshape the data
    NUM_TRAIN = 60000
    NUM_TEST = 10000
    x_train = np.reshape(x_train, (NUM_TRAIN, 28, 28, 1))
    x_test = np.reshape(x_test, (NUM_TEST, 28, 28, 1))
    return x_train, y_train, x_test, y_test

def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=[28,28, 1]))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation="swish"))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def save(model, filename):
    # First freeze the graph and remove training nodes.
    output_names = model.output.op.name
    sess = tf.keras.backend.get_session()
    frozen_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), [output_names])
    frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)
    # Save the model
    with open(filename, "wb") as ofile:
        ofile.write(frozen_graph.SerializeToString())

def main():
    get_custom_objects().update({'swish': Activation(swish)})
    x_train, y_train, x_test, y_test = process_dataset()

    model = create_model()
    # Train the model on the data
    model.fit(x_train, y_train, epochs = 5, verbose = 1)
    # Evaluate the model on test data
    model.evaluate(x_test, y_test)

    inference_start_time = time.time()
    for i in range(1000):
        pred_confidence_level = model.predict(np.expand_dims(x_test[0], axis=0))

    end_time = time.time()
    print("Time taken for one input without tensorrt", (end_time-inference_start_time)/1000)
    #save(model, filename="./models/lenet5.pb")

if __name__ == '__main__':
    main()
