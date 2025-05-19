import tensorflow as tf

path = "./models/CIFAR10/googlenet/GoogLeNet.keras"
model = tf.keras.models.load_model(path)
model.summary()
