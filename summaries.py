import tensorflow as tf

path = "./models/flatten_Vgg11_bn.keras"
model = tf.keras.models.load_model(path)
model.summary()
