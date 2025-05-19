import keras

path = "./models/flatten_Vgg11_bn.keras"
model = keras.models.load_model(path)
model.summary()
