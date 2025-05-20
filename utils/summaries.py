import keras

path = "./models/flatten_Vgg11_bn.keras"
model = keras.models.load_model(path, compile=False)
model.summary()