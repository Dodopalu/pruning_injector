model = tf.keras.models.load_model("./models/ResNet20.keras")
model.save("./models/saved_model")