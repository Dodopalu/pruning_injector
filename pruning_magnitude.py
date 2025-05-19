from tensorflow import keras
from loaderCIFAR10 import load
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tempfile

def sparse_pruning(PATH : str, OUTPUT_DIR : str, pruned_file_name : str, initial_sparsity : float, final_sparsity : float, begin_step : int, end_step :int):

    # load model
    model = keras.models.load_model(PATH)
    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
        )

    # pruning
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=initial_sparsity,
            final_sparsity=final_sparsity,
            begin_step=begin_step,
            end_step=end_step
        )
    }

    def apply_pruning_recursively(layer):
        """Apply pruning into functional layers."""
        # Go into functional layers
        if isinstance(layer, tf.keras.Model):
            print(f"Examining Functional layer: {layer.name}")
            return tf.keras.models.clone_model(
                layer,
                clone_function=apply_pruning_recursively
            )
        # Prune
        elif isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
            print(f"Applying pruning to: {layer.name}")
            return prune_low_magnitude(layer, **pruning_params)
        return layer


    pruned_model = tf.keras.models.clone_model(
        model,
        clone_function=apply_pruning_recursively
    )

    # train post-pruning
    pruned_model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    train_dataset, test_dataset = load()

    train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

    logdir = tempfile.mkdtemp()

    history = pruned_model.fit(
        train_dataset,
        epochs=2,
        validation_data=test_dataset,
        callbacks=[tfmot.sparsity.keras.UpdatePruningStep(),
                tfmot.sparsity.keras.PruningSummaries(log_dir=logdir)],
        verbose=1
    )

    final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)

    # save
    import os
    import datetime


    os.makedirs(OUTPUT_DIR, exist_ok=True) 

    # File name with timestamp
    model_name = os.path.basename(PATH).split('.')[0]

    # Keras con sparsity aware option
    sparse_keras_file = os.path.join(OUTPUT_DIR, f"{model_name}_{pruned_file_name}.keras")
    final_model.save(sparse_keras_file, save_format='keras')
    print('Saving sparse-optimized model to:', sparse_keras_file)

    NEW_PATH = os.path.join(OUTPUT_DIR, f"{model_name}_{pruned_file_name}.keras")
    return NEW_PATH


if __name__ == "__main__":

    OUTPUT_DIR = "/Users/domenicopalumbo/keras_weight_injector_data/models/CIFAR10"

    PATH = "/Users/domenicopalumbo/keras_weight_injector_data/models/CIFAR10/ResNet20.keras"

    #sparse_pruning(
    #    PATH=PATH, 
    #    OUTPUT_DIR=OUTPUT_DIR, 
    #    pruned_file_name="sparse_50ptc",
    #    initial_sparsity=0, 
    #    final_sparsity=0.5, 
    #    begin_step=0, 
    #    end_step=1000)