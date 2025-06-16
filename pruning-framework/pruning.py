from tensorflow import keras
from loaderCIFAR10 import load
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tempfile


def apply_pruning_recursively(layer : tf.keras.layers.Layer):
    """
    Apply pruning into functional layers. 

    functional layer contains other layers, so when we encounter a functional layer,
    we recursively apply pruning to its sub-layers.
    """
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
        return tfmot.sparsity.keras.prune_low_magnitude(layer)
    return layer

def structural_pruning(
        PATH : str, 
        OUTPUT_DIR : str, 
        pruned_file_name : str, 
        sparsity : tuple[int, int], 
        test_dataset, 
        train_dataset
        ):
    
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
        'sparsity_m_by_n': sparsity,
    }

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

    #train_dataset, test_dataset = load()
    #train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    #test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

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


    # Save the model .keras
    import os

    model_name = os.path.basename(PATH).split('.')[0] 


    sparse_keras_file = os.path.join(OUTPUT_DIR, f"{model_name}_{pruned_file_name}.keras")
    final_model.save(sparse_keras_file, save_format='keras')
    print('Saving sparse-optimized model to:', sparse_keras_file)

    NEW_PATH = os.path.join(OUTPUT_DIR, f"{model_name}_{pruned_file_name}.keras")
    return NEW_PATH


def sparse_pruning(
        PATH : str, 
        OUTPUT_DIR : str, 
        pruned_file_name : str, 
        initial_sparsity : float, 
        final_sparsity : float, 
        begin_step : int, 
        end_step :int, 
        train_dataset, 
        test_dataset
        ):

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

    #train_dataset, test_dataset = load()
    #train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    #test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

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

    PATH = "ResNet20.keras" # input model path
    OUTPUT_DIR = "./"

    # load dataset to prune after pruning
    train_dataset, test_dataset = load()
    train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)



    # Structural pruining example
    structural_pruning(
        PATH=PATH, 
        OUTPUT_DIR=OUTPUT_DIR, 
        pruned_file_name="structural_2_4", # change name of the pruned model
        sparsity=(2, 4), # change this to the desired sparsity
        test_dataset=test_dataset,
        train_dataset=train_dataset
        )
    
    # Magnitude pruning example
    sparse_pruning(
        PATH=PATH, 
        OUTPUT_DIR=OUTPUT_DIR, 
        pruned_file_name="sparse_0_5", # change name of the pruned model
        initial_sparsity=0.0, # initial sparsity
        final_sparsity=0.5, # final sparsity
        begin_step=0, # begin step
        end_step=1000, # end step
        train_dataset=train_dataset,
        test_dataset=test_dataset
    )
    
    
