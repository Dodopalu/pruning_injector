import numpy as np
import csv
import keras


file = open('result_example.csv', 'r')
reader = csv.reader(file)

def get_convolutional_and_dense_layers(model):
    layers = []
    for layer in model.layers:
        if isinstance(layer, (keras.layers.Conv2D, keras.layers.Dense)):
            layers.append(layer)
        elif isinstance(layer, keras.models.Model):
            layers.extend(get_convolutional_and_dense_layers(layer))
    return layers

# layer -> list(cords, criticals)
layer_cords = {}
for row in reader:

    # skip header
    if row[1] == "target_layer" or row[0] == "GOLDEN":
        continue
    
    layer = row[1]
    if layer not in layer_cords:
        layer_cords[layer] = []

    cords = np.array(eval(row[2])) # "(1,2,3,4)" -> [1,2,3,4]
    critical = int(row[11])
    layer_cords[layer].append((cords, critical))


# layer -> shape
model = keras.models.load_model("ResNet20.keras")

layers = get_convolutional_and_dense_layers(model)
layer_shape = {}

for layer in layers:
    if isinstance(layer, keras.layers.Conv2D) or isinstance(layer, keras.layers.Dense):
        layer_shape[layer.name] = layer.get_weights()[0].shape

# Create heatmaps
import matplotlib.pyplot as plt

# shape = (x, y, z, w) 
# plot the heatmap using the last two dimensions (z, w)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_heatmap(layer_name):
    """
    Genera una heatmap di criticità per un layer specifico.
    
    Args:
        layer_name: Nome del layer per cui generare la heatmap
    
    Returns:
        True se la heatmap è stata generata correttamente, False altrimenti
    """
    if layer_name not in layer_cords.keys():
        print(f"Errore: Layer '{layer_name}' non trovato nei dati di iniezione")
        return False
        
    if layer_name not in layer_shape.keys():
        print(f"Errore: Layer '{layer_name}' non trovato nel modello")
        return False
    
    shape = layer_shape[layer_name]
    coordinates_list = layer_cords[layer_name]
    
    print(f"Generazione heatmap per {layer_name} con shape {shape}")
    
    # Ottieni le dimensioni delle ultime due coordinate
    if len(shape) == 4:  # Conv2D
        z_dim, w_dim = shape[2], shape[3]
    elif len(shape) == 2:  # Dense
        z_dim, w_dim = shape[0], shape[1]
    else:
        print(f"Shape non supportata per il layer {layer_name}: {shape}")
        return False
    
    # Inizializza le matrici per contare eventi critici e totali
    critical_sum = np.zeros((z_dim, w_dim))
    injection_count = np.zeros((z_dim, w_dim))
    
    # Aggrega i dati per le coordinate (z, w)
    for coords, critical in coordinates_list:
        # Ora critical è già un valore numerico che indica quante classificazioni errate ha causato
        
        # Prendi le ultime due coordinate
        if len(coords) == 4:  # Conv2D
            z, w = coords[2], coords[3]
        elif len(coords) == 2:  # Dense
            z, w = coords[0], coords[1]
        else:
            continue
        
        # Incrementa contatori se le coordinate sono valide
        if 0 <= z < z_dim and 0 <= w < w_dim:
            critical_sum[z, w] += critical  # Somma il valore di criticità
            injection_count[z, w] += 1      # Conta l'iniezione
    
    # Calcola il tasso medio di criticità (somma criticità / numero iniezioni)
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_criticality = np.divide(critical_sum, injection_count)
    
    # Sostituisci NaN con 0 per celle senza iniezioni
    avg_criticality = np.nan_to_num(avg_criticality, nan=0.0)
    
    # Trova il valore massimo per la scala della heatmap
    max_criticality = np.max(avg_criticality)
    if max_criticality == 0:
        max_criticality = 1.0  # Imposta un valore predefinito se non ci sono criticità
    
    # Crea la heatmap
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(avg_criticality, cmap="YlOrRd", annot=False, 
                vmin=0.0, vmax=max_criticality,  # Scala adattata al massimo valore
                cbar_kws={'label': f'Tasso medio di criticità (0-{max_criticality:.1f})'})
    
    plt.title(f'Heatmap di criticità per il layer {layer_name}')
    plt.xlabel('Dimensione W')
    plt.ylabel('Dimensione Z')
    
    # Aggiungi informazioni sul numero di iniezioni
    total_injections = np.sum(injection_count)
    total_critical_sum = np.sum(critical_sum)
    
    # Calcola il tasso medio di criticità complessivo
    avg_critical_rate = 0.0 if total_injections == 0 else (total_critical_sum/total_injections)
    
    plt.figtext(0.5, 0.01, 
                f'Totale iniezioni: {total_injections}, Somma criticità: {total_critical_sum}, '
                f'Tasso medio: {avg_critical_rate:.2f} classificazioni errate per iniezione',
                ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'heatmap_{layer_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap generata per il layer {layer_name}")
    return True

generate_heatmap("conv2d_11")
