
PATH = "ResNet20.keras"
PATH_pruned = "ResNet20_2_4.keras"

import keras
import numpy as np

regular = keras.models.load_model(PATH)
pruned = keras.models.load_model(PATH_pruned)


regular.summary()
pruned.summary()


'''
Select only convolutional and dense layers because they are the only ones that can be pruned.
If layer is functional, we need to go into the functional layers and extract the underlying layers.
'''

def get_convolutional_and_dense_layers(model):
    layers = []
    for layer in model.layers:
        if isinstance(layer, (keras.layers.Conv2D, keras.layers.Dense)):
            layers.append(layer)
        elif isinstance(layer, keras.models.Model):
            layers.extend(get_convolutional_and_dense_layers(layer))
    return layers

regular_layers = get_convolutional_and_dense_layers(regular) # list of layers in the regular model
pruned_layers = get_convolutional_and_dense_layers(pruned) # list of layers in the pruned model


# extract corresponding weights from regular and pruned models
# (layer_name, weight_before_pruning, weight_after_pruning)

'''
Extract weights from the regular and pruned models.

regular_layers, pruned_layers -> (layer_name, weight_before_pruning, weight_after_pruning)
'''

def extract_weights_comparison(regular_layers: list[keras.layers.Layer] , pruned_layers : list[keras.layers.Layer]):
    weight_comparison = []

    if (len(regular_layers) != len(pruned_layers)):
        raise ValueError("The number of layers in the regular and pruned models must be the same.")
    
    for l1, l2 in zip(regular_layers, pruned_layers):
        if l1.name != l2.name:
            raise ValueError(f"Layer names do not match: {l1.name} != {l2.name}")
        
        weights_before = l1.get_weights()[0]
        weights_after = l2.get_weights()[0]
        
        weight_comparison.append((l1.name, weights_before, weights_after))

    return weight_comparison


weights_comparison = extract_weights_comparison(regular_layers, pruned_layers)



'''
Evaluate the masks for each layer. Boolean numpy array where:
- True means the weight was pruned (0 in pruned weights but not in regular weights)
- False means the weight was not pruned (non-zero in both regular and pruned weights)

(name, reg_weights, pruned_weights) -> (name, mask)

doubious weights are those that are 0 in both regular and pruned weights.
'''
name_mask = []

global doubious
doubious = 0

def evaluate_mask(reg : np.ndarray, pruned : np.ndarray):
    
    # !0 and 0 -> pruned
    # !0 and !0 -> not pruned
    # 0 and 0 -> don't know

    global doubious

    mask = (reg != 0) & (pruned == 0)

    # count the 0 and 0
    dub = (reg == 0) & (pruned == 0)
    count = np.sum(dub)
    doubious += count

    return mask

    
# (name, reg, prun) -> (name, mask)
for comp in weights_comparison:
    name, reg, prun = comp
    mask = evaluate_mask(reg, prun)
    name_mask.append((name, mask))


print(f"Doubious weights: {doubious}") # fortunately 0 for ResNet20


'''
Evaluate the lists of pruned cordinates and not pruned coordinates for each layer.
Computation of how mainy weights to inject in each layer based on the fault list dimension (FL_DIM).
Injected weights in a layer are proportional to its size compared to the total weights in the model.

(name, mask) -> (name, pruned_cordinates, not_pruned_coordinates, number_of_weight_to_inect)
'''

FL_DIM = 10000  # Fault List Dimension
TOTAL_WEIGHTS = sum(mask.size for _, mask in name_mask)

name_cordinates_toInject = []
for name, mask in name_mask:
    pruned_cordinates = np.argwhere(mask)
    not_pruned_coordinates = np.argwhere(~mask)

    if len(pruned_cordinates) + len(not_pruned_coordinates) != mask.size:
        raise ValueError(f"Mask size mismatch for layer {name}: {len(pruned_cordinates) + len(not_pruned_coordinates)} != {mask.size}")


    LAYER_TO_INJECT = int(FL_DIM * mask.size / TOTAL_WEIGHTS) + 1

    print(f"Layer: {name}, total weights {mask.size}, LAYER_TO_INJECT: {LAYER_TO_INJECT}")

    idx = np.random.choice(len(pruned_cordinates), size=LAYER_TO_INJECT, replace=False)
    pruned_cordinates = pruned_cordinates[idx]

    idx = np.random.choice(len(not_pruned_coordinates), size=LAYER_TO_INJECT, replace=False)
    not_pruned_coordinates = not_pruned_coordinates[idx]

    name_cordinates_toInject.append((name, pruned_cordinates, not_pruned_coordinates, LAYER_TO_INJECT))


'''
Generate the csv file of the fault lists for pruned and not pruned weights.
Each entry in the fault list is a tuple (Injection, Layer, TensorIndex, Bit).
'''

import csv

pruned_fl_file = open('pruned_fl.csv', 'w', newline='')
not_pruned_fl_file = open('not_pruned_fl.csv', 'w', newline='')

pruned_fl = csv.writer(pruned_fl_file)
not_pruned_fl = csv.writer(not_pruned_fl_file)


pruned_fl.writerow(['Injection', 'Layer', 'TensorIndex', 'Bit'])
not_pruned_fl.writerow(['Injection', 'Layer', 'TensorIndex', 'Bit'])

pruned_injection_idx = 0
not_pruned_injection_idx = 0
for name, pruned_cordinates, not_pruned_coordinates, INJECTED_IN_THIS_LAYER in name_cordinates_toInject:


    print(f"Name : {name}, Pruned indices: {len(pruned_cordinates)}, Not pruned indices: {len(not_pruned_coordinates)}")

    # add to pruned_fl a tuple (name, layer, tensor_index, bit)
    for injection in pruned_cordinates:
        bit = np.random.randint(0, 32)
        pruned_fl.writerow([pruned_injection_idx, name, str(tuple(injection)), bit])
        pruned_injection_idx += 1


    # add to not_pruned_fl a tuple (name, layer, tensor_index, bit)
    for injection in pruned_cordinates:
        bit = np.random.randint(0, 32)
        not_pruned_fl.writerow([not_pruned_injection_idx, name, str(tuple(injection)), bit])
        not_pruned_injection_idx += 1


pruned_fl_file.close()
not_pruned_fl_file.close()




# istogram to visualize the percentage of layer to inject
# plot names_cordinates_toInject

'''
import matplotlib.pyplot as plt
import numpy as np

# Crea grafico a barre per confrontare pesi totali e iniezioni
fig, ax = plt.subplots(figsize=(14, 8))

# Estrai i dati necessari
layer_names = []
total_weights = []
injected_weights = []

# Raccogli i dati per ogni layer
for name, mask in name_mask:
    layer_names.append(name)
    total_weights.append(mask.size)  # Numero totale di pesi nel layer

# Abbina le informazioni sulle iniezioni per ogni layer
for name, _, _, inj_count in name_cordinates_toInject:
    # Trova l'indice corrispondente in layer_names
    idx = layer_names.index(name)
    injected_weights.append(inj_count)

# Posizione delle barre
x = np.arange(len(layer_names))
bar_width = 0.8

# Imposta la scala logaritmica sull'asse Y
ax.set_yscale('log')

# Crea le barre
total_bars = ax.bar(x, total_weights, bar_width, color='lightgray', alpha=0.7, label='Total weights')
inject_bars = ax.bar(x, injected_weights, bar_width, color='steelblue', label='Injected weights')

# Aggiungi etichette con i valori (adattate per scala logaritmica)
for i, (total, injected) in enumerate(zip(total_weights, injected_weights)):
    percentage = (injected / total) * 100
    
    # Posiziona le etichette in modo appropriato sulla scala logaritmica
    log_injected = np.log10(injected) if injected > 0 else 0
    log_total = np.log10(total)
    
    # Etichetta per i pesi iniettati
    ax.text(i, injected * 1.1, f'{injected}\n({percentage:.1f}%)', 
            ha='center', va='bottom', color='black', fontweight='bold')
    
    # Etichetta per i pesi totali
    ax.text(i, total * 1.1, f'{total}', 
            ha='center', va='bottom', color='black')

# Personalizza il grafico
ax.set_title('Total weights vs injected (log scale)', fontsize=16)
ax.set_xlabel('Layer', fontsize=14)
ax.set_ylabel('Weights (log scale)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(layer_names, rotation=45, ha='right')
ax.legend()

# Aggiungi griglia per facilitare la lettura della scala logaritmica
ax.grid(True, which="both", ls="-", alpha=0.2)

plt.tight_layout()
plt.savefig('injection_comparison_log.png', dpi=300, bbox_inches='tight')
plt.show()
'''