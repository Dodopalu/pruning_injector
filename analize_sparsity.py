import tensorflow as tf
import numpy as np

def analyze_tflite_sparsity(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    tensor_details = interpreter.get_tensor_details()
    
    sparse_tensors = 0
    dense_tensors = 0
    total_params = 0
    zero_params = 0
    
    tensors_with_sparse_format = 0
    
    for tensor in tensor_details:
        # Verifica se esiste il parametro di sparsità
        has_sparse_format = False
        if 'sparsity_parameters' in tensor:
            sparse_params = tensor['sparsity_parameters']
            if sparse_params is not None:
                tensors_with_sparse_format += 1
                has_sparse_format = True
                print(f"\nTensor con formato sparso: {tensor['name']}")
                
                # Analizza il formato sparso
                if 'dim_metadata' in sparse_params:
                    for i, dim in enumerate(sparse_params['dim_metadata']):
                        if 'format' in dim:
                            format_type = "DENSO" if dim['format'] == 0 else "SPARSO"
                            print(f"  Dimensione {i}: {format_type}")
                
                # Se hai array_segments e array_indices, è un formato CSR
                for dim in sparse_params.get('dim_metadata', []):
                    if 'array_segments' in dim and 'array_indices' in dim:
                        print("  Formato: CSR (Compressed Sparse Row)")
                        print(f"  Segmenti: {len(dim['array_segments'])}")
                        print(f"  Indici: {len(dim['array_indices'])}")
                        print(f"  Compressione stimata: {len(dim['array_indices'])/tensor['shape'].prod():.4f}")
                        break

        # Verifica se è un tensore di pesi
        if tensor['name'].endswith('weight') or 'kernel' in tensor['name']:
            tensor_data = interpreter.get_tensor(tensor['index'])
            
            # Calcola sparsità
            n_zeros = np.sum(tensor_data == 0)
            n_elements = tensor_data.size
            sparsity = n_zeros / n_elements
            
            total_params += n_elements
            zero_params += n_zeros
            
            print(f"Tensor: {tensor['name']}")
            print(f"  Shape: {tensor_data.shape}")
            print(f"  Sparsity: {sparsity:.4f} ({n_zeros} zeros out of {n_elements})")
            print(f"  Formato sparso codificato: {has_sparse_format}")
            
            if sparsity > 0.2:  # Consideriamo sparso se ha più del 20% di zeri
                sparse_tensors += 1
            else:
                dense_tensors += 1
    
    overall_sparsity = zero_params / total_params if total_params > 0 else 0
    print(f"\nRiepilogo:")
    print(f"  Tensori sparsi (in base al conteggio degli zeri): {sparse_tensors}")
    print(f"  Tensori densi: {dense_tensors}")
    print(f"  Tensori con formato di sparsità codificato: {tensors_with_sparse_format}")
    print(f"  Sparsità complessiva: {overall_sparsity:.4f}")

    # Verifica le operazioni
    print("\nVerifica se il modello contiene operazioni sparse:")
    try:
        # Prova ad accedere al buffer del modello per analisi più dettagliate
        with open(model_path, 'rb') as f:
            model_content = f.read()
        
        # Cerca stringhe che suggeriscano operazioni sparse
        sparse_ops = ["SparseToDense", "SparseMatMul", "CSR", "Sparse"]
        for op in sparse_ops:
            if op.encode() in model_content:
                print(f"  Trovato riferimento a operazione sparsa: {op}")
    except Exception as e:
        print(f"  Errore nell'analisi delle operazioni: {e}")

# Analizza il modello prunato
pruned_model_path = "/Users/domenicopalumbo/keras_weight_injector_data/models/CIFAR10/ResNet20_pruned_50pct_20250514_185607.tflite"
print("Analisi del modello prunato:")
analyze_tflite_sparsity(pruned_model_path)