# Verifica del supporto hardware per operazioni sparse
import tensorflow as tf
import numpy as np
import os
import time

def check_sparse_acceleration():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        gpu_details = tf.config.experimental.get_device_details(physical_devices[0])
        print(f"GPU details: {gpu_details}")
        
    # Random sparsity test
    import time
    import numpy as np
    
    # Dense matrix
    dense_mat = np.random.rand(1000, 1000).astype(np.float64)
    dense_tensor = tf.constant(dense_mat)
    
    # Sparse 0.5 density matrix
    sparse_mat = np.random.rand(1000, 1000).astype(np.float64)
    mask = np.random.choice([0, 1], size=(1000, 1000), p=[0.5, 0.5])
    sparse_mat = sparse_mat * mask
    sparse_tensor = tf.sparse.from_dense(sparse_mat)
    
    # Benchmark
    start = time.time()
    for _ in range(100):
        _ = tf.matmul(dense_tensor, dense_tensor)
    dense_time = time.time() - start
    
    start = time.time()
    for _ in range(100):
        _ = tf.sparse.sparse_dense_matmul(
            tf.sparse.reorder(sparse_tensor), 
            tf.transpose(dense_tensor)
        )
    sparse_time = time.time() - start
    
    print(f"Dense execution time: {dense_time:.4f}s")
    print(f"Sparse execution time: {sparse_time:.4f}s")
    print(f"Speedup: {dense_time/sparse_time:.2f}x")



check_sparse_acceleration()