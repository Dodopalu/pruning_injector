from pruning_magnitude import sparse_pruning
from pruning_structural import structural_pruning
from benchmark_tfLite import test
from convert_in_tfLite import convert_to_tfLite


OUTPUT_DIR = "/Users/domenicopalumbo/keras_weight_injector_data/models/CIFAR10"
BASE_MODEL = "/Users/domenicopalumbo/keras_weight_injector_data/models/CIFAR10/ResNet20.keras"


MAGNITUDE_50 = sparse_pruning(BASE_MODEL, OUTPUT_DIR, "sparse_50pct",
                              initial_sparsity=0.0, final_sparsity=0.8,
                              begin_step=0, end_step=1000
                              )

MAGNITUDE_80 = sparse_pruning(BASE_MODEL, OUTPUT_DIR, "sparse_80pct",
                              initial_sparsity=0.8, final_sparsity=0.8,
                              begin_step=0, end_step=1000
                              )

STRUCTURAL_2_4 = structural_pruning(BASE_MODEL, OUTPUT_DIR, "structural_2_4", (2, 4))
STRUCTURAL_5_7 = structural_pruning(BASE_MODEL, OUTPUT_DIR, "structural_5_7", (5, 7))


MAGNITUDE_50 = convert_to_tfLite(MAGNITUDE_50, OUTPUT_DIR)
TF_MAGNITUDE_80 = convert_to_tfLite(MAGNITUDE_80, OUTPUT_DIR)
TF_STRUCTURAL_2_4 = convert_to_tfLite(STRUCTURAL_2_4, OUTPUT_DIR)
TF_STRUCTURAL_5_7 = convert_to_tfLite(STRUCTURAL_5_7, OUTPUT_DIR)


#test(TF_MAGNITUDE_80, OUTPUT_DIR)
#test(TF_STRUCTURAL_5_7, OUTPUT_DIR)