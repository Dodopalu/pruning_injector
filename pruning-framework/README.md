
# pruning.py
Contains two distinct pruning methods:
- Magnitude Pruning
- Structural Pruning (N:M Sparsity)

To use these functions, modify the `PATH` variable in the main section and call your preferred pruning method.

### Structural Pruning
Implements N:M structured sparsity patterns. For example, a (2,4) sparsity pattern means weights are divided into groups of 4, and the 2 weights with the lowest magnitudes in each group are set to zero.
```
structural_pruning(
        PATH=PATH, 
        OUTPUT_DIR=OUTPUT_DIR, 
        pruned_file_name="structural_2_4", # change name of the pruned model
        sparsity=(2, 4), # change this to the desired sparsity
        test_dataset=test_dataset,
        train_dataset=train_dataset
        )
```


### Magnitude Pruning
Implements gradual global pruning based on weight magnitudes. A sparsity of 0.5 means that the 50% of weights with the lowest absolute values are set to zero, while the top 50% remain unchanged. The pruning process happens gradually from `begin_step` to `end_step`.
```
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
```


# generate_fl.py
For fault list generation, specify the paths to your models:

- Set `PATH` to the path of your original (non-pruned) model
- Set `PATH_pruned` to the path of your pruned model

### Output Files
The program generates two CSV output files:

1. `pruned_fl.csv` : Contains 10,000 fault injections targeting weights that were eliminated during the pruning process.

2. `not_pruned_fl.csv` : Contains 10,000 fault injections targeting weights that were preserved (not pruned) in the model.



