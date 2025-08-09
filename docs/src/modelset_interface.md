```@meta
CurrentModule = SoleXplorer
```

```@contents
Pages = ["modelset_interface.md"]
Depth = 3
```

# Model Interface

This module defines the core types and functionality for machine learning models in the SoleXplorer framework, with a focus on model configuration, training, and rule extraction for interpretable machine learning.

## Type Aliases

```@docs
Cat_Value
Reg_Value
Y_Value
TypeTreeForestC
TypeTreeForestR
TypeModalForest
```

## Model Setup

The `ModelSetup` structure defines the configuration for machine learning models.

```@docs
ModelSetup
```

### Accessor Functions

```@docs
get_config
get_params
get_features
get_winparams
get_tuning
get_resample
get_preprocess
get_rulesparams
get_pfeatures
get_treatment
get_algo
get_rawmodel
get_resampled_rawmodel
get_learn_method
get_resampled_learn_method
```

### Setter Functions

```@docs
set_config!
set_params!
set_features!
set_winparams!
set_tuning!
set_resample!
set_rulesparams!
set_rawmodel!
set_learn_method!
```

## Model Types

```@docs
TypeDTC
TypeRFC
TypeABC
TypeDTR
TypeRFR
TypeMDT
TypeMRF
TypeMAB
TypeXGC
TypeXGR
```

## Default Parameters

```@docs
DEFAULT_MODEL_SETUP
DEFAULT_FEATS
DEFAULT_PREPROC
PREPROC_KEYS
AVAIL_MODELS
```

## Results Structures

```@docs
ClassResults
RegResults
RESULTS
```

## Modelset Structure

The `Modelset` structure serves as the primary container for machine learning models in the framework.

```@docs
Modelset
```

## Usage Examples

Here are some common usage patterns for the model interface:

### Creating a Simple Model

```julia
using SoleXplorer, MLJ

# Load data
X, y = @load_iris
ds = Dataset(X, y)

# Create a decision tree model setup
dt_setup = ModelSetup{TypeDTC}(
    decisiontree,
    (treatment=:none, algo=:classification),
    (max_depth=5,),
    nothing,  # No feature extraction
    nothing,  # No resampling
    WinParams(),
    MLJ.DecisionTreeClassifier,
    MLJ.fit!,
    false,  # No tuning
    true,   # Extract rules with default params
    (train_ratio=0.8, valid_ratio=0.2, rng=123)
)

# Create a modelset
modelset = Modelset(dt_setup, ds)

# Train the model
train!(modelset)

# Extract rules
rules_extraction!(modelset)

# Print the extracted rules
println(modelset.rules)

# Calculate and print accuracy
accuracy = get_accuracy(modelset)
println("Model accuracy: ", accuracy)
```

### With Hyperparameter Tuning

```julia
# Create a model with tuning
rf_setup = ModelSetup{TypeRFC}(
    randomforest,
    (treatment=:none, algo=:classification),
    (n_trees=100,),
    nothing,  # No feature extraction
    Resample(CV(), 5),  # 5-fold cross-validation
    WinParams(),
    MLJ.RandomForestClassifier,
    MLJ.fit!,
    TuningParams(  # Custom tuning configuration
        TuningStrategy(grid, (resolution=10,)),
        (n=20, measure=accuracy),
        (range(:n_trees, values=[50, 100, 150]),
         range(:max_depth, values=[3, 5, 7]))
    ),
    RulesParams(:intrees, (prune_rules=true,)),
    (train_ratio=0.7, valid_ratio=0.3, rng=42)
)

# Create and train the model
tuned_modelset = Modelset(rf_setup, ds)
train!(tuned_modelset)

# Show best parameters
println("Best parameters: ", get_best_params(tuned_modelset))
```

## See Also

- [`tuning`](@ref) - Hyperparameter tuning capabilities
- [`extractrules_interface`](@ref) - Rule extraction methods
- [`dataset_interface`](@ref) - Dataset handling