```@meta
CurrentModule = SoleXplorer
```
```@contents
Pages = ["dataset_interface.md"]
```

# [Dataset Interface](@id dataset-interface)

This module provides structures and interfaces for dataset configuration, preparation, and splitting in machine learning workflows.

## [Overview](@id dataset-overview)

The dataset interface consists of several key components:
- `DatasetInfo`: Configuration for dataset preparation
- `TT_indexes`: Train-validation-test split indices
- `Dataset`: The main structure that holds data splits and views

## [DatasetInfo Structure](@id datasetinfo-structure)

`DatasetInfo` encapsulates parameters for data preprocessing, train-validation-test splitting, and random state management.

### [Fields](@id datasetinfo-fields)

| Field | Type | Description |
|-------|------|-------------|
| `algo` | `Symbol` | Algorithm to use for dataset processing |
| `treatment` | `Symbol` | Data treatment method (e.g., `:standardize`, `:normalize`) |
| `modalreduce` | `Union{<:Base.Callable, Nothing}` | Optional function for dimensionality reduction |
| `train_ratio` | `Real` | Proportion of data to use for training (0-1) |
| `valid_ratio` | `Real` | Proportion of data to use for validation (0-1) |
| `rng` | `AbstractRNG` | Random number generator for reproducible splits |
| `resample` | `Bool` | Whether to perform resampling for cross-validation |
| `vnames` | `Union{Vector{<:AbstractString}, Nothing}` | Optional feature/variable names |

### [Constructor](@id datasetinfo-constructor)

```julia
DatasetInfo(
    algo::Symbol, 
    treatment::Symbol,
    modalreduce::Union{<:Base.Callable, Nothing},
    train_ratio::Real,
    valid_ratio::Real,
    rng::AbstractRNG,
    resample::Bool,
    vnames::Union{Vector{<:AbstractString}, Nothing}
)
```

Creates a new `DatasetInfo` instance with the specified parameters. Validates that `train_ratio` and `valid_ratio` are between 0 and 1.

### [Methods](@id datasetinfo-methods)

- `get_resample(dsinfo::DatasetInfo)::Bool`: Returns the `resample` field value
- `Base.show(io::IO, info::DatasetInfo)`: Pretty-prints the structure

## [Usage Example](@id datasetinfo-usage)

```julia
using SoleXplorer
using Random

# Create a basic dataset configuration
ds_info = DatasetInfo(
    :classification,     # algorithm
    :standardize,        # treatment
    nothing,             # no dimensionality reduction
    0.7,                 # train ratio
    0.15,                # validation ratio
    MersenneTwister(42), # random seed
    false,               # no resampling
    ["feature1", "feature2", "feature3"]  # variable names
)

# Use with Dataset constructor
X = rand(100, 3)  # Example feature matrix
y = rand(100)     # Example target vector
dataset = Dataset(X, y, ds_info)
```

This configuration can be used with the `Dataset` constructor to prepare data for machine learning training workflows.

## [Related Structures](@id related-structures)

- [`TT_indexes`](@ref): Stores train-validation-test split indices
- [`Dataset`](@ref): Main structure that holds data and split information

```@docs
DatasetInfo
get_resample
```