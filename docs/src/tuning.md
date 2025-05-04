```@meta
CurrentModule = SoleXplorer
```

```@contents
Pages = ["tuning.md"]
Depth = 3
```

# Hyperparameter Tuning

This module provides comprehensive hyperparameter tuning capabilities for machine learning models, based on MLJ's tuning infrastructure.

## Overview

Hyperparameter tuning is essential for maximizing model performance. This package offers several strategies ranging from simple grid search to advanced optimization methods like Particle Swarm Optimization.

## Types

```@docs
TuningStrategy
TuningParams
```

## Available Tuning Methods

The following tuning methods are supported:

| Method | Description |
|--------|-------------|
| `grid` | Grid search over discrete parameter values |
| `randomsearch` | Random sampling from parameter distributions |
| `latinhypercube` | Latin Hypercube sampling for more efficient exploration |
| `treeparzen` | Tree-structured Parzen Estimator for adaptive parameter search |
| `particleswarm` | Particle Swarm Optimization for global parameter optimization |
| `adaptiveparticleswarm` | Adaptive Particle Swarm with dynamic adjustment strategies |

## Default Parameters

Each tuning method comes with sensible default parameters:

```julia
const TUNING_METHODS_PARAMS = Dict{Union{DataType, UnionAll},NamedTuple}(
    grid => (
        goal = nothing,
        resolution = 10,
        shuffle = true,
        rng = TaskLocalRNG()
    ),
    randomsearch => (
        rng = TaskLocalRNG()
    ),
    latinhypercube => (
        rng = TaskLocalRNG()
    ),
    treeparzen => (
        max_iterations = 20,
        gamma = nothing,
        rng = TaskLocalRNG()
    ),
    particleswarm => (
        n_particles = 3,
        w = 1.0,
        c1 = 2.0,
        c2 = 2.0,
        prob_shift = 0.25,
        rng = TaskLocalRNG()
    ),
    adaptiveparticleswarm => (
        n_particles = 3,
        c1 = 2.0,
        c2 = 2.0,
        prob_shift = 0.25,
        rng = TaskLocalRNG()
    )
)
```

## Default Task Settings

```julia
const TUNING_PARAMS = Dict{Symbol,NamedTuple}(
    :classification => (
        n = 25,
        resampling = CV(nfolds=5),
        measure = MLJ.LogLoss(),
        train_best = true,
        acceleration = CPU1()
    ),
    :regression => (
        n = 25,
        resampling = CV(nfolds=5),
        measure = MLJ.RootMeanSquaredError(),
        train_best = true,
        acceleration = CPU1()
    )
)
```

## Range Function

```@docs
range(field::Union{Expr, Symbol}; kwargs...)
```

### Examples

```julia
# Create a range for max_depth from 2 to 10
depth_range = range(:max_depth, lower=2, upper=10)

# Create a range for min_samples_leaf with specific values
leaf_range = range(:min_samples_leaf, values=[1, 5, 10, 20])

# Create a log-scale range for learning rate
lr_range = range(:learning_rate, lower=1e-4, upper=0.1, scale=:log)
```

## Usage Examples

### Basic Usage

```julia
using SoleXplorer, MLJ

# Load data
X, y = @load_iris

# Simple random search tuning
model = symbolic_analysis(
    X, y;
    model=(type=:decisiontree, params=(;)),
    tuning=(
        method=(type=randomsearch, params=(;rng=123)),
        params=(n=20,),
        ranges=(
            range(:max_depth, lower=2, upper=10),
            range(:min_samples_leaf, values=[1, 5, 10])
        )
    )
)
```

### Advanced Usage with Particle Swarm

```julia
# Tune a random forest with PSO
modelset = symbolic_analysis(
    X, y;
    model=(type=:randomforest, params=(;n_trees=100)),
    tuning=(
        method=(type=particleswarm, params=(n_particles=10,)),
        params=(n=30, resampling=CV(nfolds=5)),
        ranges=(
            range(:max_depth, lower=3, upper=15),
            range(:min_samples_split, lower=2, upper=20)
        )
    )
)

# Show tuning results
best_params = get_best_params(modelset)
println("Best parameters: ", best_params)
println("Best score: ", get_best_score(modelset))
```

## Parameter Range Tips

When defining parameter ranges, consider:

- Use `scale=:log` for parameters that vary by orders of magnitude (learning rates, regularization)
- Use `values` for categorical or discrete parameters
- For parameters with a natural scale, use `lower` and `upper` with appropriate bounds

```julia
# Examples of well-defined ranges
learning_rate_range = range(:eta, lower=1e-5, upper=0.1, scale=:log)
depth_range = range(:max_depth, values=[3, 5, 7, 9])
regularization_range = range(:lambda, lower=1e-3, upper=10.0, scale=:log)
```

## See Also

- [`MLJTuning`](https://alan-turing-institute.github.io/MLJ.jl/dev/tuning_models/) - MLJ's tuning functionality
- [`MLJParticleSwarmOptimization.jl`](https://github.com/JuliaAI/MLJParticleSwarmOptimization.jl) - Particle Swarm Optimization for MLJ
- [`symbolic_analysis`](@ref) - Main analysis function using the tuning interface
```<!-- filepath: /home/paso/Documents/Aclai/PasoStudio73/SoleXplorer/docs/src/tuning.md -->
```@meta
CurrentModule = SoleXplorer
```

```@contents
Pages = ["tuning.md"]
Depth = 3
```

# Hyperparameter Tuning

This module provides comprehensive hyperparameter tuning capabilities for machine learning models, based on MLJ's tuning infrastructure.

## Overview

Hyperparameter tuning is essential for maximizing model performance. This package offers several strategies ranging from simple grid search to advanced optimization methods like Particle Swarm Optimization.

## Types

```@docs
TuningStrategy
TuningParams
```

## Available Tuning Methods

The following tuning methods are supported:

| Method | Description |
|--------|-------------|
| `grid` | Grid search over discrete parameter values |
| `randomsearch` | Random sampling from parameter distributions |
| `latinhypercube` | Latin Hypercube sampling for more efficient exploration |
| `treeparzen` | Tree-structured Parzen Estimator for adaptive parameter search |
| `particleswarm` | Particle Swarm Optimization for global parameter optimization |
| `adaptiveparticleswarm` | Adaptive Particle Swarm with dynamic adjustment strategies |

## Default Parameters

Each tuning method comes with sensible default parameters:

```julia
const TUNING_METHODS_PARAMS = Dict{Union{DataType, UnionAll},NamedTuple}(
    grid => (
        goal = nothing,
        resolution = 10,
        shuffle = true,
        rng = TaskLocalRNG()
    ),
    randomsearch => (
        rng = TaskLocalRNG()
    ),
    latinhypercube => (
        rng = TaskLocalRNG()
    ),
    treeparzen => (
        max_iterations = 20,
        gamma = nothing,
        rng = TaskLocalRNG()
    ),
    particleswarm => (
        n_particles = 3,
        w = 1.0,
        c1 = 2.0,
        c2 = 2.0,
        prob_shift = 0.25,
        rng = TaskLocalRNG()
    ),
    adaptiveparticleswarm => (
        n_particles = 3,
        c1 = 2.0,
        c2 = 2.0,
        prob_shift = 0.25,
        rng = TaskLocalRNG()
    )
)
```

## Default Task Settings

```julia
const TUNING_PARAMS = Dict{Symbol,NamedTuple}(
    :classification => (
        n = 25,
        resampling = CV(nfolds=5),
        measure = MLJ.LogLoss(),
        train_best = true,
        acceleration = CPU1()
    ),
    :regression => (
        n = 25,
        resampling = CV(nfolds=5),
        measure = MLJ.RootMeanSquaredError(),
        train_best = true,
        acceleration = CPU1()
    )
)
```

## Range Function

```@docs
range(field::Union{Expr, Symbol}; kwargs...)
```

### Examples

```julia
# Create a range for max_depth from 2 to 10
depth_range = range(:max_depth, lower=2, upper=10)

# Create a range for min_samples_leaf with specific values
leaf_range = range(:min_samples_leaf, values=[1, 5, 10, 20])

# Create a log-scale range for learning rate
lr_range = range(:learning_rate, lower=1e-4, upper=0.1, scale=:log)
```

## Usage Examples

### Basic Usage

```julia
using SoleXplorer, MLJ

# Load data
X, y = @load_iris

# Simple random search tuning
model = symbolic_analysis(
    X, y;
    model=(type=:decisiontree, params=(;)),
    tuning=(
        method=(type=randomsearch, params=(;rng=123)),
        params=(n=20,),
        ranges=(
            range(:max_depth, lower=2, upper=10),
            range(:min_samples_leaf, values=[1, 5, 10])
        )
    )
)
```

### Advanced Usage with Particle Swarm

```julia
# Tune a random forest with PSO
modelset = symbolic_analysis(
    X, y;
    model=(type=:randomforest, params=(;n_trees=100)),
    tuning=(
        method=(type=particleswarm, params=(n_particles=10,)),
        params=(n=30, resampling=CV(nfolds=5)),
        ranges=(
            range(:max_depth, lower=3, upper=15),
            range(:min_samples_split, lower=2, upper=20)
        )
    )
)

# Show tuning results
best_params = get_best_params(modelset)
println("Best parameters: ", best_params)
println("Best score: ", get_best_score(modelset))
```

## Parameter Range Tips

When defining parameter ranges, consider:

- Use `scale=:log` for parameters that vary by orders of magnitude (learning rates, regularization)
- Use `values` for categorical or discrete parameters
- For parameters with a natural scale, use `lower` and `upper` with appropriate bounds

```julia
# Examples of well-defined ranges
learning_rate_range = range(:eta, lower=1e-5, upper=0.1, scale=:log)
depth_range = range(:max_depth, values=[3, 5, 7, 9])
regularization_range = range(:lambda, lower=1e-3, upper=10.0, scale=:log)
```

## See Also

- [`MLJTuning`](https://alan-turing-institute.github.io/MLJ.jl/dev/tuning_models/) - MLJ's tuning functionality
- [`MLJParticleSwarmOptimization.jl`](https://github.com/JuliaAI/MLJParticleSwarmOptimization.jl) - Particle Swarm Optimization for MLJ
- [`symbolic_analysis`](@ref) - Main analysis function using the tuning interface