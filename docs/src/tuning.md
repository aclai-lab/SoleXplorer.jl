```@meta
CurrentModule = SoleXplorer
```

# Hyperparameter Tuning

SoleXplorer integrates with MLJ's tuning infrastructure, providing a
simplified interface for common tuning strategies.

## Overview

Tuning is configured via a `Tuning` object passed to `setup_dataset` or
`solexplorer`. The range of hyperparameters to explore is specified
with `SoleXplorer.range`.

## Defining a Range

```@docs
SoleXplorer.range
```

## Tuning Strategies

```@docs
GridTuning
RandomTuning
```

## Examples

### Grid Search

```julia
r = SoleXplorer.range(
    :min_purity_increase; lower=0.001, upper=1.0, scale=:log
)
modelset = solexplorer(
    X, y;
    model=DecisionTreeClassifier(),
    resampling=CV(nfolds=5, shuffle=true),
    rng=1,
    tuning=GridTuning(
        resolution=10,
        resampling=CV(nfolds=3),
        range=r,
        measure=accuracy,
        repeats=2
    )
)
```

### Random Search

```julia
r = SoleXplorer.range(
    :min_purity_increase; lower=0.001, upper=1.0, scale=:log
)
modelset = solexplorer(
    X, y;
    model=DecisionTreeClassifier(),
    rng=1,
    tuning=RandomTuning(
        n=50,
        resampling=CV(nfolds=3),
        range=r,
        measure=accuracy
    )
)
```

### Particle Swarm Optimization

```julia
r = SoleXplorer.range(
    :min_purity_increase; lower=0.001, upper=1.0, scale=:log
)
modelset = solexplorer(
    X, y;
    model=DecisionTreeClassifier(),
    rng=1,
    tuning=PSOTuning(
        n=30,
        resampling=CV(nfolds=3),
        range=r,
        measure=accuracy
    )
)
```

### Multiple Ranges

```julia
r1 = SoleXplorer.range(
    :min_purity_increase; lower=0.001, upper=1.0, scale=:log
)
r2 = SoleXplorer.range(
    :max_depth; lower=2, upper=10
)
modelset = solexplorer(
    X, y;
    model=DecisionTreeClassifier(),
    rng=1,
    tuning=GridTuning(
        resolution=5,
        resampling=CV(nfolds=3),
        range=(r1, r2),
        measure=accuracy
    )
)
```