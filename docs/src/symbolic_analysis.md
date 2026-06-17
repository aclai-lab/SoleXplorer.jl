```@meta
CurrentModule = SoleXplorer
```

# Symbolic Analysis

This page documents the symbolic model training, rule extraction, and
evaluation API.

## Overview

After dataset setup, SoleXplorer trains an MLJ model on each CV fold,
converts it to a Sole symbolic model, and evaluates it using the
specified measures.

The full pipeline is:

```
DataSet  →  train/test  →  SoleModel  →  rule extraction  →  Measures
```

## API Reference

```@docs
ModelSet
solexplorer
solexplorer!
get_ds
get_sole
get_measures
get_values
```

## Examples

### Basic Classification

```julia
using SoleXplorer, MLJ, DataFrames

X, y = @load_iris
X = DataFrame(X)

modelset = solexplorer(X, y)

# inspect the first fold's symbolic model
m = get_sole(modelset)[1]
```

### Custom Measures

```julia
modelset = solexplorer(
    X, y;
    measures=(accuracy, log_loss, confusion_matrix, kappa)
)
vals = get_values(modelset)
```

### Classification with Cross-Validation and Tuning

```julia
range = SoleXplorer.range(
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
        range=range,
        measure=accuracy,
        repeats=2
    ),
    measures=(accuracy, log_loss, confusion_matrix, kappa)
)
```

### Modal Time-Series

```julia
using DataTreatments

dt = load_dataset(X,y)
# dt is a DataTreatment built from multivariate time-series
modelset = solexplorer(
    dt;
    model=XGBoostClassifier(),
    resampling=Holdout(fraction_train=0.7, shuffle=true),
    rng=1,
    measures=(accuracy, log_loss)
)
```

### From a Pre-built DataTreatment

```julia
modelset = solexplorer(dt; model=DecisionTreeClassifier(), rng=1)
```

### Re-evaluating an Existing ModelSet

```julia
# add or update measures without retraining
solexplorer!(modelset; measures=(accuracy, kappa))
```

### Accessing Results

```julia
ds = get_ds(modelset)
models = get_sole(modelset)
perf = get_measures(modelset)
vals = get_values(modelset)
```