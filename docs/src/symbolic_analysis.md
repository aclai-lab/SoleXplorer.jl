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

## Examples

### Basic Classification

```julia
using SoleXplorer, MLJ, DataFrames

X, y = @load_iris
X = DataFrame(X)

modelset = solexplorer(X, y)

# inspect the first fold's symbolic model
m = get_sole(modelset)[1]
printmodel(m)
```

### Custom Measures

```julia
modelset = solexplorer(
    X, y;
    measures=(accuracy, log_loss, confusion_matrix, kappa)
)
vals = get_values(modelset)
```

### Modal Time-Series

```julia
# dt is a DataTreatment built from multivariate time-series
modelset = solexplorer(
    dt;
    model=ModalDecisionTree(),
    resampling=Holdout(fraction_train=0.7, shuffle=true),
    rng=1,
    measures=(accuracy, log_loss)
)
```

### Re-evaluating an Existing ModelSet

```julia
# add or update measures without retraining
solexplorer!(modelset; measures=(accuracy, kappa))
```
