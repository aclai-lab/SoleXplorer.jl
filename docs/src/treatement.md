```@meta
CurrentModule = SoleXplorer
```

# Data Treatment

This page documents the data treatment and preprocessing API, which
handles loading, transforming, and preparing raw data for symbolic
analysis.

## Overview

A `DataTreatment` encapsulates the feature matrix, target vector, and
all preprocessing steps applied before training. It is the recommended
entry point for modal (time-series) workflows.

## Examples

### Tabular Data

For standard tabular data, `setup_dataset` and `solexplorer` accept a
`DataFrame` directly without needing an explicit `DataTreatment`:

```julia
using SoleXplorer, MLJ, DataFrames

X, y = @load_iris
X = DataFrame(X)

modelset = solexplorer(X, y)
```

### Time-Series (Modal) Data

For multivariate time-series, build a `DataTreatment` first:

```julia
using SoleXplorer, DataTreatments

dt = DataTreatments.load_dataset(
    X_timeseries,
    y;
    treatments=DataTreatments.DefaultTreatmentGroup
)

modelset = solexplorer(
    dt;
    model=ModalDecisionTree(),
    resampling=Holdout(fraction_train=0.7, shuffle=true),
    rng=1
)
```

### From a Raw Matrix

```julia
modelset = solexplorer(
    X_matrix,
    variable_names,
    y;
    treatment_ds=true,
    leftover_ds=false,
    float_type=Float64
)
```
