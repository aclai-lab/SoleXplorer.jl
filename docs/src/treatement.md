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
using SoleXplorer, SoleData, MLJ, DataFrames

natopsloader = SoleData.Artifacts.NatopsLoader()
X, y = SoleData.Artifacts.load(natopsloader)

modelset = solexplorer(X, y)
```

### Time-Series (Modal) Data

For multivariate time-series, build a `DataTreatment` first:

```julia
dt = SoleXplorer.load_dataset(
    X,
    y,
    TreatmentGroup(
        aggrfunc=SoleXplorer.reducesize(
            reducefunc=mean,
            win=(SoleXplorer.splitwindow(nwindows=5),)
        )
    );
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
X_matrix = Matrix(X)
variable_names = names(X)

modelset = solexplorer(
    X_matrix,
    variable_names,
    y;
    float_type=Float64
)
```
