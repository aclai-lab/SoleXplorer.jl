```@meta
CurrentModule = SoleXplorer
```

# Data Treatment

This page documents the data-preparation options available through
[DataTreatments.jl](https://github.com/PasoStudio73/DataTreatments.jl),
which SoleXplorer wraps transparently. All preprocessing is configured
via keyword arguments passed directly to `setup_dataset`/`solexplorer`,
no separate `DataTreatment` object needs to be built by hand.

## Overview

`setup_dataset` accepts a `DataFrame` (or raw matrix), a target vector,
and an Sole model, plus a set of optional treatment keywords:

- `aggrfunc`: how multidimensional columns (time-series, images) are
  turned into model-ready data, see [Aggregation vs. Size Reduction](@ref).
- `impute`: missing/NaN handling, powered by
  [Impute.jl](https://github.com/invenia/Impute.jl).
- `balance`: class-imbalance correction, powered by
  [Imbalance.jl](https://github.com/JuliaAI/Imbalance.jl).
- `norm`: normalization, powered by
  [Normalization.jl](https://github.com/PasoStudio73/Normalization.jl).
- `valid_ratio`: fraction of the training set reserved for validation
  (used by models with early stopping, e.g. `XGBoostClassifier`).

## Examples

### Tabular Data

For standard tabular data, no treatment keywords are required:

```julia
using SoleXplorer, SoleData, MLJ, DataFrames

X, y = @load_iris
X = DataFrame(X)

modelset = solexplorer(X, y; model=XGBoostClassifier())
```

### Aggregation vs. Size Reduction

Multidimensional columns (time-series, images) are handled by
`aggrfunc`, which selects one of two strategies:

- **`reducesize`**, shrinks each element (windowing + reduction
  function) while preserving its array structure. Use this with
  **modal** models, which operate natively on vectors/matrices.
- **`aggregate`**, extracts scalar features (via windowing) from each
  element, producing a flat tabular matrix. Use this with
  **traditional** (non-modal) models.

#### Modal (Time-Series) Data, `reducesize`

```julia
using SoleXplorer, SoleData, MLJ, DataFrames

natopsloader = SoleData.Artifacts.NatopsLoader()
X, y = SoleData.Artifacts.load(natopsloader)

modelset = solexplorer(
    X, y;
    model=ModalDecisionTree(),
    aggrfunc=reducesize(
        reducefunc=mean,
        win=(splitwindow(nwindows=5),)
    ),
    resampling=Holdout(fraction_train=0.7, shuffle=true),
    rng=42
)
```

The same applies to `ModalRandomForest` and `ModalAdaBoost`.

#### Traditional Models, `aggregate`

```julia
modelset = solexplorer(
    X, y;
    model=DecisionTreeClassifier(),
    aggrfunc=SoleXplorer.aggregate(
        features=(mean, maximum, minimum),
        win=(splitwindow(nwindows=3),)
    )
)
```

Available windowing functions include `splitwindow`, `movingwindow`,
`adaptivewindow`, and `wholewindow`. `adaptivewindow` is especially
useful for datasets whose elements have **non-uniform length**, since
it produces a comparable number of windows regardless of the original
element size:

```julia
aggrfunc=SoleXplorer.aggregate(win=(adaptivewindow(nwindows=3, overlap=0.2),))
```

### Missing Value Imputation

```julia
X, y = @load_iris
X = DataFrame(X)

modelset = solexplorer(
    X, y;
    model=DecisionTreeClassifier(),
    impute=(LOCF(), NOCB())
)
```

### Class Imbalance Correction

```julia
modelset = solexplorer(
    X, y;
    model=DecisionTreeClassifier(),
    balance=SMOTE(k=5)
)
```

> [!WARNING]
> `balance` requires data free of `missing`/`NaN` values, since
> [Imbalance.jl](https://github.com/JuliaAI/Imbalance.jl) cannot handle
> them. When combining `impute` and `balance`, SoleXplorer guarantees
> imputation is always applied first.

> [!WARNING]
> At the time of writing, [Imbalance.jl](https://github.com/JuliaAI/Imbalance.jl)
> only supports tabular data: `balance` cannot be applied to
> multidimensional columns (time-series, images).

### Normalization

```julia
natopsloader = SoleData.Artifacts.NatopsLoader()
X, y = SoleData.Artifacts.load(natopsloader)

modelset = solexplorer(
    X, y;
    model=DecisionTreeClassifier(),
    norm=ZScore
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
    model=DecisionTreeClassifier(),
    float_type=Float32
)
```