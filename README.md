<div align="center">
    <img src="banner.png" alt="SoleXplorer" width="900">
</div>

## A symbolic journey through your datasets

[![main](https://img.shields.io/badge/docs-main-blue.svg)](https://aclai-lab.github.io/SoleXplorer.jl/)
[![CI](https://github.com/aclai-lab/SoleXplorer.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/aclai-lab/SoleXplorer.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/aclai-lab/SoleXplorer.jl/graph/badge.svg?token=EJQ1MJOTDO)](https://codecov.io/gh/aclai-lab/SoleXplorer.jl)

**SoleXplorer.jl** is a Julia package for end-to-end symbolic machine
learning analysis. It wraps [MLJ.jl](https://juliaai.github.io/MLJ.jl/)
and [Sole.jl](https://github.com/aclai-lab/Sole.jl) to provide a unified
interface for training, evaluating, and extracting interpretable rules
from symbolic models, including modal decision trees and random forests
for time-series and image data.

## Features

- **Unified workflow**: dataset setup, training, evaluation, and rule
  extraction in a single call, either via `solexplorer(X, y; ...)` or by
  chaining `setup_dataset` and `solexplorer(ds)`.
- **Cross-validation**: full support for MLJ resampling strategies
  (`Holdout`, `CV`, `StratifiedCV`, `TimeSeriesCV`), plus SoleXplorer's own
  `pCV` (parametrized repeated cross-validation).
- **Modal models**: native support for
  [ModalDecisionTree](https://github.com/aclai-lab/ModalDecisionTrees.jl),
  `ModalRandomForest`, and `ModalAdaBoost` on multivariate time-series and
  image datasets.
- **Classic models**: `DecisionTreeClassifier/Regressor`,
  `RandomForestClassifier/Regressor`, `AdaBoostStumpClassifier`,
  `XGBoostClassifier/Regressor` (including early-stopping via a validation
  watchlist).
- **[DataTreatments.jl](https://github.com/PasoStudio73/DataTreatments.jl)
  integration**: since datasets are prepared through `DataTreatments.jl`,
  SoleXplorer inherits, all of its data-preparation power:
  - **Missing/NaN imputation** via
  [Impute.jl](https://github.com/invenia/Impute.jl)
    (`LOCF`, `NOCB`, `Interpolate`, `Substitute`, `SVD`, ...), applied at the
    tabular level and *inside* multidimensional elements (vectors, matrices).
  - **Class imbalance correction** via
    [Imbalance.jl](https://github.com/JuliaAI/Imbalance.jl), oversampling
    (`SMOTE`, `ROSE`, `RandomOversampler`, `BorderlineSMOTE1`, `SMOTENC`, ...)
    and undersampling (`RandomUndersampler`, `TomekUndersampler`,
    `ClusterUndersampler`, `ENNUndersampler`, ...).
  - **Multidimensional data support**: time-series, images, and other
    array-valued columns can be transformed via `aggregate` (feature
    extraction → flat tabular matrix, for traditional ML models) or
    `reducesize` (dimensionality reduction → array output, for modal
    analysis on large images/signals).
  - **Windowing functions** (`splitwindow`, `movingwindow`,
    `adaptivewindow`, `wholewindow`) let you work naturally with datasets
    whose elements have **non-uniform sizes** (e.g. variable-length
    time-series), by dividing each element into a comparable number of
    windows regardless of its original length.
  - **Normalization** via
    [Normalization.jl](https://github.com/PasoStudio73/Normalization.jl)
    (`ZScore`, `MinMax`, `Center`, `Sigmoid`, `UnitPower`, `Scale`,
    `ScaleMad`, `ScaleFirst`, `PNorm1`, `PNormInf`), applicable to both
    tabular and multidimensional data.
- **Interpretability**: automatic conversion of trained MLJ models into
  Sole symbolic models, ready for rule extraction and semantic analysis.

## Installation

```julia
using Pkg
Pkg.add("SoleXplorer")
```

Or from the REPL:

```
] add SoleXplorer
```

## Quick Start

```julia
using SoleXplorer, MLJ, DataFrames

# load a dataset
X, y = @load_iris
X = DataFrame(X)

# run the full workflow with default settings
modelset = solexplorer(X, y; model=RandomForestClassifier(n_trees=20), rng=42)

# access results
ds     = get_ds(modelset)        # DataSet configuration
models = get_sole(modelset)      # trained symbolic models (one per fold)
perf   = get_measures(modelset)  # performance evaluation
vals   = get_values(modelset)    # raw measure values
```

## Two-Step Workflow

`setup_dataset` and `solexplorer` can be split, which is useful when you
want to inspect or reuse the same `DataSet` across multiple runs:

```julia
ds = setup_dataset(X, y; model=RandomForestClassifier(n_trees=20), rng=42)
modelset = solexplorer(ds)
```

## Cross-Validation

```julia
modelset = solexplorer(
    X, y;
    model=DecisionTreeClassifier(),
    resampling=CV(nfolds=5, shuffle=true),
    rng=42,
    measures=(Accuracy(), LogLoss(), ConfusionMatrix(), Kappa())
)
```

Other supported strategies:

```julia
solexplorer(X, y; model=DecisionTreeClassifier(),
    resampling=Holdout(fraction_train=0.7, shuffle=true), rng=7)

solexplorer(X, y; model=DecisionTreeClassifier(),
    resampling=StratifiedCV(nfolds=4, shuffle=true), rng=99)
```

## Modal Models on Time-Series and Images

Multidimensional columns (time-series, images) are automatically detected.
Use `reducesize` to shrink them while preserving their array structure, so
they can be fed to modal models:

```julia
using SoleData

natopsloader = SoleData.Artifacts.NatopsLoader()
X, y = SoleData.Artifacts.load(natopsloader)

modelset = solexplorer(
    X, y;
    model=ModalDecisionTree(),
    aggrfunc=reducesize(
        reducefunc=mean,
        win=(splitwindow(nwindows=5),)
    ),
    resampling=StratifiedCV(nfolds=4, shuffle=true),
    rng=42,
    measures=(Accuracy(),)
)
```

The same applies to `ModalRandomForest` and `ModalAdaBoost`:

```julia
modelset = solexplorer(
    X, y;
    model=ModalRandomForest(),
    aggrfunc=reducesize(reducefunc=mean, win=(splitwindow(nwindows=3),)),
    resampling=Holdout(fraction_train=0.75, shuffle=true),
    rng=42,
    measures=(Accuracy(),)
)
```

## From Time-Series to Tabular Data

Use `aggregate` instead of `reducesize` to extract scalar features
(via windowing) and feed multidimensional data to traditional, non-modal
models:

```julia
modelset = solexplorer(
    X, y;
    model=DecisionTreeClassifier(),
    aggrfunc=SoleXplorer.aggregate(
        features=(mean, maximum, minimum),
        win=(splitwindow(nwindows=3),)
    ),
    measures=(Accuracy(),)
)
```

Available windowing functions, including `adaptivewindow`, which
gracefully handles datasets whose elements have **non-uniform length**:

```julia
aggrfunc=SoleXplorer.aggregate(win=(adaptivewindow(nwindows=3, overlap=0.2),))
```

## XGBoost with Early Stopping

```julia
modelset = solexplorer(
    X, y;
    model=XGBoostClassifier(early_stopping_rounds=10),
    resampling=CV(nfolds=3, shuffle=true),
    valid_ratio=0.2,
    rng=42,
    measures=(Accuracy(), ConfusionMatrix())
)
```

## Normalization

```julia
modelset = solexplorer(
    X, y;
    model=DecisionTreeClassifier(),
    norm=ZScore  # or MinMax, Center, Sigmoid, UnitPower, Scale, ...
)
```

## Missing Value Imputation

```julia
modelset = solexplorer(
    X, y;
    model=DecisionTreeClassifier(),
    impute=(LOCF(), NOCB())
)

# or, for continuous data
modelset = solexplorer(
    X, y;
    model=DecisionTreeClassifier(),
    impute=(Interpolate(),)
)
```

## Class Imbalance Correction

```julia
X, y = @load_iris
X = DataFrame(X)[1:end-25,:]
y = y[1:end-25]

modelset = solexplorer(
    X, y;
    model=DecisionTreeClassifier(),
    balance=SMOTE(k=5)
)

# undersampling is available too
modelset = solexplorer(
    X, y;
    model=DecisionTreeClassifier(),
    balance=TomekUndersampler()
)
```

## Regression

```julia
Xr, yr = @load_boston
Xr = DataFrame(Xr)

modelset = solexplorer(
    Xr, yr;
    model=XGBoostRegressor(),
    resampling=CV(nfolds=5, shuffle=true),
    rng=42,
    measures=(RootMeanSquaredError(), LPLoss())
)
```

## About
The package is developed by the
[ACLAI Lab](https://aclai.unife.it/en/) @ University of Ferrara.
