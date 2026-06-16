```@meta
CurrentModule = SoleXplorer
```

# SoleXplorer.jl

**SoleXplorer.jl** is a Julia package for end-to-end symbolic machine
learning analysis. It wraps [MLJ.jl](https://juliaai.github.io/MLJ.jl/)
and [Sole.jl](https://github.com/aclai-lab/Sole.jl) to provide a unified
interface for training, evaluating, and extracting interpretable rules
from symbolic models — including modal decision trees and random forests
for time-series data.

## Features

- **Unified workflow**: dataset setup, training, evaluation, and rule
  extraction in a single call.
- **Cross-validation**: full support for MLJ resampling strategies
  (`Holdout`, `CV`, `StratifiedCV`, `pCV`).
- **Hyperparameter tuning**: grid search, random search, and particle
  swarm optimization via MLJ tuning.
- **Modal models**: native support for
  [ModalDecisionTree](https://github.com/aclai-lab/ModalDecisionTrees.jl)
  and `ModalRandomForest` on multivariate time-series.
- **Interpretability**: automatic rule extraction from trained symbolic
  models.

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
modelset = solexplorer(X, y)

# access results
ds      = get_ds(modelset)        # DataSet configuration
models  = get_sole(modelset)      # trained symbolic models (one per fold)
perf    = get_measures(modelset)  # performance evaluation
vals    = get_values(modelset)    # raw measure values
```

## Cross-Validation

```julia
modelset = solexplorer(
    X, y;
    resampling=CV(nfolds=5, shuffle=true),
    rng=1,
    measures=(accuracy, log_loss, confusion_matrix)
)
```

## Hyperparameter Tuning

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
        measure=accuracy
    )
)
```

## Contents

```@contents
Pages = ["index.md", "dataset.md", "tuning.md",
    "symbolic_analysis.md", "treatement.md"]
Depth = 2
```

## About
The package is developed by the
[ACLAI Lab](https://aclai.unife.it/en/) @ University of Ferrara.
