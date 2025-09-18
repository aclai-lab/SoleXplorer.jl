# SoleXplorer
## A symbolic journey through your datasets

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://aclai-lab.github.io/SoleXplorer.jl/)
[![CI](https://github.com/aclai-lab/SoleXplorer.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/aclai-lab/SoleXplorer.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/aclai-lab/SoleXplorer.jl/graph/badge.svg?token=EJQ1MJOTDO)](https://codecov.io/gh/aclai-lab/SoleXplorer.jl)

**SoleXplorer.jl** is an interactive interface for exploring symbolic machine learning models, built on top of the [Sole.jl](https://github.com/aclai-lab/Sole.jl) ecosystem. It provides tools for visualizing, inspecting, and interacting with models derived from (logic-based) symbolic learning algorithms.

Key features:
* Can handle both classification and regression tasks.
* Inspect metrics.
* Works also on time-series based datasets using modal logic.
* View rules and their metrics.
* Inspect logical formulas and their evaluation.
* View modal rule associations.
* Integrated GUI.

## Installation

```julia
using Pkg
Pkg.add SoleXplorer
```

## Quick Start

### Decision tree

Every parameter is defaulted: start analysis simply passing your raw dataset:

```julia
using SoleXplorer, MLJ

# Load example dataset
Xc, yc = @load_iris

# Train a decision tree
modelc = symbolic_analysis(Xc, yc)
```

Of course, customizations are possible:

```julia
using Random

range = SoleXplorer.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
modelc = symbolic_analysis(
    Xc, yc;
    model=DecisionTreeClassifier(),
    resample=CV(nfolds=5, shuffle=true),
    rng=Xoshiro(1),
    tuning=(tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=accuracy, repeats=2),
    extractor=InTreesRuleExtractor(),
    measures=(accuracy, log_loss, confusion_matrix, kappa)      
)
```

### Temporal association rules

```julia
# Load a temporal dataset
natopsloader = NatopsLoader()
Xts, yts = SoleXplorer.load(natopsloader)

# Train a modal decision tree
modelts = symbolic_analysis(
    Xts, yts;
    model=ModalDecisionTree(),
    resample=Holdout(fraction_train=0.8, shuffle=true),
    rng=Xoshiro(1),
    features=(minimum, maximum),
    measures=(log_loss, accuracy, confusion_matrix, kappa)
)
```

## Related packages
SoleXplorer extensively uses the following packages:
* [`SoleLogics`](https://github.com/aclai-lab/SoleLogics.jl): modal and temporal logic systems.
* [`MLJ`](https://github.com/JuliaAI/MLJ.jl): provides all machine learning frameworks.
* [`SolePostHoc`](https://github.com/aclai-lab/SolePostHoc.jl): for rule extraction.
* [`ModalAssociationRules`](https://github.com/aclai-lab/ModalAssociationRules.jl): for mining association rules.

## About

The package is developed by the [ACLAI Lab](https://aclai.unife.it/en/) @ University of Ferrara.

