```@meta
CurrentModule = SoleXplorer
```

# SoleXplorer
![header](https://raw.githubusercontent.com/aclai-lab/SoleXplorer.jl/refs/heads/main/logo.png)

## Introduction

Welcome to the documentation for [SoleXplorer](https://github.com/aclai-lab/SoleXplorer.jl), a Julia package for symbolic learning, timeseries analysis, rule extraction and mining.

## Installation

To install SoleXplorer, simply launch:
```julia
using Pkg
Pkg.add("SoleXplorer")
```

## [Feature Summary](@id feature-summary)

**SoleXplorer** is an interactive interface for exploring symbolic machine learning models, built on top of the [Sole.jl](https://github.com/aclai-lab/Sole.jl) ecosystem. It provides tools for visualizing, inspecting, and interacting with models derived from (logic-based) symbolic learning algorithms.
Built upon [MLJ framework](https://juliaai.github.io/MLJ.jl/stable/), extending its funcionality with tools like the ability of treat **time-series** analysis or the ability to retrieve what **rules** was used and how are interconnected.
It also offers the user the ability to build a machine-learining pipeline using only a function call, as the following code snippet demonstrates.

## Installation

You can install SoleXplorer by typing the following in the Julia REPL:
```julia
using Pkg
Pkg.add SoleXplorer
```

## Usage
```julia
using SoleXplorer, MLJ
# load a dataset
Xc, yc = @load_iris
```

SoleXplorer operates through 3 high-level functions, designed to be used sequentially:

**setup_dataset()**: prepares the dataset for analysis, where you set the parameters needed for dataset formatting and the related MLJ machine.
- choose the model to use.
- decide the resampling (or cross validation) strategy, with related train, validation and test ratio values.
- decide the tuning strategy to apply.
- set the global RNG seed.
- For time series datasets, it's possible to set features, windows for the type of compression suitable for the chosen model.

```julia
range = SOleXplorer.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
dsc = setup_dataset(
    Xc, yc;
    model=DecisionTreeClassifier(),
    resample=CV(nfolds=5, shuffle=true),
    rng=Xoshiro(1),
    tuning=Grid(resolution=10, resampling=CV(nfolds=3), range=range, measure=accuracy, repeats=2)    
)
```

**train_test()**: is the engine of SoleXplorer, built on top of MLJ and Sole:
- trains the machine through MLJ.
- converts the result into a Logic structure through SoleModels.
- tests the model.

```julia
solemc = train_test(dsc)
```

**symbolic_analysis()**: extracts the requested information
- we can specify the type and related parameters for rule extraction.
- we can request standard analysis metrics (accuracy, rms, ...).

```julia
modelc = symbolic_analysis(
    dsc, solemc;
    extractor=lumenExtractor(),
    measures=(accuracy, log_loss, confusion_matrix, kappa)
)
```

But it's also possible to condense the various parameters into a single call to symbolic_analysis:

```julia
range = SoleXplorer.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
modelc = symbolic_analysis(
    Xc, yc;
    model=DecisionTreeClassifier(),
    resample=CV(nfolds=5, shuffle=true),
    rng=Xoshiro(1),
    tuning=Grid(resolution=10, resampling=CV(nfolds=3), range=range, measure=accuracy, repeats=2),
    extractor=InTreesRuleExtractor(),
    measures=(accuracy, log_loss, confusion_matrix, kappa)      
)
```

One of the peculiarities of SoleXplorer is being as automated as possible, to make it easily usable even for machine learning newcomers.
The simplest way to launch an analysis with SoleXplorer is:

```julia
modelc = symbolic_analysis(Xc, yc)
```

## Under the Hood: How It Works

1. **Feature extraction**
   Data instances are seen as models of logical formalisms (see [`SoleLogics.jl`](https://github.com/aclai-lab/SoleLogics.jl)), and are represented in an optimized form for model checking (i.e., as _logisets_, see [`SoleData.jl`](https://github.com/aclai-lab/SoleData.jl)).

2. **Model Fitting**
   Symbolic models (e.g. decision trees, modal association rules) are trained via [`SoleModels.jl`](https://github.com/aclai-lab/SoleModels.jl)-compliant packages.

4. **Post-hoc Analysis**
   Metrics and human-readable logic representations are obtained via rule extraction algorithms from [`SolePostHoc.jl`](https://github.com/aclai-lab/SolePostHoc.jl).

5. **Association Analysis**
   Extract the association rules hidden in data via mining algorithms from [`ModalAssociationRules.jl`](https://github.com/aclai-lab/ModalAssociationRules.jl).

6. **Exploration Interface**
   `SoleXplorer.jl` ties everything into an interactive exploration session, allowing you to inspect rules, formulas, and patterns from your symbolic datasets and models.

## Components

- **SoleBase.jl**: Core types and windowing functions for symbolic learning, including 
  `Label` types (`CLabel`, `RLabel`, `XGLabel`) and windowing strategies (`movingwindow`, 
  `wholewindow`, `splitwindow`, `adaptivewindow`).

- **SoleData.jl**: Data structures and utilities for symbolic datasets, including 
  `scalarlogiset` and ARFF dataset loading capabilities.

- **SoleModels.jl**: Symbolic model implementations including `DecisionTree`, 
  `DecisionEnsemble`, `DecisionSet`, and rule extraction via `RuleExtractor`.

- **SolePostHoc.jl**: Post-hoc explainability methods, featuring `LumenRuleExtractor` 
  and other rule extraction algorithms for model interpretation.

- **ModalAssociationRules.jl**: Extract the association rules hidden in data via mining algorithms.

- **MLJ Ecosystem**: Full integration with MLJ for model evaluation, tuning, and 
  performance assessment including classification measures (`accuracy`, `confusion_matrix`, 
  `kappa`, `log_loss`) and regression measures (`rms`, `l1`, `l2`, `mae`, `mav`).

- **Time Series Features**: Comprehensive feature extraction via Catch22 library with 
  predefined feature sets (`base_set`, `catch9`, `catch22_set`, `complete_set`).

- **External Models**: Integration with popular ML libraries including DecisionTree.jl, 
  XGBoost.jl, and modal decision tree implementations.

## About

The package is developed by the [ACLAI Lab](https://aclai.unife.it/en/) @ University of Ferrara.