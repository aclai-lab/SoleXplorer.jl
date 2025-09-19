```@meta
CurrentModule = SoleXplorer
```

# [Treatement](@id treatement)
With multidimensional datasets there are two possible types of work:

1. Use of Propositional algorithms (DecisionTree, XGBoost):
   - Applies windowing to divide time series into segments
   - Extracts scalar features (max, min, mean, etc.) from each window
   - Returns a standard tabular DataFrame

2. Use of Modal algorithms (ModalDecisionTree):
   - Creates windowed time series preserving temporal structure
   - Applies reduction functions to manage dimensionality

```@docs
treatment(X::AbstractDataFrame, treat::Symbol)
```

Windowing strategies availables for reduce/aggregation time-series datasets.

```@docs
MovingWindow
WholeWindow
SplitWindow
AdaptiveWindow
AbstractWinFunction
WinFunction
```