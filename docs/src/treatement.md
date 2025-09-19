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

# [Featuresets](@id featuresets)

### Basic Statistics
Standard statistical measures: `maximum`, `minimum`, `mean`, `median`, `std`, `cov`

### Catch22 Features
Canonical time-series characteristics covering:
- Distribution properties and extreme events
- Linear and nonlinear autocorrelation structures  
- Forecasting performance and scaling properties
- Symbolic dynamics and transition patterns

### Predefined Feature Sets

- [`base_set`](@ref): Minimal statistical features (4 features)
- [`catch9`](@ref): Curated subset combining statistics + key Catch22 (9 features)  
- [`catch22_set`](@ref): Complete Catch22 suite (22 features)
- [`complete_set`](@ref): All features combined (28 features)

### References

The Catch22 features are based on the Canonical Time-series Characteristics:
- **Repository**: https://github.com/DynamicsAndNeuralSystems/catch22
- **Paper**: Lubba, C.H., Sethi, S.S., Knaute, P. et al. "catch22: CAnonical Time-series CHaracteristics." *Data Min Knowl Disc* 33, 1821–1852 (2019). https://doi.org/10.1007/s10618-019-00647-x

```@docs
base_set
catch9
catch22_set
complete_set
```

See also: [`treatment`](@ref), [`setup_dataset`](@ref)

## All Catch22 Features

```@docs
mode_5
mode_10  
embedding_dist
acf_timescale
acf_first_min
ami2
trev
outlier_timing_pos
outlier_timing_neg
whiten_timescale
forecast_error
ami_timescale
high_fluctuation
stretch_decreasing
stretch_high
entropy_pairs
rs_range
dfa
low_freq_power
centroid_freq
transition_variance
periodicity
```