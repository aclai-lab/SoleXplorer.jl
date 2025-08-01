"""
   SoleXplorer

[`SoleXplorer`](https://github.com/aclai-lab/SoleXplorer.jl) is a comprehensive toolbox for 
symbolic machine learning and explainable AI in Julia.

It brings together functionality from the Sole ecosystem components, providing 
symbolic learning algorithms, rule extraction methods, and modal timeseries feature extraction.

# Components

- **SoleBase.jl**: Core types and windowing functions for symbolic learning, including 
  `Label` types (`CLabel`, `RLabel`, `XGLabel`) and windowing strategies (`movingwindow`, 
  `wholewindow`, `splitwindow`, `adaptivewindow`)

- **SoleData.jl**: Data structures and utilities for symbolic datasets, including 
  `scalarlogiset` and ARFF dataset loading capabilities

- **SoleModels.jl**: Symbolic model implementations including `DecisionTree`, 
  `DecisionEnsemble`, `DecisionSet`, and rule extraction via `RuleExtractor`

- **SolePostHoc.jl**: Post-hoc explainability methods, featuring `InTreesRuleExtractor` 
  and other rule extraction algorithms for model interpretation

- **MLJ Ecosystem**: Full integration with MLJ for model evaluation, tuning, and 
  performance assessment including classification measures (`accuracy`, `confusion_matrix`, 
  `kappa`, `log_loss`) and regression measures (`rms`, `l1`, `l2`, `mae`, `mav`)

- **Time Series Features**: Comprehensive feature extraction via Catch22 library with 
  predefined feature sets (`base_set`, `catch9`, `catch22_set`, `complete_set`)

- **External Models**: Integration with popular ML libraries including DecisionTree.jl, 
  XGBoost.jl, and modal decision tree implementations

# Typical Workflow

```julia
using SoleXplorer

# Load and prepare data
dataset = load_arff_dataset("path/to/data.arff")
X, y = setup_dataset(dataset)

# Apply windowing for time series
windowed_data = MovingWindow(window_size=10)(X)

# Train a modal decision tree
model = ModalDecisionTree()
mach = machine(model, windowed_data, y)
fit!(mach)

# Extract rules for explainability
extractor = InTreesRuleExtractor()
rules = extractrules(mach, extractor)

# Evaluate performance
cv_results = evaluate!(mach, resampling=CV(nfolds=5), measure=accuracy)
```

"""
module SoleXplorer
using  Reexport

using  SoleBase: Label, CLabel, RLabel, XGLabel
using  SoleBase: movingwindow, wholewindow, splitwindow, adaptivewindow
using  SoleData: scalarlogiset
using  SoleModels: Branch, ConstantModel
using  SoleModels: DecisionEnsemble, DecisionTree
using  SoleModels: AbstractModel, solemodel, weighted_aggregation, apply!
using  SoleModels: RuleExtractor, DecisionSet
using  SolePostHoc
@reexport using SolePostHoc: InTreesRuleExtractor
# @reexport using SolePostHoc: 
#     LumenRuleExtractor, BATreesRuleExtractor, REFNERuleExtractor, RULECOSIPLUSRuleExtractor     

# ---------------------------------------------------------------------------- #
#                                     MLJ                                      #
# ---------------------------------------------------------------------------- #
using  MLJ
using  MLJ: MLJBase, MLJTuning

# performance measures for classification
@reexport using MLJ: accuracy, confusion_matrix, kappa, log_loss
# performance measures for regression 
@reexport using MLJ: rms, l1, l2, mae, mav

# ---------------------------------------------------------------------------- #
#                              external packages                               #
# ---------------------------------------------------------------------------- #
using  DataFrames
using  Random

@reexport using SoleData: load_arff_dataset

# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
"""
    Optional{T}

Type alias for `Union{T, Nothing}`.
"""
const Optional{T} = Union{T, Nothing}

# ---------------------------------------------------------------------------- #
#                                    utils                                     #
# ---------------------------------------------------------------------------- #
# feature extraction via Catch22
using  Catch22
include("featureset.jl")

# export user friendly Catch22 nicknames
export mode_5, mode_10, embedding_dist, acf_timescale, acf_first_min, ami2, trev, outlier_timing_pos
export outlier_timing_neg, whiten_timescale, forecast_error, ami_timescale, high_fluctuation, stretch_decreasing
export stretch_high, entropy_pairs, rs_range, dfa, low_freq_power, centroid_freq, transition_variance, periodicity
export base_set, catch9, catch22_set, complete_set

# ---------------------------------------------------------------------------- #
#                                 interfaces                                   #
# ---------------------------------------------------------------------------- #
@reexport using MLJ: Holdout, CV, StratifiedCV, TimeSeriesCV
include("partition.jl")
export partition
export get_X, get_y, get_train, get_test

include("treatment.jl")
export WinFunction, MovingWindow, WholeWindow, SplitWindow, AdaptiveWindow

@reexport using MLJ: Grid, RandomSearch, LatinHypercube
@reexport using MLJParticleSwarmOptimization: ParticleSwarm, AdaptiveParticleSwarm

# ---------------------------------------------------------------------------- #
#                                   models                                     #
# ---------------------------------------------------------------------------- #
using MLJDecisionTreeInterface
@reexport using MLJDecisionTreeInterface: 
    DecisionTreeClassifier, DecisionTreeRegressor,
    RandomForestClassifier, RandomForestRegressor,
    AdaBoostStumpClassifier

using ModalDecisionTrees
@reexport using ModalDecisionTrees: 
    ModalDecisionTree, ModalRandomForest, ModalAdaBoost

using XGBoost, MLJXGBoostInterface
@reexport using MLJXGBoostInterface: 
    XGBoostClassifier, XGBoostRegressor

# ---------------------------------------------------------------------------- #
#                                   modules                                    #
# ---------------------------------------------------------------------------- #
include("dataset.jl")
export code_dataset, range
export setup_dataset

include("apply.jl")
include("train_test.jl")
export train_test

include("extractrules.jl")
include("symbolic_analysis.jl")
export symbolic_analysis
export solemodels

end
