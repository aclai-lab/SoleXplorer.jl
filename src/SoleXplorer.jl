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
using SoleXplorer, MLJ

range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
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

"""
module SoleXplorer
using  Reexport

using  SoleBase: Label, CLabel, RLabel, XGLabel
using  SoleBase: movingwindow, wholewindow, splitwindow, adaptivewindow
using  SoleData: scalarlogiset
using  SoleModels: Branch, ConstantModel
using  SoleModels: DecisionEnsemble, DecisionTree, DecisionXGBoost
using  SoleModels: AbstractModel, solemodel, weighted_aggregation, apply!
using  SoleModels: RuleExtractor, DecisionSet

using  SolePostHoc
@reexport using SolePostHoc: InTreesRuleExtractor, LumenRuleExtractor, BATreesRuleExtractor
@reexport using SolePostHoc: RULECOSIPLUSRuleExtractor, REFNERuleExtractor, TREPANRuleExtractor

using ModalAssociationRules
@reexport using ModalAssociationRules: Item, Atom, ScalarCondition, VariableMin, VariableMax
@reexport using ModalAssociationRules: IA_L, box, diamond
@reexport using ModalAssociationRules: gsupport, gconfidence, glift, gconviction, gleverage

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
    Maybe{T}

Type alias for `Union{T, Nothing}`.
"""
const Maybe{T} = Union{T, Nothing}

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
include("associationrules.jl")
export Apriori, FPGrowth, Eclat

include("symbolic_analysis.jl")
export symbolic_analysis, symbolic_analysis!
export dsetup, solemodels, rules, associations

end
