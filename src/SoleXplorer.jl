"""
   SoleXplorer

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

- **SolePostHoc.jl**: Post-hoc explainability methods, featuring `LumenRuleExtractor` 
  and other rule extraction algorithms for model interpretation

- **ModalAssociationRules.jl**: Extract the association rules hidden in data via mining algorithms

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
    extractor=LumenExtractor(),
    measures=(accuracy, log_loss, confusion_matrix, kappa)      
)
```

"""
module SoleXplorer
using  Reexport

using  SoleBase: Label, CLabel, RLabel, XGLabel
using  SoleBase: movingwindow, wholewindow, splitwindow, adaptivewindow
using  SoleData: scalarlogiset
using  SoleData.Artifacts
using  SoleModels: Branch, ConstantModel
using  SoleModels: DecisionEnsemble, DecisionTree, DecisionXGBoost
using  SoleModels: AbstractModel, solemodel, weighted_aggregation, apply!
using  SoleModels: RuleExtractor, DecisionSet

@reexport using SoleData.Artifacts: NatopsLoader, load

@reexport using SolePostHoc: InTreesRuleExtractor, LumenRuleExtractor, BATreesRuleExtractor
@reexport using SolePostHoc: RULECOSIPLUSRuleExtractor, REFNERuleExtractor, TREPANRuleExtractor
using  SolePostHoc

@reexport using ModalAssociationRules: Item, Atom, ScalarCondition, VariableMin, VariableMax
@reexport using ModalAssociationRules: IA_L, box, diamond
@reexport using ModalAssociationRules: gsupport, gconfidence, glift, gconviction, gleverage
using ModalAssociationRules

# ---------------------------------------------------------------------------- #
#                                     MLJ                                      #
# ---------------------------------------------------------------------------- #

# performance measures for classification
@reexport using MLJ: accuracy, confusion_matrix, kappa, log_loss
# performance measures for regression 
@reexport using MLJ: rms, l1, l2, mae, mav
# cross-validation
@reexport using MLJ: Holdout, CV, StratifiedCV, TimeSeriesCV
# tuning
@reexport using MLJ: Grid, RandomSearch, LatinHypercube
@reexport using MLJParticleSwarmOptimization: ParticleSwarm, AdaptiveParticleSwarm
using  MLJ
using  MLJ: MLJBase, MLJTuning

# ---------------------------------------------------------------------------- #
#                              external packages                               #
# ---------------------------------------------------------------------------- #
@reexport using SoleData: load_arff_dataset
using  DataFrames
using  Random

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
# export user friendly Catch22 nicknames
export mode_5, mode_10, embedding_dist, acf_timescale, acf_first_min, ami2, trev, outlier_timing_pos
export outlier_timing_neg, whiten_timescale, forecast_error, ami_timescale, high_fluctuation, stretch_decreasing
export stretch_high, entropy_pairs, rs_range, dfa, low_freq_power, centroid_freq, transition_variance, periodicity
export base_set, catch9, catch22_set, complete_set
using  Catch22
include("featureset.jl")

# ---------------------------------------------------------------------------- #
#                                 interfaces                                   #
# ---------------------------------------------------------------------------- #

export partition
export get_X, get_y, get_train, get_test
include("partition.jl")

export WinFunction, MovingWindow, WholeWindow, SplitWindow, AdaptiveWindow
include("treatment.jl")

# ---------------------------------------------------------------------------- #
#                                   models                                     #
# ---------------------------------------------------------------------------- #
@reexport using MLJDecisionTreeInterface: 
    DecisionTreeClassifier, DecisionTreeRegressor,
    RandomForestClassifier, RandomForestRegressor,
    AdaBoostStumpClassifier
using MLJDecisionTreeInterface

@reexport using ModalDecisionTrees: 
    ModalDecisionTree, ModalRandomForest, ModalAdaBoost
using ModalDecisionTrees

@reexport using MLJXGBoostInterface: 
    XGBoostClassifier, XGBoostRegressor
using XGBoost, MLJXGBoostInterface

# ---------------------------------------------------------------------------- #
#                                   modules                                    #
# ---------------------------------------------------------------------------- #
export code_dataset, range
export setup_dataset
include("dataset.jl")

export train_test
include("apply.jl")
include("train_test.jl")

export Apriori, FPGrowth, Eclat
include("extractrules.jl")
include("associationrules.jl")

include("measures.jl")

export symbolic_analysis, symbolic_analysis!
export dsetup, solemodels, rules, associations
include("symbolic_analysis.jl")

end
