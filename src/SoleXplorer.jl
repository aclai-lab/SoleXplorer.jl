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
@reexport using SoleModels: readmetrics

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
using  MLJParticleSwarmOptimization
const  PSO = MLJParticleSwarmOptimization
using  MLJ
using  MLJ: MLJBase, MLJTuning
# balancing
using MLJBalancing
@reexport using MLJBalancing:
    BorderlineSMOTE1, ClusterUndersampler, ENNUndersampler, ROSE,
    RandomOversampler, RandomUndersampler, RandomWalkOversampler,
    SMOTE, SMOTEN, SMOTENC, TomekUndersampler

# ---------------------------------------------------------------------------- #
#                              external packages                               #
# ---------------------------------------------------------------------------- #
@reexport using SoleData: load_arff_dataset
using  DataFrames
using  Random
using  JLD2

# ---------------------------------------------------------------------------- #
#                                   timeout                                    #
# ---------------------------------------------------------------------------- #
# using TimeOut

const preprocess = "using SoleXplorer\n"

# ---------------------------------------------------------------------------- #
#                                 maybe types                                  #
# ---------------------------------------------------------------------------- #
"""
    Maybe{T}

Type alias for `Union{T, Nothing}`.
"""
const Maybe{T} = Union{T, Nothing}

const MaybeVector = Maybe{AbstractVector}
const MaybeNTuple = Maybe{NamedTuple}

# ---------------------------------------------------------------------------- #
#                                    utils                                     #
# ---------------------------------------------------------------------------- #
# feature extraction via Catch22
# export user friendly Catch22 nicknames
export mode_5, mode_10, embedding_dist, acf_timescale, acf_first_min, ami2,
       trev, outlier_timing_pos, outlier_timing_neg, whiten_timescale,
       forecast_error, ami_timescale, high_fluctuation, stretch_decreasing,
       stretch_high, entropy_pairs, rs_range, dfa, low_freq_power, centroid_freq,
       transition_variance, periodicity, base_set, catch9, catch22_set, complete_set
using  Catch22
include("featureset.jl")

# ---------------------------------------------------------------------------- #
#                                 interfaces                                   #
# ---------------------------------------------------------------------------- #
export get_X, get_y, get_train, get_test
include("partition.jl")

export AbstractWinFunction, WinFunction
export MovingWindow, WholeWindow, SplitWindow, AdaptiveWindow
export treatment
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

const Regression = Union{DecisionTreeRegressor, RandomForestRegressor, XGBoostRegressor}
const Modal  = Union{ModalDecisionTree, ModalRandomForest, ModalAdaBoost}

# ---------------------------------------------------------------------------- #
#                                   modules                                    #
# ---------------------------------------------------------------------------- #
include("measures.jl")
export range
export get_range, get_strategy, get_resampling, get_measure, get_repeats
export GridTuning, RandomTuning, CubeTuning, ParticleTuning, AdaptiveTuning
include("tuning.jl")

export AbstractDataSet
export PropositionalDataSet, ModalDataSet, DataSet
export code_dataset
export get_mach, get_mach_model, get_logiset
export setup_dataset
include("dataset.jl")

export train_test
include("apply.jl")
include("train_test.jl")

include("extractrules.jl")

export Apriori, FPGrowth, Eclat
include("associationrules.jl")

export AbstractModelSet, ModelSet
export dsetup, solemodels, rules, associations
export performance, measures, values
export show_measures
export symbolic_analysis, symbolic_analysis!
include("symbolic_analysis.jl")

# ---------------------------------------------------------------------------- #
#                                  load save                                   #
# ---------------------------------------------------------------------------- #
export soleload, solesave
include("serialize.jl")

end
