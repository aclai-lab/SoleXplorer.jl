module SoleXplorer
using Reexport

using SoleData: scalarlogiset
using SoleData.Artifacts

@reexport using SoleModels: Label, CLabel, RLabel, XGLabel
using SoleModels:
    Branch, ConstantModel,
    DecisionEnsemble, DecisionTree, DecisionXGBoost,
    AbstractModel, solemodel, weighted_aggregation, apply!,
    RuleExtractor, DecisionSet,
    readmetrics

@reexport using SoleData.Artifacts: NatopsLoader, load

@reexport using SolePostHoc.RuleExtraction:
    InTreesRuleExtractor, LumenRuleExtractor, BATreesRuleExtractor,
# @reexport using SolePostHoc.RuleExtraction: RULECOSIPLUSRuleExtractor
    REFNERuleExtractor, TREPANRuleExtractor
using SolePostHoc.RuleExtraction

# ---------------------------------------------------------------------------- #
#                                     MLJ                                      #
# ---------------------------------------------------------------------------- #
@reexport using MLJ:
# performance measures for classification
    Accuracy, Kappa, LogLoss, FScore,
    FalseNegativeRate, FalsePositiveRate, TrueNegativeRate, TruePositiveRate,
    ConfusionMatrix,
# performance measures for regression 
    RootMeanSquaredError, LPLoss,
# cross-validation
    Holdout, CV, StratifiedCV, TimeSeriesCV

# tuning
using MLJParticleSwarmOptimization
const PSO = MLJParticleSwarmOptimization

using MLJ
using MLJ: MLJBase, MLJTuning
# custom resampling strategy
import MLJ.MLJBase: train_test_pairs

# ---------------------------------------------------------------------------- #
#                              external packages                               #
# ---------------------------------------------------------------------------- #
@reexport using DataTreatments:
    load_dataset, get_tabular, get_target,
    TreatmentGroup, aggregate, reducesize,
    # catch22
    mode_5, mode_10, embedding_dist, acf_timescale, acf_first_min, ami2,
    trev, outlier_timing_pos, outlier_timing_neg, whiten_timescale,
    forecast_error, ami_timescale, high_fluctuation, stretch_decreasing,
    stretch_high, entropy_pairs, rs_range, dfa, low_freq_power,
    centroid_freq, transition_variance, periodicity,
    # features sets
    base_set, catch9, catch22_set, complete_set,
    # windowing
    movingwindow, wholewindow, splitwindow, adaptivewindow,
    # balancing
    RandomOversampler, RandomWalkOversampler, ROSE, SMOTE,
    BorderlineSMOTE1, SMOTEN, SMOTENC, RandomUndersampler,
    ClusterUndersampler, ENNUndersampler, TomekUndersampler,
    # normalization
    ZScore, MinMax, Center, Sigmoid, UnitEnergy, UnitPower,
    Scale, ScaleMad, ScaleFirst, PNorm1, PNorm, PNormInf,
    MissingSafe, Robust
    
using DataTreatments
const DT = DataTreatments

using CategoricalArrays
using DataFrames
using Random

# ---------------------------------------------------------------------------- #
#                                 interfaces                                   #
# ---------------------------------------------------------------------------- #
export partition, pCV
export get_X, get_y, get_train, get_test
include("partition.jl")

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

@reexport using ModalDecisionLists:
    DecisionListClassifier, RandomDecisionListClassifier
using ModalDecisionLists

const Regression =
    Union{DecisionTreeRegressor,RandomForestRegressor,XGBoostRegressor}
const Modal =
    Union{ModalDecisionTree,ModalRandomForest,ModalAdaBoost}

# ---------------------------------------------------------------------------- #
#                                   modules                                    #
# ---------------------------------------------------------------------------- #
include("measures.jl")

export range,
    get_range, get_strategy, get_resampling, get_measure, get_repeats,
    GridTuning, RandomTuning, ParticleTuning, AdaptiveTuning
include("tuning.jl")

export AbstractDataSet, PropositionalDataSet, ModalDataSet, DataSet,
    code_dataset, get_mach, get_mach_model, get_logiset, setup_dataset
include("dataset.jl")

export train_test
include("apply.jl")
include("train_test.jl")

include("extractrules.jl")

export AbstractModelSet, ModelSet, solexplorer, solexplorer!,
    get_ds, get_sole, get_rules, get_measures, get_values
include("solexplorer.jl")

end
