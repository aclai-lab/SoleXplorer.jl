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
    movingwindow, wholewindow, splitwindow, adaptivewindow,
    TreatmentGroup, aggregate, reducesize
using DataTreatments
const DT = DataTreatments

using CategoricalArrays
using DataFrames
using Random
using JLD2

# ---------------------------------------------------------------------------- #
#                                    utils                                     #
# ---------------------------------------------------------------------------- #
# feature extraction via Catch22
# export user friendly Catch22 nicknames
@reexport using DataTreatments:
    TreatmentGroup,
# export user friendly Catch22 nicknames
    mode_5, mode_10, embedding_dist, acf_timescale,
    acf_first_min, ami2, trev, outlier_timing_pos, outlier_timing_neg,
    whiten_timescale, forecast_error, ami_timescale, high_fluctuation,
    stretch_decreasing, stretch_high, entropy_pairs, rs_range, dfa,
    low_freq_power, centroid_freq, transition_variance, periodicity, base_set,
# feature sets
    catch9, catch22_set, complete_set,
# windowing
    movingwindow, wholewindow, splitwindow, adaptivewindow,
# imputation
    Interpolate, LOCF, NOCB, Substitute, SVD,
# balancing
    RandomOversampler, RandomWalkOversampler, ROSE, SMOTE,
    BorderlineSMOTE1, SMOTEN, SMOTENC, RandomUndersampler,
    ClusterUndersampler, ENNUndersampler, TomekUndersampler,
# normalization
    ZScore, MinMax, Center, Sigmoid, UnitEnergy, UnitPower,
    Scale, ScaleMad, ScaleFirst, PNorm1, PNorm, PNormInf,
    MissingSafe, Robust

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

const Regression =
    Union{DecisionTreeRegressor,RandomForestRegressor,XGBoostRegressor}
const Modal =
    Union{ModalDecisionTree,ModalRandomForest,ModalAdaBoost}

# ---------------------------------------------------------------------------- #
#                                   modules                                    #
# ---------------------------------------------------------------------------- #
include("measures.jl")

export range
export get_range, get_strategy, get_resampling, get_measure, get_repeats
export GridTuning, RandomTuning, ParticleTuning, AdaptiveTuning
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

export AbstractModelSet, ModelSet
export get_ds, get_sole, get_rules, get_measures, get_values
export solexplorer, solexplorer!
include("solexplorer.jl")

# ---------------------------------------------------------------------------- #
#                                  load save                                   #
# ---------------------------------------------------------------------------- #
export soleload, solesave
include("serialize.jl")

end
