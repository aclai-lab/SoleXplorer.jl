module SoleXplorer
using Reexport

using SoleData: PropositionalLogiset, scalarlogiset

@reexport using SoleModels: Label, CLabel, RLabel, XGLabel
using SoleModels:
    Branch, ConstantModel,
    DecisionEnsemble, DecisionTree, DecisionXGBoost,
    AbstractModel, solemodel, weighted_aggregation, apply!,
    DecisionSet, readmetrics

# ---------------------------------------------------------------------------- #
#                                     MLJ                                      #
# ---------------------------------------------------------------------------- #
@reexport using MLJ:
# performance measures for classification
    Accuracy, Kappa, LogLoss, FScore,
    TruePositiveRate, # sensitivity, binary
    TrueNegativeRate, # specificity, binary
    FalseNegativeRate, FalsePositiveRate,
    ConfusionMatrix,
# performance measures for regression 
    RootMeanSquaredError, LPLoss,
# cross-validation
    Holdout, CV, StratifiedCV, TimeSeriesCV

using MLJ
using MLJ: MLJBase
# custom resampling strategy
import MLJ.MLJBase: train_test_pairs

# ---------------------------------------------------------------------------- #
#                              external packages                               #
# ---------------------------------------------------------------------------- #
@reexport using DataTreatments:
    load_dataset, get_tabular, get_target,
    TreatmentGroup, aggregate, reducesize,
    # windowing
    wholewindow, splitwindow, adaptivewindow,
    # balancing
    RandomOversampler, RandomWalkOversampler, ROSE, SMOTE,
    BorderlineSMOTE1, SMOTEN, SMOTENC, RandomUndersampler,
    ClusterUndersampler, ENNUndersampler, TomekUndersampler,
    # normalization
    ZScore, MinMax, Center, Sigmoid, UnitPower,
    Scale, ScaleMad, ScaleFirst, PNorm1, PNormInf,
    # imputation
    Interpolate, LOCF, NOCB, SVD, Substitute
    
using DataTreatments
const DT = DataTreatments

@reexport using SignalEncodings: Uniform, Quantile, Jenks

using SignalEncodings
const SE = SignalEncodings

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

const Regression =
    Union{DecisionTreeRegressor,RandomForestRegressor,XGBoostRegressor}
const Modal =
    Union{ModalDecisionTree,ModalRandomForest,ModalAdaBoost}

# ---------------------------------------------------------------------------- #
#                                  sections                                    #
# ---------------------------------------------------------------------------- #
include("measures.jl")

export AbstractDataSet, DataSet, setup_dataset,
    get_X, get_y, get_mach, get_mach_model, get_rng
include("dataset.jl")

include("apply.jl")

export train_test
include("train_test.jl")

export AbstractModelSet, ModelSet, solexplorer, solexplorer!,
    get_ds, get_sole, get_rules, get_measures, get_values,
    get_dataset, get_targets, show_measures
include("main.jl")

end
