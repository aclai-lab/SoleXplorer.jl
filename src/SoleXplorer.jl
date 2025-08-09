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
@reexport using SolePostHoc: InTreesRuleExtractor, LumenRuleExtractor, BATreesRuleExtractor
@reexport using SolePostHoc: RULECOSIPLUSRuleExtractor, REFNERuleExtractor, TREPANRuleExtractor  

# ---------------------------------------------------------------------------- #
#                                     MLJ                                      #
# ---------------------------------------------------------------------------- #
using  MLJ
using  MLJ: MLJBase, MLJTuning

# classification measures
@reexport using MLJ: accuracy, confusion_matrix, kappa, log_loss
# regression measures
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
const Optional{T} = Union{T, Nothing}

# ---------------------------------------------------------------------------- #
#                                    utils                                     #
# ---------------------------------------------------------------------------- #
using  Catch22
include("featureset.jl")
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
export solemodels, rules

end
