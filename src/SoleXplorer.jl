module SoleXplorer
using  Reexport

using  SoleBase: Label, CLabel, RLabel, XGLabel
using  SoleBase: movingwindow, wholewindow, splitwindow, adaptivewindow
using  SoleData
using  SoleData: PatchedFunction, nanpatchedfunction
using  SoleModels
using  SoleModels: AbstractModel, DecisionList, DecisionForest, DecisionEnsemble, DecisionSet
using  SoleModels: RuleExtractor
using  SolePostHoc
@reexport using SolePostHoc: InTreesRuleExtractor
# @reexport using SolePostHoc: 
#     LumenRuleExtractor, BATreesRuleExtractor, REFNERuleExtractor, RULECOSIPLUSRuleExtractor     

import MultiData.hasnans

using  DataFrames
using  Random

# ---------------------------------------------------------------------------- #
#                                     MLJ                                      #
# ---------------------------------------------------------------------------- #
using  MLJ, MLJBase
import MLJ: MLJType

# classification measures
@reexport using MLJ: accuracy, confusion_matrix, kappa, log_loss
# regression measures
@reexport using MLJ: rms, l1, l2, mae, mav

# ---------------------------------------------------------------------------- #
#                                 show utils                                   #
# ---------------------------------------------------------------------------- #
# simplified string rep of a Type:
simple_repr(T) = string(T.name.name)

# ---------------------------------------------------------------------------- #
#                                    utils                                     #
# ---------------------------------------------------------------------------- #
using  Catch22
include("featureset.jl")
export mode_5, mode_10, embedding_dist, acf_timescale, acf_first_min, ami2, trev, outlier_timing_pos
export outlier_timing_neg, whiten_timescale, forecast_error, ami_timescale, high_fluctuation, stretch_decreasing
export stretch_high, entropy_pairs, rs_range, dfa, low_freq_power, centroid_freq, transition_variance, periodicity
export base_set, catch9, catch22_set, complete_set

# utility from other packages
@reexport using SoleData: load_arff_dataset
@reexport using Random: seed!, Xoshiro, MersenneTwister

# ---------------------------------------------------------------------------- #
#                                 interfaces                                   #
# ---------------------------------------------------------------------------- #
@reexport using MLJ: Holdout, CV, StratifiedCV, TimeSeriesCV
include("partition.jl")
export partition

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

# export compute_results!, symbolic_analysis
# export get_algo, get_labels, get_predictions
# # export get_accuracy

# # import MLJ: predict, predict_mode, predict_mean
import SoleModels: apply
include("apply.jl")
include("train_test.jl")
export train_test

include("extractrules.jl")
include("symbolic_analysis.jl")
export symbolic_analysis

end
