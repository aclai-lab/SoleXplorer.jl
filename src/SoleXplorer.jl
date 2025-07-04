module SoleXplorer

using SoleData
using SoleData: PatchedFunction, nanpatchedfunction
using SoleModels
using SoleModels: AbstractModel, DecisionList, DecisionForest, DecisionEnsemble, DecisionSet
using SolePostHoc
# using ModalDecisionTrees
# using ModalDecisionLists
import MultiData.hasnans

# using Reexport
# using MLJ
# @reexport using MLJ: Grid as grid, RandomSearch as randomsearch, LatinHypercube as latinhypercube
# using TreeParzen: Config
# @reexport using TreeParzen: MLJTreeParzenTuning as treeparzen
# @reexport using MLJParticleSwarmOptimization: ParticleSwarm as particleswarm, AdaptiveParticleSwarm as adaptiveparticleswarm

# using MLJDecisionTreeInterface, MLJXGBoostInterface
# import MLJModelInterface as MMI

# import DecisionTree as DT
# import XGBoost as XGB

using DataFrames
# using OrderedCollections
# using Random

# using Base.Threads: @threads

using Random
using Reexport

# ---------------------------------------------------------------------------- #
#                                     MLJ                                      #
# ---------------------------------------------------------------------------- #
using MLJ, MLJBase

# classification measures
@reexport using MLJ: accuracy, confusion_matrix, kappa, log_loss

# regression measures
@reexport using MLJ: rms, l1, l2, mae, mav

# ---------------------------------------------------------------------------- #
#                                    utils                                     #
# ---------------------------------------------------------------------------- #
using Catch22
include("utils/featureset.jl")
export mode_5, mode_10, embedding_dist, acf_timescale, acf_first_min, ami2, trev, outlier_timing_pos
export outlier_timing_neg, whiten_timescale, forecast_error, ami_timescale, high_fluctuation, stretch_decreasing
export stretch_high, entropy_pairs, rs_range, dfa, low_freq_power, centroid_freq, transition_variance, periodicity
export base_set, catch9, catch22_set, complete_set

# utility from other packages
# @reexport using SoleData: load_arff_dataset
# @reexport using Random: seed!, Xoshiro, MersenneTwister

# ---------------------------------------------------------------------------- #
#                                 interfaces                                   #
# ---------------------------------------------------------------------------- #
include("interfaces/base_interface.jl")
export modeltype

@reexport using SoleBase: movingwindow, wholewindow, splitwindow, adaptivewindow
include("interfaces/windowing_interface.jl")

include("interfaces/dataset_interface.jl")
export Dataset, get_X, get_y, get_tt, get_info, get_Xtrain, get_Xvalid, get_Xtest, get_ytrain, get_yvalid, get_ytest

@reexport using MLJ: CV, Holdout, StratifiedCV, TimeSeriesCV
include("interfaces/resample_interface.jl")

@reexport using MLJ: Grid as grid, RandomSearch as randomsearch, LatinHypercube as latinhypercube
@reexport using MLJParticleSwarmOptimization: ParticleSwarm as particleswarm, AdaptiveParticleSwarm as adaptiveparticleswarm
include("interfaces/tuning_interface.jl")
export range

include("interfaces/extractrules_interface.jl")

include("interfaces/model_interface.jl")
export Modelset

include("interfaces/measures_interface.jl")

# ---------------------------------------------------------------------------- #
#                                   models                                     #
# ---------------------------------------------------------------------------- #
import DecisionTree as DT
using MLJDecisionTreeInterface
include("models/decisiontrees.jl")

using ModalDecisionTrees
const MDT = ModalDecisionTrees
include("models/modaldecisiontrees.jl")

using XGBoost
const XGB = XGBoost
using MLJXGBoostInterface
include("models/xgboost.jl")
export makewatchlist

include("utils/validate_modelsetup.jl")

# ---------------------------------------------------------------------------- #
#                                   modules                                    #
# ---------------------------------------------------------------------------- #
include("modules/prepare_dataset.jl")
export code_dataset
export prepare_dataset

include("modules/train_test.jl")
export train_test

include("modules/symbolic_analysis.jl")
export compute_results!, symbolic_analysis
export get_algo, get_labels, get_predictions
# export get_accuracy

# import MLJ: predict, predict_mode, predict_mean
include("utils/apply.jl")
# export get_predict

end
