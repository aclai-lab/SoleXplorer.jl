module SoleXplorer

using Sole
import MultiData.hasnans
using SoleData
using SoleData: PatchedFunction, nanpatchedfunction

using DataFrames
using CategoricalArrays
using Random

using MLJ
# using MLJBase: Probabilistic, ParamRange, train_test_pairs
using MLJTuning
using MLJDecisionTreeInterface
import MLJModelInterface as MMI

import MLJ: Grid as grid, RandomSearch as randomsearch, LatinHypercube as latinhypercube
using TreeParzen: MLJTreeParzenTuning as treeparzen
using MLJParticleSwarmOptimization: ParticleSwarm as particleswarm, AdaptiveParticleSwarm as adaptiveparticleswarm

import XGBoost as XGB

include("mlj/xgboost.jl")
using .MLJXGBoostInterface
export MLJXGBoostInterface, XGBoostRegressor, XGBoostClassifier, XGBoostCount

export grid, randomsearch, latinhypercube, treeparzen, particleswarm, adaptiveparticleswarm

using TreeParzen
using Distributions

using ModalDecisionTrees
using ModalDecisionLists
using SoleDecisionTreeInterface

using Catch22
using StatsBase

include("utils/catch9.jl")
export mode_5, mode_10, embedding_dist, acf_timescale, acf_first_min, ami2, trev, outlier_timing_pos
export outlier_timing_neg, whiten_timescale, forecast_error, ami_timescale, high_fluctuation, stretch_decreasing
export stretch_high, entropy_pairs, rs_range, dfa, low_freq_power, centroid_freq, transition_variance, periodicity
export catch9

include("utils/avail_models.jl")

include("user_interfaces/models.jl")
export ModelConfig, range, get_model

include("user_interfaces/preprocess.jl")
export preprocess_dataset

include("user_interfaces/fit.jl")
export modelfit

include("user_interfaces/test.jl")
export modeltest

include("user_interfaces/rules.jl")
export get_rules

include("user_interfaces/predict.jl")
export get_predict

end
