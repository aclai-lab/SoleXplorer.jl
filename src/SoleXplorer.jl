module SoleXplorer

using Sole
using SolePostHoc
import MultiData.hasnans
using SoleData
using SoleData: PatchedFunction, nanpatchedfunction

# Utilities
using DataFrames
using CategoricalArrays
using Random
using Accessors: @delete

import DecisionTree as DT
# import XGBoost as XGB

using MLJ
# using MLJBase: Probabilistic, ParamRange, train_test_pairs
using MLJTuning
using MLJDecisionTreeInterface
import MLJModelInterface as MMI

using MLJ: Grid as grid, RandomSearch as randomsearch, LatinHypercube as latinhypercube
using TreeParzen: MLJTreeParzenTuning as treeparzen
using MLJParticleSwarmOptimization: ParticleSwarm as particleswarm, AdaptiveParticleSwarm as adaptiveparticleswarm

# using MLJXGBoostInterface

# include("mlj/xgboost.jl")
# using .MLJXGBoostInterface
# export MLJXGBoostInterface, XGBoostRegressor, XGBoostClassifier, XGBoostCount

# include("utils/MLJAdaBoostModalInterface.jl")
# using .MLJAdaBoostModalInterface
# export AdaBoostModalClassifier

export grid, randomsearch, latinhypercube, treeparzen, particleswarm, adaptiveparticleswarm

using TreeParzen
using Distributions

# Learning algorithms
using ModalDecisionTrees
using ModalDecisionLists

# Features
using Catch22
using StatsBase
using ScientificTypes

include("utils/catch9.jl")
export mode_5, mode_10, embedding_dist, acf_timescale, acf_first_min, ami2, trev, outlier_timing_pos
export outlier_timing_neg, whiten_timescale, forecast_error, ami_timescale, high_fluctuation, stretch_decreasing
export stretch_high, entropy_pairs, rs_range, dfa, low_freq_power, centroid_freq, transition_variance, periodicity
export catch9

include("models_structs.jl")
export ModelConfig, range
include("models/decisiontrees.jl")
include("models/modaldecisiontrees.jl")

include("modules/models.jl")
export getmodel

include("modules/prepare_dataset.jl")
export prepare_dataset

include("modules/fit&test.jl")
export fitmodel, testmodel

include("user_interfaces/rules.jl")
export get_rules

include("user_interfaces/predict.jl")
export get_predict

include("modules/validate_modelset.jl")
export validate_modelset

include("symbolic_analysis.jl")
export symbolic_analysis, traintest

end
