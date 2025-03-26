module SoleXplorer

using Sole
using SoleData
using SoleData: PatchedFunction, nanpatchedfunction
using SoleFeatures
using SolePostHoc
using ModalDecisionTrees
using ModalDecisionLists
import MultiData.hasnans

using MLJ
using MLJDecisionTreeInterface, MLJXGBoostInterface
import MLJModelInterface as MMI

using MLJ: Grid as grid, RandomSearch as randomsearch, LatinHypercube as latinhypercube
using TreeParzen: MLJTreeParzenTuning as treeparzen
using MLJParticleSwarmOptimization: ParticleSwarm as particleswarm, AdaptiveParticleSwarm as adaptiveparticleswarm
using TreeParzen
using Distributions

export grid, randomsearch, latinhypercube, treeparzen, particleswarm, adaptiveparticleswarm

import DecisionTree as DT
import XGBoost as XGB

using DataFrames
using CategoricalArrays, OrderedCollections
using Random
using StatsBase, ScientificTypes

using Base.Threads: @threads

using Reexport
@reexport using SoleFeatures: mode_5, mode_10, embedding_dist, acf_timescale, acf_first_min, ami2, trev, outlier_timing_pos
@reexport using SoleFeatures: outlier_timing_neg, whiten_timescale, forecast_error, ami_timescale, high_fluctuation, stretch_decreasing
@reexport using SoleFeatures: stretch_high, entropy_pairs, rs_range, dfa, low_freq_power, centroid_freq, transition_variance, periodicity
@reexport using SoleFeatures: base_set, catch9, catch22_set, complete_set

@reexport using SoleFeatures: movingwindow, wholewindow, splitwindow, adaptivewindow

@reexport using SoleModels: PlainRuleExtractor
@reexport using SolePostHoc: InTreesRuleExtractor

@reexport using StatsBase: cov

# utility from other packages
@reexport using SoleData: load_arff_dataset
@reexport using Random: seed!, Xoshiro, MersenneTwister

@reexport using MLJ: CV, Holdout, StratifiedCV, TimeSeriesCV

include("utils/code_dataframe.jl")
export code_dataframe

include("interface.jl")
export Modelset, range
export plainrule, lumenrule, intreesrule

include("models/decisiontrees.jl")
include("models/modaldecisiontrees.jl")
include("models/xgboost.jl")
export makewatchlist

include("modules/models.jl")
export getmodel

include("modules/prepare_dataset.jl")
export prepare_dataset

include("modules/fit&test.jl")
export fitmodel, testmodel

include("modules/validate_modelsetup.jl")
export validate_modelset

include("modules/traintest.jl")
export traintest

include("user_interfaces/symbolic_analysis.jl")
export symbolic_analysis

include("user_interfaces/predict.jl")
export get_predict

end
