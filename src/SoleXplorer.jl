module SoleXplorer

using Sole

using DataFrames
using CategoricalArrays
using Random

using MLJ
# using MLJBase: Probabilistic, ParamRange, train_test_pairs
using MLJTuning
using MLJDecisionTreeInterface
using MLJParticleSwarmOptimization: ParticleSwarm, AdaptiveParticleSwarm
using TreeParzen
using Distributions

using ModalDecisionTrees
using SoleDecisionTreeInterface

using Catch22
using StatsBase
# using SoleData
# using SoleData: PatchFunction, patchedfunction

using IterTools # da cancellare appena ritrovo le movingwindow che erano in SoleBase

include("utils/catch9.jl")
export mode_5, mode_10, embedding_dist, acf_timescale, acf_first_min, ami2, trev, outlier_timing_pos
export outlier_timing_neg, whiten_timescale, forecast_error, ami_timescale, high_fluctuation, stretch_decreasing
export stretch_high, entropy_pairs, rs_range, dfa, low_freq_power, centroid_freq, transition_variance, periodicity
export catch9

include("utils/worlds_filters.jl")
export absolute_movingwindow, absolute_splitwindow, realtive_movingwindow, relative_splitwindow

include("user_interfaces/models_interface.jl")
export ModelConfig, get_model

include("user_interfaces/tuning_interface.jl")
export get_tuning

include("user_interfaces/treatment_interface.jl")
export get_treatment

include("user_interfaces/partition_interface.jl")
export TTIdx, get_partition

include("user_interfaces/fit_interface.jl")
export get_fit

include("user_interfaces/test_interface.jl")
export get_test

include("user_interfaces/rules_interface.jl")
export get_rules

end
