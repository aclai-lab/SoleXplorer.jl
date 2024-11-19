module SoleXplorer

using Sole

using DataFrames
using CategoricalArrays
using Random

using MLJ
# using MLJBase: Probabilistic, ParamRange, train_test_pairs
# using MLJTuning
# using MLJDecisionTreeInterface
# using MLJParticleSwarmOptimization: ParticleSwarm, AdaptiveParticleSwarm
using TreeParzen
using Distributions

using ModalDecisionTrees
using SoleDecisionTreeInterface

using Catch22
using StatsBase
# using SoleData
# using SoleData: PatchFunction, patchedfunction

include("features_utils/catch9.jl")
export mode_5, mode_10, embedding_dist, acf_timescale, acf_first_min, ami2, trev, outlier_timing_pos
export outlier_timing_neg, whiten_timescale, forecast_error, ami_timescale, high_fluctuation, stretch_decreasing
export stretch_high, entropy_pairs, rs_range, dfa, low_freq_power, centroid_freq, transition_variance, periodicity
export catch9

end
