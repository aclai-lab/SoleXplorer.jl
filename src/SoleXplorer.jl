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
using MLJParticleSwarmOptimization: ParticleSwarm, AdaptiveParticleSwarm
using TreeParzen
using Distributions

using ModalDecisionTrees
using SoleDecisionTreeInterface

using Catch22
using StatsBase

using IterTools # da cancellare appena ritrovo le movingwindow che erano in SoleBase

export range

export grid, randomsearch, latinhypercube, treeparzen, particleswarm, adaptiveparticleswarm

using MLJ: Grid as grid, RandomSearch as randomsearch, LatinHypercube as latinhypercube
using TreeParzen: MLJTreeParzenTuning as treeparzen
using MLJParticleSwarmOptimization: ParticleSwarm as particleswarm, AdaptiveParticleSwarm as adaptiveparticleswarm

include("utils/catch9.jl")
export mode_5, mode_10, embedding_dist, acf_timescale, acf_first_min, ami2, trev, outlier_timing_pos
export outlier_timing_neg, whiten_timescale, forecast_error, ami_timescale, high_fluctuation, stretch_decreasing
export stretch_high, entropy_pairs, rs_range, dfa, low_freq_power, centroid_freq, transition_variance, periodicity
export catch9

include("utils/worlds_filters.jl")
export fixedlength_windows, whole
export absolute_movingwindow, absolute_splitwindow, relative_movingwindow, relative_splitwindow
export adaptive_moving_windows

include("user_interfaces/models.jl")
export ModelConfig, get_model

include("user_interfaces/tuning.jl")
export get_tuning

include("user_interfaces/treatment.jl")
export get_treatment

include("user_interfaces/partition.jl")
export TTIdx, get_partition

include("user_interfaces/fit.jl")
export get_fit

include("user_interfaces/test.jl")
export get_test

include("user_interfaces/rules.jl")
export get_rules

include("user_interfaces/predict.jl")
export get_predict

end
