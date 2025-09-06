# # ---------------------------------------------------------------------------- #
# #                               abstract types                                 #
# # ---------------------------------------------------------------------------- #
# abstract type AbstractBalancing end

# # ---------------------------------------------------------------------------- #
# #                               Balancing struct                               #
# # ---------------------------------------------------------------------------- #
# mutable struct Balancing{T} <: AbstractBalancing
#     strategy::T
#     range::RangeSpec
#     resampling::MaybeResampling
#     measure::MaybeMeasure
#     repeats::Int64
    
#     function Balancing{T}(strategy::T, range::RangeSpec, resampling, measure, repeats) where T
#         repeats > 0 || throw(ArgumentError("repeats must be positive, got $repeats"))
#         new{T}(strategy, normalize_range(range), resampling, measure, repeats)
#     end
# end

# Balancing(strategy::T, range, resampling=nothing, measure=nothing, repeats=1) where T = 
#     Balancing{T}(strategy, range, resampling, measure, repeats)

# ---------------------------------------------------------------------------- #
#                            MLJ Balancing adapters                            #
# ---------------------------------------------------------------------------- #
const BorderlineSMOTE1(; kwargs...)      = Imbalance.MLJ.BorderlineSMOTE1(; kwargs...)
    @doc (@doc Imbalance.MLJ.BorderlineSMOTE1) BorderlineSMOTE1
const ClusterUndersampler(; kwargs...)   = Imbalance.MLJ.ClusterUndersampler(; kwargs...)
    @doc (@doc Imbalance.MLJ.ClusterUndersampler) ClusterUndersampler
const ENNUndersampler(; kwargs...)       = Imbalance.MLJ.ENNUndersampler(; kwargs...)
    @doc (@doc Imbalance.MLJ.ENNUndersampler) ENNUndersampler
const ROSE(; kwargs...)                  = Imbalance.MLJ.ROSE(; kwargs...)
    @doc (@doc Imbalance.MLJ.ROSE) ROSE
const RandomUndersampler(; kwargs...)    = Imbalance.MLJ.RandomUndersampler(; kwargs...)
    @doc (@doc Imbalance.MLJ.RandomUndersampler) RandomUndersampler
const RandomWalkOversampler(; kwargs...) = Imbalance.MLJ.RandomWalkOversampler(; kwargs...)
    @doc (@doc Imbalance.MLJ.RandomWalkOversampler) RandomWalkOversampler
const SMOTE(; kwargs...)                 = Imbalance.MLJ.SMOTE(; kwargs...)
    @doc (@doc Imbalance.MLJ.SMOTE) SMOTE
const SMOTEN(; kwargs...)                = Imbalance.MLJ.SMOTEN(; kwargs...)
    @doc (@doc Imbalance.MLJ.SMOTEN) SMOTEN
const SMOTENC(; kwargs...)               = Imbalance.MLJ.SMOTENC(; kwargs...)
    @doc (@doc Imbalance.MLJ.SMOTENC) SMOTENC
const TomekUndersampler(; kwargs...)     = Imbalance.MLJ.TomekUndersampler(; kwargs...)
    @doc (@doc Imbalance.MLJ.TomekUndersampler) TomekUndersampler
