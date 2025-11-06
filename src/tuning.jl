# hyperparameter tuning infrastructure

# this module provides a unified interface for hyperparameter tuning in SoleXplorer,
# supporting multiple optimization strategies and seamless integration with MLJ's
# tuning framework.

# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
# abstract supertype for all hyperparameter tuning configurations
abstract type AbstractTuning end

# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
const EitherMeasures = Union{RobustMeasure, FussyMeasure}
const MaybeMeasure   = Maybe{EitherMeasures}

const RangeSpec = Union{
    Tuple,
    Tuple{Vararg{Tuple}},
    Vector{<:MLJ.NumericRange},
    MLJBase.NominalRange
}

"""
Hyperparameter tuning configuration with strategy, ranges, and evaluation settings.
"""
mutable struct Tuning{T} <: AbstractTuning
    "Parameter range specification from a tuning configuration."
    strategy   :: T
    "Tuning strategy from a tuning configuration."
    range      :: RangeSpec
    "Resampling strategy from a tuning configuration."
    resampling :: MLJ.ResamplingStrategy
    "Reference performance measure from a tuning configuration."
    measure    :: MaybeMeasure
    "Number of repetitions from a tuning configuration."
    repeats    :: Int64
    
    # convenience constructor for Tuning{T} that infers the type parameter
    Tuning(strategy::T, range, resampling=nothing, measure=nothing, repeats=1) where T = 
        Tuning{T}(strategy, range, resampling, measure, repeats)

    function Tuning{T}(strategy::T, range::RangeSpec, resampling, measure, repeats) where T
        repeats > 0 || throw(ArgumentError("repeats must be positive, got $repeats"))
        new{T}(strategy, range, resampling, measure, repeats)
    end
end

# ---------------------------------------------------------------------------- #
#                                   methods                                    #
# ---------------------------------------------------------------------------- #
get_range(t::Tuning)::RangeSpec = t.range
get_strategy(t::Tuning)::Any = t.strategy
get_resampling(t::Tuning)::MaybeResampling = t.resampling
get_measure(t::Tuning)::MaybeMeasure = t.measure
get_repeats(t::Tuning)::Int64 = t.repeats

# convert a Tuning configuration to a NamedTuple suitable for MLJ TunedModel construction
@inline tuning_params(t::Tuning) = (
    tuning     = get_strategy(t),
    range      = get_range(t), 
    resampling = get_resampling(t),
    measure    = get_measure(t),
    repeats    = get_repeats(t)
)

# ---------------------------------------------------------------------------- #
#                                    range                                     #
# ---------------------------------------------------------------------------- #
"""
    range(field::Union{Symbol,Expr}; kwargs...)

Wrapper for MLJ.range in hyperparameter tuning contexts.

# Arguments
- `field::Union{Symbol,Expr}`: Model field to tune
- `kwargs...`: Range specification arguments

# Returns
- Tuple of (field, kwargs) for later processing by tuning setup

This function provides a more convenient syntax for specifying hyperparameter
ranges that will be converted to proper MLJ ranges once the model is available.
"""
Base.range(field::Union{Symbol,Expr}; kwargs...) = field, kwargs...

# ---------------------------------------------------------------------------- #
#                             MLJ Tuning adapter                               #
# ---------------------------------------------------------------------------- #
# internal function to create tuning configurations with strategy-specific parameters
@inline function setup_tuning(
    strategy_type :: Type{<:Any};
    range         :: RangeSpec,
    resampling    :: MLJ.ResamplingStrategy=Holdout(fraction_train=0.7, shuffle=true),
    measure       :: MaybeMeasure=nothing,
    repeats       :: Int64=1,
    kwargs...
)::Tuning
    strategy = strategy_type(; kwargs...)
    return Tuning(strategy, range, resampling, measure, repeats)
end

"""
    GridTuning(; kwargs...)::Tuning

Create a grid search tuning configuration.
Parameters reference: [MLJTuning.Grid](https://juliaai.github.io/MLJ.jl/dev/tuning_models/#MLJTuning.Grid)
"""
const GridTuning(; kwargs...)::Tuning = setup_tuning(MLJ.Grid; kwargs...)

"""
    RandomTuning(; kwargs...)::Tuning

Create a random search tuning configuration.
Parameters reference: [MLJTuning.RandomSearch](https://juliaai.github.io/MLJ.jl/dev/tuning_models/#MLJTuning.RandomSearch)
"""
const RandomTuning(; kwargs...)::Tuning = setup_tuning(MLJ.RandomSearch; kwargs...)

"""
    CubeTuning(; kwargs...)::Tuning

Create a Latin hypercube sampling tuning configuration.
Parameters reference: [MLJTuning.LatinHypercube](https://juliaai.github.io/MLJ.jl/dev/tuning_models/#MLJTuning.LatinHypercube)
"""
const CubeTuning(; kwargs...)::Tuning = setup_tuning(MLJ.LatinHypercube; kwargs...)

"""
    ParticleTuning(; kwargs...)::Tuning

Create a particle swarm optimization tuning configuration.
Parameters reference: [MLJParticleSwarmOptimization](https://github.com/JuliaAI/MLJParticleSwarmOptimization.jl/)
"""
const ParticleTuning(; kwargs...)::Tuning = setup_tuning(PSO.ParticleSwarm; kwargs...)

"""
    AdaptiveTuning(; kwargs...)::Tuning

Create an adaptive particle swarm optimization tuning configuration.
Parameters reference: [MLJParticleSwarmOptimization](https://github.com/JuliaAI/MLJParticleSwarmOptimization.jl/)
"""
const AdaptiveTuning(; kwargs...)::Tuning = setup_tuning(PSO.AdaptiveParticleSwarm; kwargs...)


# ---------------------------------------------------------------------------- #
#                                show methods                                  #
# ---------------------------------------------------------------------------- #
function Base.show(io::IO, ::MIME"text/plain", t::Tuning{T}) where T
    println(io, "Tuning{", T, "}:")
    println(io, "  strategy:   ", t.strategy)
    println(io, "  range:      ", t.range)
    println(io, "  resampling: ", t.resampling)
    println(io, "  measure:    ", t.measure)
    print(io,   "  repeats:    ", t.repeats)
end

function Base.show(io::IO, t::Tuning{T}) where T
    print(io, "Tuning{", T, "}(", t.strategy, ", ..., repeats=", t.repeats, ")")
end
