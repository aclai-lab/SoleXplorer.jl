# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
abstract type AbstractTuning end

# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
const EitherMeasures  = Union{RobustMeasure, FussyMeasure}
const MaybeResampling = Maybe{MLJ.ResamplingStrategy}
const MaybeMeasure   = Maybe{EitherMeasures}

const RangeSpec = Union{
    Tuple,
    Tuple{Vararg{Tuple}},
    Vector{<:MLJ.NumericRange},
    MLJBase.NominalRange
}

# ---------------------------------------------------------------------------- #
#                                Tuning struct                                 #
# ---------------------------------------------------------------------------- #
mutable struct Tuning{T} <: AbstractTuning
    strategy   :: T
    range      :: RangeSpec
    resampling :: MaybeResampling
    measure    :: MaybeMeasure
    repeats    :: Int64
    
    function Tuning{T}(strategy::T, range::RangeSpec, resampling, measure, repeats) where T
        repeats > 0 || throw(ArgumentError("repeats must be positive, got $repeats"))
        new{T}(strategy, range, resampling, measure, repeats)
    end
end

Tuning(strategy::T, range, resampling=nothing, measure=nothing, repeats=1) where T = 
    Tuning{T}(strategy, range, resampling, measure, repeats)

# ---------------------------------------------------------------------------- #
#                                   methods                                    #
# ---------------------------------------------------------------------------- #
Base.propertynames(::Tuning) = (:strategy, :range, :resampling, :measure, :repeats)
Base.getproperty(t::Tuning, s::Symbol) = getfield(t, s)

get_range(t::Tuning)      = t.range
get_strategy(t::Tuning)   = t.strategy
get_resampling(t::Tuning) = t.resampling
get_measure(t::Tuning)    = t.measure
get_repeats(t::Tuning)    = t.repeats

@inline tuning_params(t::Tuning) = (
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
@inline function setup_tuning(
    strategy_type :: Type{<:Any};
    range         :: RangeSpec,
    resampling    :: MaybeResampling=nothing,
    measure       :: MaybeMeasure=nothing,
    repeats       :: Int64=1,
    kwargs...
)::Tuning
    strategy = strategy_type(; kwargs...)
    return Tuning(strategy, range, resampling, measure, repeats)
end

const GridTuning(; kwargs...)::Tuning     = setup_tuning(MLJ.Grid; kwargs...)
const RandomTuning(; kwargs...)::Tuning   = setup_tuning(MLJ.RandomSearch; kwargs...)
const CubeTuning(; kwargs...)::Tuning     = setup_tuning(MLJ.LatinHypercube; kwargs...)
const ParticleTuning(; kwargs...)::Tuning = setup_tuning(PSO.ParticleSwarm; kwargs...)
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
