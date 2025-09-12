# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
abstract type AbstractTuning end

# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
const MaybeResampling = Maybe{MLJ.ResamplingStrategy}
const MaybeMeasure = Maybe{EitherMeasure}
const MaybeInt = Maybe{Int}

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
    strategy::T
    range::RangeSpec
    resampling::MaybeResampling
    measure::MaybeMeasure
    repeats::Int64
    
    function Tuning{T}(strategy::T, range::RangeSpec, resampling, measure, repeats) where T
        repeats > 0 || throw(ArgumentError("repeats must be positive, got $repeats"))
        new{T}(strategy, range, resampling, measure, repeats)
    end
end

Tuning(strategy::T, range, resampling=nothing, measure=nothing, repeats=1) where T = 
    Tuning{T}(strategy, range, resampling, measure, repeats)

# ---------------------------------------------------------------------------- #
#                             MLJ Tuning adapter                               #
# ---------------------------------------------------------------------------- #
"""
    setup_tuning(strategy_type; range, kwargs...)

Create a tuning configuration with the specified strategy type.
"""
@inline function setup_tuning(
    strategy_type::Type{<:Any};
    range::Union{Tuple, Tuple{Vararg{Tuple}}, MLJBase.NominalRange},
    resampling::MaybeResampling=nothing,
    measure::MaybeMeasure=nothing,
    repeats::Int64=1,
    kwargs...
)::Tuning
    strategy = strategy_type(; kwargs...)
    return Tuning(strategy, range, resampling, measure, repeats)
end

# ---------------------------------------------------------------------------- #
#                             MLJ Tuning adapter                               #
# ---------------------------------------------------------------------------- #
const GridTuning(; kwargs...)::Tuning     = setup_tuning(MLJ.Grid; kwargs...)
const RandomTuning(; kwargs...)::Tuning   = setup_tuning(MLJ.RandomSearch; kwargs...)
const CubeTuning(; kwargs...)::Tuning     = setup_tuning(MLJ.LatinHypercube; kwargs...)
const ParticleTuning(; kwargs...)::Tuning = setup_tuning(PSO.ParticleSwarm; kwargs...)
const AdaptiveTuning(; kwargs...)::Tuning = setup_tuning(PSO.AdaptiveParticleSwarm; kwargs...)

"""Enable splatting and iteration over Tuning struct."""
Base.propertynames(::Tuning) = (:strategy, :range, :resampling, :measure, :repeats)
Base.getproperty(t::Tuning, s::Symbol) = getfield(t, s)

@inline tuning_params(t::Tuning) = (
    strategy = t.strategy,
    range = t.range, 
    resampling = t.resampling,
    measure = t.measure,
    repeats = t.repeats
)

Base.pairs(t::Tuning) = pairs(tuning_params(t))

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
