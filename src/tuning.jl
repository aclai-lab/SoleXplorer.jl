# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
# abstract type AbstractTuningStrategy end
abstract type AbstractTuning end

# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
# const MaybeAbstractTuningStrategy = Maybe{AbstractTuningStrategy}

const MaybeResampling = Maybe{MLJ.ResamplingStrategy}
const MaybeMeasure = Maybe{EitherMeasure}
const MaybeInt = Maybe{Int}


# ---------------------------------------------------------------------------- #
#                                Tuning struct                                 #
# ---------------------------------------------------------------------------- #
mutable struct Tuning{T} <: AbstractTuning
    strategy::T
    range::Union{Tuple{Vararg{Tuple}}, Vector{<:MLJ.NumericRange}, MLJBase.NominalRange}
    resampling::MaybeResampling
    measure::MaybeMeasure
    repeats::Int64
end

# ---------------------------------------------------------------------------- #
#                             MLJ Tuning adapter                               #
# ---------------------------------------------------------------------------- #
GridTuning(; kwargs...)::Tuning     = setup_tuning(MLJ.Grid; kwargs...)
RandomTuning(; kwargs...)::Tuning   = setup_tuning(MLJ.RandomSearch; kwargs...)
CubeTuning(; kwargs...)::Tuning     = setup_tuning(MLJ.LatinHypercube; kwargs...)
ParticleTuning(; kwargs...)::Tuning = setup_tuning(PSO.ParticleSwarm; kwargs...)
AdaptiveTuning(; kwargs...)::Tuning = setup_tuning(PSO.AdaptiveParticleSwarm; kwargs...)

function setup_tuning(
    tuning::DataType;
    range::Union{Tuple, Tuple{Vararg{Tuple}}, MLJBase.NominalRange},
    resampling::MaybeResampling=nothing,
    measure::MaybeMeasure=nothing,
    repeats::Int64=1,
    kwargs...
)::Tuning
    strategy = tuning(; kwargs...)
    if !(range isa MLJ.NominalRange)
        range isa Tuple{Vararg{Tuple}} || (range=(range,))
    end
    Tuning{typeof(strategy)}(strategy, range, resampling, measure, repeats)
end
