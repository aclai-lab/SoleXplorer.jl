# adapters.jl — MLJ Package Integration Adapters
#
# This module provides adapter functions to integrate various MLJ ecosystem packages
# seamlessly within SoleXplorer workflows.
#
# MLJ Citation:
# Blaom, A. D., Kiraly, F., Lienart, T., Simonian, Y., Arenas, D., & Vollmer, S. J. (2020).
# MLJ: A Julia package for composable machine learning. Journal of Open Source Software, 5(55), 2704.
# https://doi.org/10.21105/joss.02704

# ---------------------------------------------------------------------------- #
#                      MLJ Resampling available functions                      #
# ---------------------------------------------------------------------------- #
const Availables_Resamplig_Funcs = (
    :Holdout,
    :CV,
    :StratifiedCV,
    :TimeSeriesCV,
)

# ---------------------------------------------------------------------------- #
#                        MLJ Tuning available functions                        #
# ---------------------------------------------------------------------------- #
const Available_Tuning_Funcs = (
    :Grid,
    :RandomSearch,
    :LatinHypercube,
)
const Extra_Tuning_Funcs = (
    :ParticleSwarm,
    :AdaptiveParticleSwarm,
)

# ---------------------------------------------------------------------------- #
#                       MLJ Balancing available functions                      #
# ---------------------------------------------------------------------------- #
# Current list is up to date to Imbalance package v0.1.6
# If a newer version has a new method, simply open a PR to add it.
# Imbalance package was created by Essam Wisam as a Google Summer of Code project,
# under the mentorship of Anthony Blaom.
# Special thanks also go to Rik Huijzer.
# https://github.com/JuliaAI
const Availables_Balancing_Funcs = (
    :BorderlineSMOTE1,
    :ClusterUndersampler,
    :ENNUndersampler,
    :ROSE,
    :RandomOversampler,
    :RandomUndersampler,
    :RandomWalkOversampler,
    :SMOTE,
    :SMOTEN,
    :SMOTENC,
    :TomekUndersampler,
)

# ---------------------------------------------------------------------------- #
#                                   Adapters                                   #
# ---------------------------------------------------------------------------- #
const Adapters = Dict(
    Availables_Resamplig_Funcs => MLJ,
    Available_Tuning_Funcs     => MLJTuning,
    Extra_Tuning_Funcs         => MLJParticleSwarmOptimization,
    Availables_Balancing_Funcs => Imbalance.MLJ,
)

for (funcs, adapter) in Adapters
    for func in funcs
        # skip if function is already globally defined,
        # ie: when MLJ is globally used
        isdefined(@__MODULE__, func) && continue

        sx_func = getfield(adapter, func)
        @eval begin
            function $(func)(args...; kwargs...)
                return $sx_func(args...; kwargs...)
            end
            export $(func)
        end
    end
end

# ---------------------------------------------------------------------------- #
#                               Range adapter                                  #
# ---------------------------------------------------------------------------- #
const RangeSpec = Union{
    Tuple,
    Tuple{Vararg{Tuple}},
    Vector{<:MLJ.NumericRange},
    MLJBase.NominalRange
}

function make_mlj_ranges(range::RangeSpec, model::MLJ.Model)
    range = range isa Tuple{Vararg{Tuple}} ? range : (range,)
    @show model
    collect(MLJ.range(model, r[1]; r[2:end]...) for r in range)
end

# wrapper for MLJ.range
Base.range(field::Union{Symbol,Expr}; kwargs...) = field, kwargs...