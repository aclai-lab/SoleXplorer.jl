# ---------------------------------------------------------------------------- #
#                       MLJ Resampling available functions                      #
# ---------------------------------------------------------------------------- #
const Availables_Resamplig_Funcs = (
    :Holdout,
    :CV,
    :StratifiedCV,
    :TimeSeriesCV,
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
    Availables_Balancing_Funcs => Imbalance.MLJ
)

for (funcs, adapter) in Adapters
    for func in funcs
        # skip if function is already globally defined,
        # ie: MLJ is globally used
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