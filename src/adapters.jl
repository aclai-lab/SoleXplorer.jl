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
    :TomekUndersampler
)

for func in Availables_Balancing_Funcs
    imbalance_func = getfield(Imbalance.MLJ, func)
    @eval begin
        function $(func)(args...; kwargs...)
            return $imbalance_func(args...; kwargs...)
        end
        export $(func)
    end
end