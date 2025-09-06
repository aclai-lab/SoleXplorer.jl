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
