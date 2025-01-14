# ---------------------------------------------------------------------------- #
#                               get tuning model                               #
# ---------------------------------------------------------------------------- #
function get_tuning(model::T, tuning::S; 
    ranges::Union{Nothing, MLJ.ParamRange, AbstractVector{<:MLJ.ParamRange}}=nothing,
    kwargs...
) where {T<:SoleXplorer.ModelConfig, S<:MLJTuning.TuningStrategy}
    # strategy_kwargs = filter(kv -> kv.first in keys(AVAIL_TUNINGS[tuning].params), kwargs)
    tuning_kwargs = merge(TUNEDMODEL_PARAMS, filter(kv -> kv.first in keys(TUNEDMODEL_PARAMS), kwargs))

    isnothing(ranges) && (ranges = [r(model.classifier) for r in model.ranges])
    ranges isa MLJ.ParamRange && (ranges = [ranges])
    
    ModelConfig( 
        MLJ.TunedModel(; model=model.classifier, tuning=tuning, ranges=ranges, tuning_kwargs...),
        nothing,
        model.ranges,
        model.data_treatment,
        model.features,
        model.treatment,
        model.treatment_params
    )
end
