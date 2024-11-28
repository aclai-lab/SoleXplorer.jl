const TUNEDMODEL_PARAMS = (;
    resampling=Holdout(),
    measure=LogLoss(tol = 2.22045e-16),
    weights=nothing,
    class_weights=nothing,
    repeats=1,
    operation=nothing,
    selection_heuristic= MLJTuning.NaiveSelection(nothing),
    n=nothing,
    train_best=true,
    acceleration=default_resource(),
    acceleration_resampling=CPU1(),
    check_measure=true,
    cache=true
)

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
        # MLJ.TunedModel(; model=model.classifier, tuning=tuning, ranges=ranges, tuning_kwargs...), 
        MLJ.TunedModel(; model=model.classifier, tuning=tuning, ranges=ranges, tuning_kwargs...), 
        model.ranges,
        model.data_treatment,
        model.default_treatment,
        model.params
    )
end
