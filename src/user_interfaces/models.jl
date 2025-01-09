using SoleModels

# ---------------------------------------------------------------------------- #
#                                 model struct                                 #
# ---------------------------------------------------------------------------- #
mutable struct ModelConfig{T}
    classifier::MLJ.Model
    mach::Union{MLJ.Machine, AbstractVector{MLJ.Machine}, Nothing}
    models::AbstractVector{T}
    model_algo::Symbol
    learn_method::Function
    tune_learn_method::Function
    apply_tuning::Bool
    ranges::AbstractVector{Function}
    data_treatment::Symbol
    features::AbstractVector{<:Base.Callable}
    nested_treatment::NamedTuple #{Base.Callable, NamedTuple}
    # nested_treatment_params::NamedTuple
    rules_method::SoleModels.RuleExtractor
end

# ---------------------------------------------------------------------------- #
#                                    tuning                                    #
# ---------------------------------------------------------------------------- #
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

function range(
    field::Union{Expr, Symbol};
    lower::Union{AbstractFloat, Int, Nothing}=nothing,
    upper::Union{AbstractFloat, Int, Nothing}=nothing,
    origin::Union{AbstractFloat, Int, Nothing}=nothing,
    unit::Union{AbstractFloat, Int, Nothing}=nothing,
    scale::Union{Symbol, Nothing}=nothing,
    values::Union{AbstractVector, Nothing}=nothing,
)
    return function(model)
        MLJ.range(
            model,
            field;
            lower=lower,
            upper=upper,
            origin=origin,
            unit=unit,
            scale=scale,
            values=values
        )
    end
end

# ---------------------------------------------------------------------------- #
#                                   get model                                  #
# ---------------------------------------------------------------------------- #
function get_model(
    model::ModelConfig,
)
    model
end

function get_model(
    model_name::Symbol;
    tuning::Union{T, Nothing}=nothing,
    ranges::Union{S, AbstractVector{S}, Nothing}=nothing,
    kwargs...
) where {T<:MLJTuning.TuningStrategy, S<:Base.Callable}
    !haskey(AVAIL_MODELS, model_name) && throw(ArgumentError("Model $model_name not found in available models. Valid options are: $(keys(AVAIL_MODELS))"))
    kwargs_dict = Dict(kwargs)
    apply_tuning = false

    params = AVAIL_MODELS[model_name].model_params
    valid_kwargs = filter(kv -> kv.first in keys(params), kwargs_dict)
    model_params = merge(params, valid_kwargs)

    # TODO verifica che in modale passi il parametro relations AI7
    classifier = AVAIL_MODELS[model_name].method(; model_params...)

    if !isnothing(tuning)
        apply_tuning = true
        tuning_kwargs = merge(TUNEDMODEL_PARAMS, filter(kv -> kv.first in keys(TUNEDMODEL_PARAMS), kwargs))

        if isnothing(ranges)
            ranges = [r(model.classifier) for r in model.ranges]
        else
            user_ranges = ranges isa AbstractVector ? ranges : [ranges]
            ranges = [r(classifier) for r in user_ranges]
        end

        classifier = MLJ.TunedModel(; 
            model=classifier, 
            tuning=tuning, 
            ranges=ranges, 
            tuning_kwargs...
        )
    end
    
    ModelConfig{AVAIL_MODELS[model_name].model.type}(
        classifier,
        nothing,
        [],
        AVAIL_MODELS[model_name].model.algo,
        AVAIL_MODELS[model_name].learn_method,
        AVAIL_MODELS[model_name].tune_learn_method,
        apply_tuning,
        AVAIL_MODELS[model_name].ranges,
        AVAIL_MODELS[model_name].data_treatment,
        AVAIL_MODELS[model_name].nested_features,
        AVAIL_MODELS[model_name].nested_treatment,
        # AVAIL_MODELS[model_name].treatment_params,
        AVAIL_MODELS[model_name].rules_method,
    )
end
