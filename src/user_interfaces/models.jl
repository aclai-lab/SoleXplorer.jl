# ---------------------------------------------------------------------------- #
#                                 model struct                                 #
# ---------------------------------------------------------------------------- #
mutable struct ModelConfig{T<:MLJ.Model, S<:Function}
    classifier::T
    mach::Union{MLJ.Machine, AbstractVector{MLJ.Machine}, Nothing}
    model_type::UnionAll
    learn_method::S
    tune_learn_method::S
    apply_tuning::Bool
    ranges::Vector{S}
    data_treatment::Symbol
    default_features::AbstractVector{<:Base.Callable}
    default_treatment::Base.Callable
    treatment_params::NamedTuple
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
    model_name::Symbol;
    tuning::Union{T, Nothing}=nothing,
    ranges::Union{S, AbstractVector{S}, Nothing}=nothing,
    kwargs...
) where {T<:MLJTuning.TuningStrategy, S<:Base.Callable}
    !haskey(AVAIL_MODELS, model_name) && throw(ArgumentError("Model $model_name not found in available models. Valid options are: $(keys(AVAIL_MODELS))"))
    kwargs_dict = Dict(kwargs)
    apply_tuning = false

    if model_name == :modal_decision_tree && haskey(kwargs_dict, :features)
        features = kwargs_dict[:features]
        X = kwargs_dict[:set]
        patched_f = collect(Iterators.flatten(SoleData.naturalconditions(X, [f]; fixcallablenans = true) for f in features))
        kwargs_dict[:features] = patched_f
    end

    params = AVAIL_MODELS[model_name].model_params
    valid_kwargs = filter(kv -> kv.first in keys(params), kwargs_dict)
    model_params = merge(params, valid_kwargs)

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
    
    ModelConfig(
        classifier,
        nothing,
        AVAIL_MODELS[model_name].model_type,
        AVAIL_MODELS[model_name].learn_method,
        AVAIL_MODELS[model_name].tune_learn_method,
        apply_tuning,
        AVAIL_MODELS[model_name].ranges,
        AVAIL_MODELS[model_name].data_treatment,
        AVAIL_MODELS[model_name].default_features,
        AVAIL_MODELS[model_name].default_treatment,
        AVAIL_MODELS[model_name].treatment_params,
    )
end
