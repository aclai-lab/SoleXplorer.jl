# ---------------------------------------------------------------------------- #
#                                 model struct                                 #
# ---------------------------------------------------------------------------- #
mutable struct ModelConfig{T}
    classifier::MLJ.Model
    mach::Union{MLJ.Machine, AbstractVector{MLJ.Machine}, Nothing}
    rules::AbstractVector{T}
    model_algo::Symbol
    learn_method::Function
    tune_learn_method::Function
    apply_tuning::Bool
    ranges::AbstractVector{Function}
    data_treatment::Symbol
    features::AbstractVector{<:Base.Callable}
<<<<<<< Updated upstream
    nested_treatment::NamedTuple #{Base.Callable, NamedTuple}
    # nested_treatment_params::NamedTuple
    rules_method::Function
=======
    treatment::NamedTuple #{Base.Callable, NamedTuple}
    # treatment_params::NamedTuple
    rules_method::SoleModels.RuleExtractor
>>>>>>> Stashed changes
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
    haskey(AVAIL_MODELS, model_name) || throw(ArgumentError("Model $model_name not found in available models. Valid options are: $(keys(AVAIL_MODELS))"))

    kwargs_dict = Dict(kwargs)
    apply_tuning = false

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
        AVAIL_MODELS[model_name].features,
        AVAIL_MODELS[model_name].treatment,
        # AVAIL_MODELS[model_name].treatment_params,
        AVAIL_MODELS[model_name].rules_method,
    )
end
