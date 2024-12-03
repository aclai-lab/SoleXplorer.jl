# ---------------------------------------------------------------------------- #
#                              available models                                #
# ---------------------------------------------------------------------------- #
const AVAIL_MODELS = Dict(
    :decision_tree => (
        method = MLJDecisionTreeInterface.DecisionTreeClassifier,

        params = (;
            max_depth=-1, 
            min_samples_leaf=1, 
            min_samples_split=2, 
            min_purity_increase=0.0, 
            n_subfeatures=0, 
            post_prune=false, 
            merge_purity_threshold=1.0, 
            display_depth=5, 
            feature_importance=:impurity, 
            rng=Random.TaskLocalRNG()
        ),
        data_treatment = :aggregate,
        default_treatment = whole,

        ranges = [
            model -> MLJ.range(model, :merge_purity_threshold, lower=0, upper=1),
            model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
        ]
    ),

    :modal_decision_tree => (
        method = ModalDecisionTree,

        params = (;
            max_depth=nothing, 
            min_samples_leaf=4, 
            min_purity_increase=0.002, 
            max_purity_at_leaf=Inf, 
            max_modal_depth=nothing, 
            relations=nothing, 
            features=nothing, 
            conditions=nothing, 
            featvaltype=Float64, 
            initconditions=nothing, 
            # downsize=SoleData.var"#downsize#482"(), 
            print_progress=false, 
            display_depth=nothing, 
            min_samples_split=nothing, 
            n_subfeatures=identity, 
            post_prune=false, 
            merge_purity_threshold=nothing, 
            feature_importance=:split,
            rng=Random.TaskLocalRNG()
        ),
        data_treatment = :reducesize,
        default_treatment = adaptive_moving_windows,

        ranges = [
            model -> MLJ.range(model, :merge_purity_threshold, lower=0, upper=1),
            model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
        ]
    )
)

mutable struct ModelConfig{T<:MLJ.Probabilistic, S<:Function}
    classifier::T
    mach::Union{MLJ.Machine, AbstractVector{MLJ.Machine}, Nothing}
    ranges::Vector{S}
    data_treatment::Symbol
    default_treatment::Base.Callable
    params::NamedTuple
end

# ---------------------------------------------------------------------------- #
#                                   get model                                  #
# ---------------------------------------------------------------------------- #
function get_model(model_name::Symbol; kwargs...)
    !haskey(AVAIL_MODELS, model_name) && throw(ArgumentError("Model $model_name not found in available models. Valid options are: $(keys(AVAIL_MODELS))"))

    kwargs_dict = Dict(kwargs)

    if model_name == :modal_decision_tree && haskey(kwargs_dict, :features)
        features = kwargs_dict[:features]
        X = kwargs_dict[:set]
        patched_f = collect(Iterators.flatten(SoleData.naturalconditions(X, [f]; fixcallablenans = true) for f in features))
        kwargs_dict[:features] = patched_f
    end

    params = AVAIL_MODELS[model_name].params
    valid_kwargs = filter(kv -> kv.first in keys(params), kwargs_dict)
    valid_params = merge(params, valid_kwargs)

    ModelConfig(
        AVAIL_MODELS[model_name].method(; valid_params...),
        nothing,
        AVAIL_MODELS[model_name].ranges,
        AVAIL_MODELS[model_name].data_treatment,
        AVAIL_MODELS[model_name].default_treatment,
        valid_params
    )
end
