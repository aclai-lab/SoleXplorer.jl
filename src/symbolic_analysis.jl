function check_unknown_params(params, default_params, source)
    isnothing(params) && return
    unknown = setdiff(keys(params), keys(default_params))
    !isempty(unknown) && throw(ArgumentError("Unknown parameters in $source: $unknown"))
end

function get_function(params, avail_functions)
    isnothing(params) && return nothing
    funcs = filter(v -> v in avail_functions, collect(values(params)))
    length(funcs) > 1 && throw(ArgumentError("Multiple window functions detected. Only one window function is allowed."))
    return isempty(funcs) ? nothing : funcs[1]
end

function validate_model(model::Symbol)
    haskey(AVAIL_MODELS, model) || throw(ArgumentError("Model $model not found in available models"))
    return AVAIL_MODELS[model]()
end

function validate_params(
    default_params::NamedTuple,
    global_params::Union{NamedTuple, Nothing},
    user_params::Union{NamedTuple, Nothing},
)        
    check_unknown_params(global_params, default_params, "global_params")
    check_unknown_params(user_params, default_params, "user_params")

    filter_valid_params(params) = isnothing(params) ? NamedTuple() : NamedTuple(k => v for (k,v) in pairs(params) if haskey(default_params, k))
    global_filt_params = filter_valid_params(global_params)
    user_filt_params = filter_valid_params(user_params)

    merge(default_params, global_filt_params, user_filt_params)
end

function validate_features(
    default_features::AbstractVector{<:Base.Callable},
    global_features::Union{AbstractVector{<:Base.Callable}, Nothing},
    user_features::Union{AbstractVector{<:Base.Callable}, Nothing}
)
    features = if isnothing(user_features) && isnothing(global_features)
        default_features
    elseif isnothing(user_features)
        global_features
    else
        user_features
    end

    all(f -> f isa Base.Callable, features) || throw(ArgumentError("All features must be functions"))

    return features
end

function validate_winparams(
    default_params::NamedTuple,
    global_params::Union{NamedTuple, Nothing},
    user_params::Union{NamedTuple, Nothing},
)
    global_win = get_function(global_params, AVAIL_WINS)
    user_win = get_function(user_params, AVAIL_WINS)
    
    type = if isnothing(user_win) && isnothing(global_win)
        default_params.type
    elseif isnothing(user_win)
        default_params = WIN_PARAMS[global_win]
        filtered_global_params = isnothing(global_params) ? nothing : NamedTuple(k => v for (k,v) in pairs(global_params) if k != :type)
        check_unknown_params(filtered_global_params, default_params, "global_winparams")
        haskey(global_params, :type) && global_params.type
    else
        default_params = WIN_PARAMS[user_win]
        filtered_user_params = isnothing(user_params) ? nothing : NamedTuple(k => v for (k,v) in pairs(user_params) if k != :type)
        check_unknown_params(filtered_user_params, default_params, "user_winparams")
        haskey(user_params, :type) && user_params.type
    end

    filter_valid_params(params) = isnothing(params) ? NamedTuple() : NamedTuple(k => v for (k,v) in pairs(params) if haskey(default_params, k))
    filtered_default_params = NamedTuple(k => v for (k,v) in pairs(default_params) if k != :type)
    global_filt_params = filter_valid_params(global_params)
    user_filt_params = filter_valid_params(user_params)

    return (type = type, params = merge(filtered_default_params, global_filt_params, user_filt_params))
end

function validate_tuning_type(
    default_type::Union{NamedTuple, Nothing},
    global_type::Union{NamedTuple, Nothing},
    user_type::Union{NamedTuple, Nothing},
)
    global_tuning = get_function(global_type, AVAIL_TUNING_METHODS)
    user_tuning = get_function(user_type, AVAIL_TUNING_METHODS)
    
    type = if isnothing(user_tuning) && isnothing(global_tuning)
        isnothing(default_type) ? nothing : default_type.type
    elseif isnothing(user_tuning)
        default_type = TUNING_METHODS_PARAMS[global_tuning]
        filtered_global_type = isnothing(global_type) ? nothing : NamedTuple(k => v for (k,v) in pairs(global_type) if k != :type)
        check_unknown_params(filtered_global_type, default_type, "global_tuning_type")
        haskey(global_type, :type) && global_type.type
    else
        default_type = TUNING_METHODS_PARAMS[user_tuning]
        filtered_user_type = isnothing(user_type) ? nothing : NamedTuple(k => v for (k,v) in pairs(user_type) if k != :type)
        check_unknown_params(filtered_user_type, default_type, "user_tuning_type")
        haskey(user_type, :type) && user_type.type
    end

    filter_valid_params(params) = isnothing(params) ? NamedTuple() : NamedTuple(k => v for (k,v) in pairs(params) if haskey(default_type, k))
    filtered_default_params = isnothing(default_type) ? NamedTuple() : NamedTuple(k => v for (k,v) in pairs(default_type) if k != :type)
    global_filt_params = filter_valid_params(global_type)
    user_filt_params = filter_valid_params(user_type)

    merged_params = merge(filtered_default_params, global_filt_params, user_filt_params)

    isempty(merged_params) ? nothing : type(; merged_params...)
end

function validate_tuning_ranges(
    default_ranges::Union{AbstractVector{<:Base.Callable}, Nothing},
    global_ranges::Union{AbstractVector{<:Base.Callable}, Nothing},
    user_ranges::Union{AbstractVector{<:Base.Callable}, Nothing}
)
    ranges = if isnothing(user_ranges) && isnothing(global_ranges)
        default_ranges
    elseif isnothing(user_ranges)
        global_ranges
    else
        user_ranges
    end
    
    all(r -> r isa Base.Callable, ranges) || throw(ArgumentError("All ranges must be functions"))

    return ranges
end

function validate_tuning(
    default_tuning::NamedTuple,
    global_tuning::Union{NamedTuple, Bool, Nothing},
    user_tuning::Union{NamedTuple, Bool, Nothing},
)
    if isa(global_tuning, Bool) 
        global_tuning = NamedTuple()
    end

    if isa(user_tuning, Bool)
        user_tuning = NamedTuple()
    end

    if !isnothing(global_tuning) || !isnothing(user_tuning)
        method = validate_tuning_type(
            default_tuning.method,
            isnothing(global_tuning) ? nothing : haskey(global_tuning, :method) ? global_tuning.method : nothing,
            isnothing(user_tuning) ? nothing : haskey(user_tuning, :method) ? user_tuning.method : nothing,    
        )

        params = validate_params(
            default_tuning.params,
            isnothing(global_tuning) ? nothing : haskey(global_tuning, :params) ? global_tuning.params : nothing,
            isnothing(user_tuning) ? nothing : haskey(user_tuning, :params) ? user_tuning.params : nothing,
        )

        ranges = validate_tuning_ranges(
            default_tuning.ranges,
            isnothing(global_tuning) ? nothing : haskey(global_tuning, :ranges) ? global_tuning.ranges : nothing,
            isnothing(user_tuning) ? nothing : haskey(user_tuning, :ranges) ? user_tuning.ranges : nothing,
        )

        return (tuning=true, method=method, params=params, ranges=ranges)
    else
        return (tuning=false, method=nothing, params=NamedTuple(), ranges=nothing)
    end  
end

function validate_modelset(
    models::AbstractVector{<:NamedTuple}, 
    global_params::Union{NamedTuple, Nothing}=nothing,
)
    modelsets = AbstractModelSet[]

    for m in models
        haskey(m, :type) || throw(ArgumentError("Each model specification must contain a 'type' field"))
        model = validate_model(m.type)

        model.params = validate_params(
            model.params,
            isnothing(global_params) ? nothing : haskey(global_params, :params) ? global_params.params : nothing,
            haskey(m, :params) ? m.params : nothing
        )

        model.features = validate_features(
            model.features,
            isnothing(global_params) ? nothing : haskey(global_params, :features) ? global_params.features : nothing,
            haskey(m, :features) ? m.features : nothing
        )

        model.winparams = validate_winparams(
            model.winparams,
            isnothing(global_params) ? nothing : haskey(global_params, :winparams) ? global_params.winparams : nothing,
            haskey(m, :winparams) ? m.winparams : nothing
        )

        model.tuning = validate_tuning(
            model.tuning,
            isnothing(global_params) ? nothing : haskey(global_params, :tuning) ? global_params.tuning : nothing,
            haskey(m, :tuning) ? m.tuning : nothing
        )

        model.learn_method = isnothing(model.tuning.method) ? model.learn_method[1] : model.learn_method[2]

        push!(modelsets, model)
    end

    return modelsets
end

"""
symbolic_analysis()

esempio di utilizzo:

models = symbolic_analysis(X, y;
    model=(
        type=:decision_tree, 
        params=(max_depth=3, min_samples_leaf=14), 
        winparams=(wintype=movingwindow, windowssize=1024), 
        features=(minimum, mean, StatsBase.cov, mode_5)
    ),
    model=(
        type=:decision_tree, 
        params=(min_samples_leaf=30), 
    ),
    global_params=(
        features=(std,)
    )
)

da notare: i parametri specificati nel model sovrascrivono i global_params.

"""

function _symbolic_analysis(
    X::AbstractDataFrame, 
    y::Union{AbstractVector, Nothing}; 
    models::AbstractVector{<:NamedTuple}, 
    global_params::Union{NamedTuple, Nothing}=nothing,
    kwargs...
)
    modelsets = validate_modelset(models, global_params)

    models = ModelConfig[]

    for m in modelsets
        ds = preprocess_dataset(X, y, m)

        classifier = get_model(m, ds)

        mach = modelfit(m, classifier, ds);
        model = modeltest(m, mach, ds);

        rules = get_rules(m, model, ds);
        accuracy = get_predict(mach, ds);

        push!(models, ModelConfig(m, ds, classifier, mach, model, rules, accuracy))
    end
    return models
end

function symbolic_analysis(
    X::AbstractDataFrame, 
    y::AbstractVector; 
    models::Union{NamedTuple, AbstractVector{<:NamedTuple}, Nothing}=nothing, 
    kwargs...
)
    check_dataframe_type(X) || throw(ArgumentError("DataFrame must contain only numeric values"))
    size(X, 1) == length(y) || throw(ArgumentError("Number of rows in DataFrame must match length of class labels"))

    isnothing(models) && throw(ArgumentError("At least one type must be specified"))

    if isa(models, NamedTuple)
        _symbolic_analysis(X, y; models=[models], kwargs...)
    else
        _symbolic_analysis(X, y; models=models, kwargs...)
    end
end
