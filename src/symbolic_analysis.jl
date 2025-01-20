function check_unknown_params(params, default_params, source)
    isnothing(params) && return
    unknown = setdiff(keys(params), keys(default_params))
    isempty(unknown) || throw(ArgumentError("Unknown parameters in $source: $unknown"))
end

function get_function(params, avail_functions)
    isnothing(params) && return
    funcs = filter(v -> v in avail_functions, values(params))
    length(funcs) > 1 && throw(ArgumentError("Multiple window functions detected. Only one window function is allowed."))
    return isempty(funcs) ? nothing : first(funcs)
end

function validate_model(model::Symbol)
    haskey(AVAIL_MODELS, model) || throw(ArgumentError("Model $model not found in available models"))
    return AVAIL_MODELS[model]()
end

function validate_params(
    defaults::NamedTuple,
    globals::Union{NamedTuple, Nothing},
    users::Union{NamedTuple, Nothing},
)        
    check_unknown_params(globals, defaults, "globals")
    check_unknown_params(users, defaults, "users")

    filter_params(p) = isnothing(p) ? NamedTuple() : NamedTuple(k => v for (k,v) in pairs(p) if haskey(defaults, k))

    merge(defaults, filter_params(globals), filter_params(users))
end

function validate_features(
    defaults::AbstractVector{<:Base.Callable},
    globals::Union{AbstractVector{<:Base.Callable}, Nothing},
    users::Union{AbstractVector{<:Base.Callable}, Nothing}
)
    features = if isnothing(users) && isnothing(globals)
        defaults
    elseif isnothing(users)
        globals
    else
        users
    end

    all(f -> f isa Base.Callable, features) || throw(ArgumentError("All features must be functions"))

    return features
end

function validate_winparams(
    defaults::NamedTuple,
    globals::Union{NamedTuple, Nothing},
    users::Union{NamedTuple, Nothing},
    treatment::Symbol
)
    global_win = get_function(globals, AVAIL_WINS)
    user_win = get_function(users, AVAIL_WINS)
    
    type = if isnothing(user_win) && isnothing(global_win)
        defaults.type
    elseif isnothing(user_win)
        defaults = WIN_PARAMS[global_win]
        filtered_globals = isnothing(globals) ? nothing : NamedTuple(k => v for (k,v) in pairs(globals) if k != :type)
        check_unknown_params(filtered_globals, defaults, "global_winparams")
        haskey(globals, :type) && globals.type
    else
        defaults = WIN_PARAMS[user_win]
        filtered_users = isnothing(users) ? nothing : NamedTuple(k => v for (k,v) in pairs(users) if k != :type)
        check_unknown_params(filtered_users, defaults, "user_winparams")
        haskey(users, :type) && users.type
    end

    filter_params(p) = isnothing(p) ? NamedTuple() : NamedTuple(k => v for (k,v) in pairs(p) if haskey(defaults, k))

    params = merge(
        NamedTuple(k => v for (k,v) in pairs(defaults) if k != :type),
        filter_params(globals),
        filter_params(users)
    )

    if treatment == :reducesize && haskey(params, :nwindows)
        params.nwindows ≥ 3 || throw(ArgumentError("For :reducesize treatment, nwindows must be ≥ 3"))
    end

    return (type = type, params = params)
end

function validate_tuning_type(
    defaults::Union{NamedTuple, Nothing},
    globals::Union{NamedTuple, Nothing},
    users::Union{NamedTuple, Nothing},
)
    global_tuning = get_function(globals, AVAIL_TUNING_METHODS)
    user_tuning = get_function(users, AVAIL_TUNING_METHODS)
    
    type = if isnothing(user_tuning) && isnothing(global_tuning)
        isnothing(defaults) ? nothing : defaults.type
    elseif isnothing(user_tuning)
        defaults = TUNING_METHODS_PARAMS[global_tuning]
        filtered_globals = isnothing(globals) ? nothing : NamedTuple(k => v for (k,v) in pairs(globals) if k != :type)
        check_unknown_params(filtered_globals, defaults, "global_tuning_type")
        haskey(globals, :type) && globals.type
    else
        defaults = TUNING_METHODS_PARAMS[user_tuning]
        filtered_users = isnothing(users) ? nothing : NamedTuple(k => v for (k,v) in pairs(users) if k != :type)
        check_unknown_params(filtered_users, defaults, "user_tuning_type")
        haskey(users, :type) && users.type
    end

    filter_params(p) = isnothing(p) ? NamedTuple() : NamedTuple(k => v for (k,v) in pairs(p) if haskey(defaults, k))
    default_filtered = isnothing(defaults) ? NamedTuple() : NamedTuple(k => v for (k,v) in pairs(defaults) if k != :type)
    merged = merge(default_filtered, filter_params(globals), filter_params(users))

    isempty(merged) ? nothing : type(; merged...)
end

function validate_tuning_ranges(
    defaults::Union{AbstractVector{<:Base.Callable}, Nothing},
    globals::Union{AbstractVector{<:Base.Callable}, Nothing},
    users::Union{AbstractVector{<:Base.Callable}, Nothing}
)
    ranges = if isnothing(users) && isnothing(globals)
        defaults
    else
        isnothing(users) ? globals : users
    end
  
    all(r -> r isa Base.Callable, ranges) || throw(ArgumentError("All ranges must be functions"))

    return ranges
end

function validate_tuning(
    defaults::NamedTuple,
    globals::Union{NamedTuple, Bool, Nothing},
    users::Union{NamedTuple, Bool, Nothing},
)
    isa(globals, Bool) && (globals = NamedTuple())
    isa(users, Bool) && (users = NamedTuple())
    
    isnothing(globals) && isnothing(users) && return (tuning=false, method=nothing, params=NamedTuple(), ranges=nothing)

    method = validate_tuning_type(
        defaults.method,
        isnothing(globals) ? nothing : get(globals, :method, nothing),
        isnothing(users) ? nothing : get(users, :method, nothing)
    )

    params = validate_params(
        defaults.params,
        isnothing(globals) ? nothing : get(globals, :params, nothing),
        isnothing(users) ? nothing : get(users, :params, nothing)
    )

    ranges = validate_tuning_ranges(
        defaults.ranges,
        isnothing(globals) ? nothing : get(globals, :ranges, nothing),
        isnothing(users) ? nothing : get(users, :ranges, nothing)
    )

    return (tuning=true, method=method, params=params, ranges=ranges)
end

function validate_modelset(
    models::AbstractVector{<:NamedTuple}, 
    globals::Union{NamedTuple, Nothing}=nothing,
)
    modelsets = AbstractModelSet[]

    for m in models
        haskey(m, :type) || throw(ArgumentError("Each model specification must contain a 'type' field"))
        model = validate_model(m.type)

        model.params = validate_params(
            model.params,
            isnothing(globals) ? nothing : get(globals, :params, nothing),
            get(m, :params, nothing)
        )

        # ModalDecisionTrees needs features to be passed also in model params
        if isnothing(model.features)
            features = validate_features(
                model.params.features,
                isnothing(globals) ? nothing : haskey(globals, :features) ? globals.features : nothing,
                haskey(m, :features) ? m.features : nothing
            )
            model.params = merge(model.params, (features=features,))
            model.features = features
        else
            model.features = validate_features(
                model.features,
                isnothing(globals) ? nothing : haskey(globals, :features) ? globals.features : nothing,
                haskey(m, :features) ? m.features : nothing
            )
        end

        model.winparams = validate_winparams(
            model.winparams,
            isnothing(globals) ? nothing : get(globals, :winparams, nothing),
            get(m, :winparams, nothing),
            model.config.treatment
        )

        model.tuning = validate_tuning(
            model.tuning,
            isnothing(globals) ? nothing : get(globals, :tuning, nothing),
            get(m, :tuning, nothing)
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
    globals::Union{NamedTuple, Nothing}=nothing,
    kwargs...
)
    modelsets = validate_modelset(models, globals)

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
