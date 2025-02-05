function check_unknown_params(params, default_params, source)
    isnothing(params) && return
    unknown = setdiff(keys(params), keys(default_params))
    isempty(unknown) || throw(ArgumentError("Unknown parameters in $source: $unknown"))
    return nothing
end

function get_function(params, avail_functions)
    isnothing(params) && return
    funcs = filter(v -> v in avail_functions, values(params))
    length(funcs) > 1 && throw(ArgumentError("Multiple methods detected. Only one method is allowed."))
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
    defaults::AbstractVector,
    globals::Union{AbstractVector, Nothing},
    users::Union{AbstractVector, Nothing}
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
        # filtered_globals = isnothing(globals) ? nothing : NamedTuple(k => v for (k,v) in pairs(globals) if k != :type)
        filtered_globals = @delete globals.type
        check_unknown_params(filtered_globals, defaults, "global_winparams")
        haskey(globals, :type) && globals.type
    else
        defaults = WIN_PARAMS[user_win]
        # filtered_users = isnothing(users) ? nothing : NamedTuple(k => v for (k,v) in pairs(users) if k != :type)
        filtered_users = @delete users.type
        check_unknown_params(filtered_users, defaults, "user_winparams")
        haskey(users, :type) && users.type
    end

    filter_params(p) = isnothing(p) ? NamedTuple() : NamedTuple(k => v for (k,v) in pairs(p) if haskey(defaults, k))

    params = merge(
        # NamedTuple(k => v for (k,v) in pairs(defaults) if k != :type),
        (@delete defaults.type),
        filter_params(globals),
        filter_params(users)
    )

    if treatment == :reducesize && haskey(params, :nwindows)
        params.nwindows ≥ 3 || throw(ArgumentError("For :reducesize treatment, nwindows must be ≥ 3"))
    end

    return (type = type, params...)
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
    defaults::Union{AbstractVector, Nothing},
    globals::Union{AbstractVector, Nothing},
    users::Union{AbstractVector, Nothing}
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
    if isa(globals, Bool) 
        globals = globals ? NamedTuple() : nothing
    end
    if isa(users, Bool) 
        users = users ? NamedTuple() : nothing
    end

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

function validate_preprocess_params(
    defaults::NamedTuple,
    preprocess::Union{NamedTuple, Nothing}
)
    isnothing(preprocess) && return defaults
    check_unknown_params(preprocess, defaults, "preprocess")
    return merge(defaults, preprocess)
end

function validate_modelset(
    models::AbstractVector{<:NamedTuple}, 
    globals::Union{NamedTuple, Nothing},
    preprocess::Union{NamedTuple, Nothing},
)
    modelsets = SymbolicModelSet[]

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

        model.preprocess = validate_preprocess_params(model.preprocess, preprocess)

        push!(modelsets, model)
    end

    return modelsets
end

validate_modelset(models::AbstractVector{<:NamedTuple}, globals::Union{NamedTuple, Nothing}) = validate_modelset(models, globals, nothing)
validate_modelset(models::AbstractVector{<:NamedTuple}) = validate_modelset(models, nothing, nothing)

validate_modelset(models::NamedTuple, args...) = validate_modelset([models], args...)