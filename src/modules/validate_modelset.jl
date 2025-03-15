# ---------------------------------------------------------------------------- #
#                                  utilities                                   #
# ---------------------------------------------------------------------------- #
function check_params(
    params::Union{NamedTuple, Nothing},
    allowed_keys::Tuple
)
    isnothing(params) && return

    unknown_keys = setdiff(keys(params), allowed_keys)
    isempty(unknown_keys) || throw(ArgumentError("Unknown fields: $unknown_keys"))
end

filter_params(p) = isnothing(p) ? NamedTuple() : p

function get_type(
    params::Union{NamedTuple, Nothing},
    avail_types::Tuple
)
    isnothing(params) && return nothing
    p_type = get(params, :type, nothing)
    isnothing(p_type) && return nothing

    p_type ∈ avail_types || throw(ArgumentError("Type $p_type not found in available types"))
    return p_type
end

# function check_unknown_params(params, default_params, source)
#     isnothing(params) && return
#     unknown = setdiff(keys(params), keys(default_params))
#     isempty(unknown) || throw(ArgumentError("Unknown parameters in $source: $unknown"))
#     return nothing
# end

function get_function(params, avail_functions)
    isnothing(params) && return
    funcs = filter(v -> v in avail_functions, values(params))
    length(funcs) > 1 && throw(ArgumentError("Multiple methods detected. Only one method is allowed."))
    return isempty(funcs) ? nothing : first(funcs)
end

# ---------------------------------------------------------------------------- #
#                             validating functions                             #
# ---------------------------------------------------------------------------- #
function validate_model(model::Symbol, y::DataType)
    if haskey(AVAIL_MODELS, model)
        return AVAIL_MODELS[model]()
    elseif eltype(y) <: Number
        model_r = Symbol(model, "_regressor")
        haskey(AVAIL_MODELS, model_r) && return AVAIL_MODELS[model_r]()
    elseif eltype(y) <: Categorical
        model_c = Symbol(model, "_classifier")
        haskey(AVAIL_MODELS, model_c) && return AVAIL_MODELS[model_c]()        
    end

    throw(ArgumentError("Model $model not found in available models"))
end

function validate_params(
    defaults::NamedTuple,
    globals::Union{NamedTuple, Nothing},
    users::Union{NamedTuple, Nothing},
    rng::Union{Nothing, AbstractRNG}
)        
    check_params(globals, keys(defaults))
    check_params(users, keys(defaults))

    isnothing(rng) && haskey(defaults, :rng) ?
        merge(defaults, filter_params(globals), filter_params(users)) :
        merge(defaults, filter_params(globals), filter_params(users), (rng=rng,))
end

function validate_features(
    defaults::AbstractVector,
    globals::Union{AbstractVector, Nothing},
    users::Union{AbstractVector, Nothing}
)
    # select features with proper priority: defaults -> globals -> users
    features = isnothing(users) ? isnothing(globals) ? defaults : globals : users

    # check if all features are functions
    all(f -> f isa Base.Callable, features) || throw(ArgumentError("All features must be functions"))

    return features
end

function validate_winparams(
    defaults::SoleFeatures.WinParams,
    globals::Union{NamedTuple, Nothing},
    users::Union{NamedTuple, Nothing},
    treatment::Symbol
)::SoleFeatures.WinParams
    # check if globals and users are valid TypeParams
    check_params(globals, (:type, :params))
    check_params(users, (:type, :params))

    # get type
    global_type = get_type(globals, SoleFeatures.AVAIL_WINS)
    user_type = get_type(users, SoleFeatures.AVAIL_WINS)

    # select the final type with proper priority: defaults -> globals -> users
    type = isnothing(user_type) ? isnothing(global_type) ? defaults.type : global_type : user_type

    def_params = SoleFeatures.WIN_PARAMS[type]

    # validate parameters
    global_params = isnothing(globals) ? NamedTuple() : (haskey(globals, :params) ? begin
        check_params(globals.params, keys(SoleFeatures.WIN_PARAMS[global_type]))
        NamedTuple(k => v for (k, v) in pairs(globals.params))
    end : NamedTuple())
    user_params = isnothing(users) ? NamedTuple() : (haskey(users, :params) ? begin
        check_params(users.params, keys(SoleFeatures.WIN_PARAMS[user_type]))
        NamedTuple(k => v for (k, v) in pairs(users.params))
    end : NamedTuple())

    params = merge(
        def_params,
        global_params,
        user_params
    )

    # ModalDecisionTrees package needs at least 3 windows to work properly
    if treatment == :reducesize && haskey(params, :nwindows)
        params.nwindows ≥ 3 || throw(ArgumentError("For :reducesize treatment, nwindows must be ≥ 3"))
    end

    return SoleFeatures.WinParams(type, params)
end

function validate_rulesparams(
    defaults::RulesParams,
    globals::Union{NamedTuple, Nothing},
    users::Union{NamedTuple, Nothing}
)::RulesParams
    # check if globals and users are valid TypeParams
    check_params(globals, (:type, :params))
    check_params(users, (:type, :params))

    # get type
    global_type = get_type(globals, SoleXplorer.AVAIL_RULES)
    user_type = get_type(users, SoleXplorer.AVAIL_RULES)

    # select the final type with proper priority: defaults -> globals -> users
    type = isnothing(user_type) ? isnothing(global_type) ? defaults.type : global_type : user_type

    def_params = SoleXplorer.RULES_PARAMS[type]

    # validate parameters
    global_params = isnothing(globals) ? NamedTuple() : haskey(globals, :params) ? begin
        check_params(globals.params, keys(SoleXplorer.RULES_PARAMS[global_type]))
        NamedTuple(k => v for (k, v) in pairs(globals.params))
    end : NamedTuple()
    user_params = isnothing(users) ? NamedTuple() : haskey(users, :params) ? begin
        check_params(users.params, keys(SoleXplorer.RULES_PARAMS[user_type]))
        NamedTuple(k => v for (k, v) in pairs(users.params))
    end : NamedTuple()

    params = merge(
        def_params,
        global_params,
        user_params
    )

    return RulesParams(type, params)
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
    check_params(preprocess, keys(defaults))
    return merge(defaults, preprocess)
end

function validate_modelset(
    models::AbstractVector{<:NamedTuple},
    y::Union{DataType, Nothing},
    globals::Union{NamedTuple, Nothing},
    preprocess::Union{NamedTuple, Nothing}
)
    # check globals and preprocess keys
    check_params(globals, MODEL_KEYS)
    check_params(preprocess, PREPROC_KEYS)

    # grab rng form preprocess and feed it to every process
    rng = if isnothing(preprocess) 
        nothing
    else
        haskey(preprocess, :rng) ? preprocess.rng : nothing
    end

    modelsets = SymbolicModelSet[]

    for m in models
        haskey(m, :type) || throw(ArgumentError("Each model specification must contain a 'type' field"))
        check_params(m, MODEL_KEYS)
        model = validate_model(m.type, y)

        model.params = validate_params(
            model.params,
            isnothing(globals) ? nothing : get(globals, :params, nothing),
            get(m, :params, nothing),
            rng
        )

        # ModalDecisionTrees package needs features to be passed also in model params
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

        model.rulesparams = validate_rulesparams(
            model.rulesparams,
            isnothing(globals) ? nothing : get(globals, :rulesparams, nothing),
            get(m, :rulesparams, nothing)
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

validate_modelset(models::AbstractVector{<:NamedTuple}, y::Union{DataType, Nothing}, globals::Union{NamedTuple, Nothing}) = validate_modelset(models, y, globals, nothing)
validate_modelset(models::AbstractVector{<:NamedTuple}, y::Union{DataType, Nothing}) = validate_modelset(models, y, nothing, nothing)

validate_modelset(models::NamedTuple, args...) = validate_modelset([models], args...)