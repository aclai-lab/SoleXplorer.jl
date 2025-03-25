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
    elseif y <: Reg_Value
        model_r = Symbol(model, "_regressor")
        haskey(AVAIL_MODELS, model_r) && return AVAIL_MODELS[model_r]()
    elseif y <: Cat_Value
        model_c = Symbol(model, "_classifier")
        haskey(AVAIL_MODELS, model_c) && return AVAIL_MODELS[model_c]()        
    end

    throw(ArgumentError("Model $model not found in available models"))
end

function validate_params(
    defaults::NamedTuple,
    users::Union{NamedTuple, Nothing},
    rng::Union{Nothing, AbstractRNG}
)        
    check_params(users, keys(defaults))

    if !isnothing(rng) && haskey(defaults, :rng)
        merge(defaults, filter_params(users), (rng=rng,))
    else
        merge(defaults, filter_params(users))
    end
end

function validate_features(
    defaults::AbstractVector,
    users::Union{AbstractVector, Nothing}
)
    features = isnothing(users) ? defaults : users

    # check if all features are functions
    all(f -> f isa Base.Callable, features) || throw(ArgumentError("All features must be functions"))

    return features
end

function validate_winparams(
    defaults::SoleFeatures.WinParams,
    users::Union{NamedTuple, Nothing},
    treatment::Symbol
)::SoleFeatures.WinParams
    check_params(users, (:type, :params))

    # get type
    user_type = get_type(users, SoleFeatures.AVAIL_WINS)

    # select the final type with proper priority: defaults -> globals -> users
    type = isnothing(user_type) ? defaults.type : user_type

    def_params = SoleFeatures.WIN_PARAMS[type]

    # validate parameters
    user_params = isnothing(users) ? NamedTuple() : (haskey(users, :params) ? begin
        check_params(users.params, keys(SoleFeatures.WIN_PARAMS[user_type]))
        NamedTuple(k => v for (k, v) in pairs(users.params))
    end : NamedTuple())

    params = merge(def_params, user_params)

    # ModalDecisionTrees package needs at least 3 windows to work properly
    if treatment == :reducesize && haskey(params, :nwindows)
        params.nwindows ≥ 3 || throw(ArgumentError("For :reducesize treatment, nwindows must be ≥ 3"))
    end

    return SoleFeatures.WinParams(type, params)
end

function validate_rulesparams(
    defaults::RulesParams,
    users::Union{NamedTuple, Nothing},
    rng::Union{Nothing, AbstractRNG}
)::RulesParams
    check_params(users, (:type, :params))

    # get type
    user_type = get_type(users, SoleXplorer.AVAIL_RULES)

    # select the final type with proper priority: defaults -> globals -> users
    type = isnothing(user_type) ? defaults.type : user_type

    def_params = SoleXplorer.RULES_PARAMS[type]

    # validate parameters
    user_params = isnothing(users) ? NamedTuple() : haskey(users, :params) ? begin
        check_params(users.params, keys(SoleXplorer.RULES_PARAMS[user_type]))
        NamedTuple(k => v for (k, v) in pairs(users.params))
    end : NamedTuple()

    params = isnothing(rng) && haskey(def_params, :rng) ?
        merge(def_params, user_params) :
        merge(def_params, user_params, (rng=rng,))

    return RulesParams(type, params)
end

function validate_tuning_type(
    defaults::Union{NamedTuple, Nothing},
    users::Union{NamedTuple, Nothing},
)
    user_tuning = get_function(users, AVAIL_TUNING_METHODS)
    
    type = if isnothing(user_tuning)
        isnothing(defaults) ? nothing : defaults.type
    else
        defaults = TUNING_METHODS_PARAMS[user_tuning]
        filtered_users = isnothing(users) ? nothing : NamedTuple(k => v for (k,v) in pairs(users) if k != :type)
        check_unknown_params(filtered_users, defaults, "user_tuning_type")
        haskey(users, :type) && users.type
    end

    filter_params(p) = isnothing(p) ? NamedTuple() : NamedTuple(k => v for (k,v) in pairs(p) if haskey(defaults, k))
    default_filtered = isnothing(defaults) ? NamedTuple() : NamedTuple(k => v for (k,v) in pairs(defaults) if k != :type)
    merged = merge(default_filtered, filter_params(users))

    isempty(merged) ? nothing : type(; merged...)
end

function validate_tuning_ranges(
    defaults::Union{AbstractVector, Nothing},
    users::Union{AbstractVector, Nothing}
)
    ranges = isnothing(users) ? defaults : users
  
    all(r -> r isa Base.Callable, ranges) || throw(ArgumentError("All ranges must be functions"))

    return ranges
end

function validate_tuning(
    defaults::NamedTuple,
    users::Union{NamedTuple, Bool, Nothing},
)
    if isa(users, Bool) 
        users = users ? NamedTuple() : nothing
    end

    isnothing(users) && return (tuning=false, method=nothing, params=NamedTuple(), ranges=nothing)

    method = validate_tuning_type(
        defaults.method,
        isnothing(users) ? nothing : get(users, :method, nothing)
    )

    params = validate_params(
        defaults.params,
        isnothing(users) ? nothing : get(users, :params, nothing),
        nothing
    )

    ranges = validate_tuning_ranges(
        defaults.ranges,
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
    model::NamedTuple,
    y::Union{DataType, Nothing},
    preprocess::Union{NamedTuple, Nothing}
)
    check_params(preprocess, PREPROC_KEYS)

    # grab rng form preprocess and feed it to every process
    rng = if isnothing(preprocess) 
        nothing
    else
        haskey(preprocess, :rng) ? preprocess.rng : nothing
    end

    haskey(model, :type) || throw(ArgumentError("Each model specification must contain a 'type' field"))
    check_params(model, MODEL_KEYS)
    modelset = validate_model(model.type, y)

    modelset.params = validate_params(
        modelset.params,
        get(model, :params, nothing),
        rng
    )

    # ModalDecisionTrees package needs features to be passed also in model params
    if isnothing(modelset.features)
        features = validate_features(
            modelset.params.features,
            haskey(model, :features) ? model.features : nothing
        )
        modelset.params = merge(model.params, (features=features,))
        modelset.features = features
    else
        modelset.features = validate_features(
            modelset.features,
            haskey(model, :features) ? model.features : nothing
        )
    end

    modelset.winparams = validate_winparams(
        modelset.winparams,
        get(model, :winparams, nothing),
        modelset.config.treatment
    )

    modelset.rulesparams = validate_rulesparams(
        modelset.rulesparams,
        get(model, :rulesparams, nothing),
        rng
    )

    modelset.tuning = validate_tuning(
        modelset.tuning,
        get(model, :tuning, nothing)
    )

    modelset.learn_method = isnothing(modelset.tuning.method) ? modelset.learn_method[1] : modelset.learn_method[2]
    modelset.preprocess = validate_preprocess_params(modelset.preprocess, preprocess)

    return modelset
end

validate_modelset(models::NamedTuple, y::Union{DataType, Nothing}) = validate_modelset(models, y, nothing)
