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
function validate_model(model::Symbol, y::DataType)::ModelSetup
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

    !isnothing(rng) && haskey(defaults, :rng) ?
        merge(defaults, filter_params(users), (rng=rng,)) :
        merge(defaults, filter_params(users))

end

function validate_features(
    defaults::AbstractVector,
    users::Union{Tuple, Nothing}
)
    features = isnothing(users) ? defaults : [users...]

    # check if all features are functions
    all(f -> f isa Base.Callable, features) || throw(ArgumentError("All features must be functions"))

    return features
end

function validate_resample(
    users::Union{NamedTuple, Nothing},
    rng::Union{AbstractRNG, Nothing}=nothing
)::Union{Resample, Nothing}    
    check_params(users, (:type, :params))
    type = get_type(users, SoleXplorer.AVAIL_RESAMPLES)
    def_params = SoleXplorer.RESAMPLE_PARAMS[type]

    # validate parameters
    user_params = isnothing(users) ? NamedTuple() : haskey(users, :params) ? begin
        check_params(users.params, keys(SoleXplorer.RESAMPLE_PARAMS[type]))
        NamedTuple(k => v for (k, v) in pairs(users.params))
    end : NamedTuple()

    params = !isnothing(rng) && haskey(def_params, :rng) ?
        merge(def_params, user_params, (rng=rng,)) :
        merge(def_params, user_params)
        
    return Resample(type, params)
end

function validate_winparams(
    defaults::SoleFeatures.WinParams,
    users::Union{NamedTuple, Nothing},
    treatment::Symbol
)::SoleFeatures.WinParams
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

    params = !isnothing(rng) && haskey(def_params, :rng) ?
        merge(def_params, user_params, (rng=rng,)) :
        merge(def_params, user_params)
        
    return RulesParams(type, params)
end

function validate_tuning_type(
    users::Union{NamedTuple, Nothing},
    rng::Union{Nothing, AbstractRNG}
)::TuningStrategy
    check_params(users, (:type, :params))
    type = get_type(users, SoleXplorer.AVAIL_TUNING_METHODS)
    def_params = SoleXplorer.TUNING_METHODS_PARAMS[type]

    # validate parameters
    user_params = isnothing(users) ? NamedTuple() : haskey(users, :params) ? begin
        check_params(users.params, keys(SoleXplorer.TUNING_METHODS_PARAMS[type]))
        NamedTuple(k => v for (k, v) in pairs(users.params))
    end : NamedTuple()

    params = !isnothing(rng) && haskey(def_params, :rng) ?
        merge(def_params, user_params, (rng=rng,)) :
        merge(def_params, user_params)
        
    return TuningStrategy(type, params)
end

function validate_tuning(
    defaults::TuningParams,
    users::Union{NamedTuple, Bool, Nothing},
    rng::Union{Nothing, AbstractRNG},
    algo::Symbol
)::Union{TuningParams, Nothing}
    isnothing(users) && return nothing

    # case 1: users is a Bool
    if isa(users, Bool) && users
        method = !isnothing(rng) && haskey(TUNING_METHODS_PARAMS[defaults.method.type], :rng) ?
            SoleXplorer.TuningStrategy(defaults.method.type, merge(defaults.method.params, (rng=rng,))) :
            SoleXplorer.TuningStrategy(defaults.method.type, defaults.method.params)
        return SoleXplorer.TuningParams(method, defaults.params, defaults.ranges)
    end

    # case 2: users is a NamedTuple
    check_params(users, (:method, :params, :ranges))
    # verify all required fields exist
    required_fields = (:method, :ranges)
    missing_fields = filter(field -> !haskey(users, field), required_fields)
    isempty(missing_fields) || throw(ArgumentError("Missing required tuning fields: $missing_fields"))

    method = validate_tuning_type(
        users.method,
        rng
    )

    params = validate_params(
        SoleXplorer.TUNING_PARAMS[algo],
        get(users, :params, nothing),
        nothing
    )

    all(r -> r isa Base.Callable, users.ranges) || throw(ArgumentError("All ranges must be functions"))

    return TuningParams(method, params, users.ranges)
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
    y::Union{DataType, Nothing};
    resample::Union{NamedTuple, Nothing}=nothing,
    win::Union{NamedTuple, Nothing}=nothing,
    features::Union{Tuple, Nothing}=nothing,
    tuning::Union{NamedTuple, Bool, Nothing}=nothing,
    rules::Union{NamedTuple, Nothing}=nothing,
    preprocess::Union{NamedTuple, Nothing}=nothing
)::ModelSetup
    check_params(model, (:type, :params))
    check_params(resample, (:type, :params))
    check_params(win, (:type, :params))
    check_params(rules, (:type, :params))
    check_params(preprocess, PREPROC_KEYS)

    # grab rng form preprocess and feed it to every process
    rng = if isnothing(preprocess) 
        nothing
    else
        haskey(preprocess, :rng) ? preprocess.rng : nothing
    end

    haskey(model, :type) || throw(ArgumentError("Each model specification must contain a 'type' field"))
    modelset = validate_model(model.type, y)

    # grab additional extra params
    user_params = get(model, :params, nothing)
    if !isnothing(user_params) && haskey(user_params, :reducefunc)
        modelset.config = merge(modelset.config, (reducefunc = user_params.reducefunc,))
        user_params = NamedTuple(k => v for (k, v) in pairs(user_params) if k != :reducefunc)
    end
    modelset.params = validate_params(
        modelset.params,
        user_params,
        rng
    )

    # ModalDecisionTrees package needs features to be passed also in model params
    if isnothing(modelset.features)
        features = validate_features(
            modelset.params.features,
            features
        )
        modelset.params = merge(model.params, (features=features,))
        modelset.features = features
    else
        modelset.features = validate_features(
            modelset.features,
            features
        )
    end

    isnothing(resample) || (modelset.resample = validate_resample(resample, rng))

    modelset.winparams = validate_winparams(
        modelset.winparams,
        win,
        modelset.config.treatment
    )

    modelset.rulesparams = validate_rulesparams(
        modelset.rulesparams,
        rules,
        rng
    )

    modelset.tuning = validate_tuning(
        modelset.tuning,
        tuning,
        rng,
        modelset.config.algo
    )

    modelset.learn_method = isnothing(modelset.tuning) ? modelset.learn_method[1] : modelset.learn_method[2]
    modelset.preprocess = validate_preprocess_params(modelset.preprocess, preprocess)

    return modelset
end
