# ---------------------------------------------------------------------------- #
#                                  utilities                                   #
# ---------------------------------------------------------------------------- #
function check_params(
    params::OptNamedTuple,
    allowed_keys::Tuple
)
    params === nothing && return
    unknown_keys = setdiff(keys(params), allowed_keys)
    isempty(unknown_keys) || throw(ArgumentError("Unknown fields: $unknown_keys"))
end

function check_params(
    params::OptNamedTuple,
    allowed_keys::NamedTuple
)
    params === nothing && return
    unknown_keys = setdiff(keys(params), keys(allowed_keys))
    isempty(unknown_keys) || throw(ArgumentError("Unknown fields: $unknown_keys"))
end

filter_params(p) = p === nothing ? NamedTuple() : p

function get_type(
    params::OptNamedTuple,
    avail_types::Tuple
)
    params === nothing && return nothing
    p_type = get(params, :type, nothing)
    p_type === nothing && return nothing

    p_type ∈ avail_types || throw(ArgumentError("Type :$p_type not found in available types"))
    return p_type
end

function check_user_params(
    users::OptNamedTuple,
    default_params::Dict
)
    users === nothing ? NamedTuple() : (haskey(users, :params) ? begin
        check_params(users.params, keys(default_params[users.type]))
        NamedTuple(k => v for (k, v) in pairs(users.params))
    end : NamedTuple())
end

function merge_params(
    defaults::NamedTuple,
    users::OptNamedTuple,
    rng::Union{AbstractRNG, Nothing}=nothing
)
    !(rng === nothing) && haskey(defaults, :rng) ?
        merge(defaults, filter_params(users), (rng=rng,)) :
        merge(defaults, filter_params(users))
end

# ---------------------------------------------------------------------------- #
#                             validating functions                             #
# ---------------------------------------------------------------------------- #
function validate_model(model::Symbol, y::DataType)::ModelSetup
    if haskey(AVAIL_MODELS, model)
        return AVAIL_MODELS[model]()
    elseif y <: SoleModels.RLabel
        model_r = Symbol(model, "_regressor")
        haskey(AVAIL_MODELS, model_r) && return AVAIL_MODELS[model_r]()
    elseif y <: SoleModels.CLabel
        model_c = Symbol(model, "_classifier")
        haskey(AVAIL_MODELS, model_c) && return AVAIL_MODELS[model_c]()        
    end

    throw(ArgumentError("Model $model not found in available models"))
end

function validate_params(
    defaults::NamedTuple,
    users::OptNamedTuple,
    rng::Union{AbstractRNG, Nothing}
)::NamedTuple     
    check_params(users, keys(defaults))
    merge_params(defaults, users, rng)
end

function validate_features(
    defaults::OptVecCall,
    users::OptTuple
)
    features = users === nothing ? defaults : [users...]

    # check if all features are functions
    all(f -> f isa Base.Callable, features) || throw(ArgumentError("All features must be functions"))

    return features
end

function validate_measures(
    users::OptTuple
)
    measures = users === nothing ? nothing : users

    # check if all measures are functions
    # all(f -> f isa Base.Callable, measures) || throw(ArgumentError("All measures must be functions"))

    return measures
end

function validate_resample(
    users::OptNamedTuple,
    rng::Union{AbstractRNG, Nothing}=nothing,
    train_ratio::Float64=0.7
)::Resample
    check_params(users, (:type, :params))
    type = get_type(users, SoleXplorer.AVAIL_RESAMPLES)
    def_params = SoleXplorer.RESAMPLE_PARAMS[type]

    # validate parameters
    user_params = check_user_params(users, SoleXplorer.RESAMPLE_PARAMS)
    params = merge_params(def_params, user_params, rng)
    # if resample type is Holdout, we need to set the fraction_ratio as preprocess train_ratio
    haskey(params, :fraction_train) && (params = merge(params, (;fraction_train=train_ratio)))
        
    return Resample(type, params)
end

function validate_winparams(
    defaults::WinParams,
    users::OptNamedTuple,
    treatment::Symbol
)::WinParams
    # get type
    user_type = get_type(users, AVAIL_WINS)

    # select the final type with proper priority: defaults -> globals -> users
    type = user_type === nothing ? defaults.type : user_type

    def_params = WIN_PARAMS[type]

    # validate parameters
    user_params = check_user_params(users, WIN_PARAMS)
    params = merge(def_params, user_params)

    # ModalDecisionTrees package needs at least 3 windows to work properly
    if treatment == :reducesize && haskey(params, :nwindows)
        params.nwindows ≥ 3 || throw(ArgumentError("For :reducesize treatment, nwindows must be ≥ 3"))
    end

    return WinParams(type, params)
end

function validate_tuning_type(
    users::OptNamedTuple,
    rng::Union{Nothing, AbstractRNG}
)::TuningStrategy
    check_params(users, (:type, :params))
    type = get_type(users, SoleXplorer.AVAIL_TUNING_METHODS)
    def_params = SoleXplorer.TUNING_METHODS_PARAMS[type]

    # validate parameters
    user_params = check_user_params(users, SoleXplorer.TUNING_METHODS_PARAMS)
    params = merge_params(def_params, user_params, rng)
        
    return TuningStrategy(type, params)
end

function validate_tuning(
    defaults::TuningParams,
    users::NamedTupleBool,
    rng::Union{Nothing, AbstractRNG},
    algo::DataType
)::Union{TuningParams, Bool}
    if isa(users, Bool) 
        if users
            method = !(rng === nothing) && haskey(TUNING_METHODS_PARAMS[defaults.method.type], :rng) ?
                SoleXplorer.TuningStrategy(defaults.method.type, merge(defaults.method.params, (rng=rng,))) :
                SoleXplorer.TuningStrategy(defaults.method.type, defaults.method.params)
            return SoleXplorer.TuningParams(method, defaults.params, defaults.ranges)
        else
            return false
        end
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

function validate_rulesparams(
    defaults::RulesParams,
    users::NamedTupleBool,
    rng::Union{Nothing, AbstractRNG},
)::Union{RulesParams, Bool}
    # case 1: users is a Bool
    if isa(users, Bool)
        users ? (return defaults) : (return false)
    end

    # case 2: users is a NamedTuple
    type = users.type === nothing ? defaults.type : begin
            haskey(EXTRACT_RULES, users.type) || throw(ArgumentError("Type $(users.type) not found in available extract rules methods."))
            users.type
        end

    def_params = RULES_PARAMS[type]

    user_params = check_user_params(users, RULES_PARAMS)
    params = merge_params(def_params, user_params, rng)

    return RulesParams(type, params)
end

# ---------------------------------------------------------------------------- #
#                              validate modelset                               #
# ---------------------------------------------------------------------------- #
function validate_modelset(
    model         :: NamedTuple,
    y             :: OptDataType;
    resample      :: NamedTuple,
    win           :: OptNamedTuple  = nothing,
    features      :: OptTuple       = nothing,
    tuning        :: NamedTupleBool = false,
    extract_rules :: NamedTupleBool = false,
    preprocess    :: OptNamedTuple  = nothing,
    # modalreduce    :: OptCallable    = nothing,
    measures      :: OptTuple       = nothing,
)::ModelSetup
    check_params(model, (:type, :params))
    check_params(resample, (:type, :params))
    check_params(win, (:type, :params))
    check_params(preprocess, DEFAULT_PREPROC)

    # grab rng form preprocess and feed it to every process
    rng = if preprocess === nothing 
        nothing
    else
        haskey(preprocess, :rng) ? preprocess.rng : nothing
    end

    haskey(model, :type) || throw(ArgumentError("Each model specification must contain a 'type' field"))
    modelset = validate_model(model.type, y)

    # grab additional extra params
    user_params = get(model, :params, nothing)
    if !(user_params === nothing) && haskey(user_params, :modalreduce)
        set_config!(modelset, merge(get_config(modelset), (modalreduce = user_params.modalreduce,)))
        user_params = NamedTuple(k => v for (k, v) in pairs(user_params) if k != :modalreduce)
    end
    set_params!(modelset, validate_params(get_params(modelset), user_params, rng))

    # ModalDecisionTrees package needs features to be passed also in model params
    pfeats = get_pfeatures(modelset)
    if pfeats === nothing
        set_features!(modelset, validate_features(get_features(modelset), features))
    else
        features = validate_features(
            pfeats,
            features
        )
        set_params!(modelset, merge(get_params(modelset), (features=features,)))
        set_features!(modelset, features)
    end

    set_winparams!(modelset, validate_winparams(get_winparams(modelset), win, get_treatment(modelset)))
    set_tuning!(modelset, validate_tuning(get_tuning(modelset), tuning, rng, modeltype(modelset)))
    set_rulesparams!(modelset, validate_rulesparams(get_rulesparams(modelset), extract_rules, rng))

    set_rawmodel!(modelset, get_tuning(modelset) == false ? get_rawmodel(modelset) : get_resampled_rawmodel(modelset))
    # set_learn_method!(modelset, get_tuning(modelset) == false ? get_learn_method(modelset) : get_resampled_learn_method(modelset))
    preprocess === nothing || (modelset.preprocess = merge(get_preprocess(modelset), preprocess))
    set_resample!(modelset, validate_resample(resample, rng, modelset.preprocess.train_ratio))
    set_measures!(modelset, validate_measures(measures))
    
    # modelset.preprocess.modalreduce === nothing || (modelset.config = merge(get_config(modelset), (modalreduce=modalreduce,)))

    return modelset
end
