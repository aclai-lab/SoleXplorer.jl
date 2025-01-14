function check_unknown_params(params, default_params, source)
    isnothing(params) && return
    unknown = setdiff(keys(params), keys(default_params))
    !isempty(unknown) && throw(ArgumentError("Unknown parameters in $source: $unknown"))
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


function get_wintype(params)
    isnothing(params) && return nothing
    win_values = filter(v -> v in AVAIL_WINS, collect(values(params)))
    length(win_values) > 1 && throw(ArgumentError("Multiple window functions detected. Only one window function is allowed."))
    return isempty(win_values) ? nothing : win_values[1]
end

function validate_winparams(
    default_params::NamedTuple,
    global_params::Union{NamedTuple, Nothing},
    user_params::Union{NamedTuple, Nothing},
)
    global_win = get_wintype(global_params)
    user_win = get_wintype(user_params)
    
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
    global_filt_params = filter_valid_params(global_params)
    user_filt_params = filter_valid_params(user_params)

    filtered_default_params = NamedTuple(k => v for (k,v) in pairs(default_params) if k != :type)

    return (type = type, params = merge(filtered_default_params, global_filt_params, user_filt_params))
end

function validate_ranges(
    default_ranges::AbstractVector{<:Base.Callable},
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

function validate_modelset(
    models::AbstractVector{<:NamedTuple}, 
    global_params::Union{NamedTuple, Nothing}=nothing,
)
    modelsets = AbstractModelSet[]

    for m in models
        haskey(m, :model) || throw(ArgumentError("Each model specification must contain a 'model' field"))
        model = validate_model(m.model)

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

        if haskey(m, :tuneparams) && m.tuneparams == true
            model.learn_method = model.learn_method[2]
            model.ranges = validate_ranges(
                model.ranges,
                isnothing(global_params) ? nothing : haskey(global_params, :ranges) ? global_params.ranges : nothing,
                haskey(m, :ranges) ? m.ranges : nothing
            )
        else
            model.learn_method = model.learn_method[1]
            model.ranges = Base.Callable[]
        end


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
    y::AbstractVector; 
    models::AbstractVector{<:NamedTuple}, 
    global_params::Union{NamedTuple, Nothing}=nothing,
    kwargs...
)
    modelsets = validate_modelset(models, global_params)

    for m in modelsets
    #     model = SX.get_model(m)
        # ds = SX.preprocess_dataset(X, y, model)

    #     SX.modelfit!(model, ds);
    #     SX.modeltest!(model, ds);

    #     @test_nowarn SX.get_rules(model, ds);
    #     @test_nowarn SX.get_predict(model, ds);
    end
    return modelsets
end

function symbolic_analysis(
    X::AbstractDataFrame, 
    y::AbstractVector; 
    model::Union{NamedTuple, Nothing}=nothing, 
    models::Union{AbstractVector{<:NamedTuple}, Nothing}=nothing, 
    kwargs...
)
    check_dataframe_type(X) || throw(ArgumentError("DataFrame must contain only numeric values"))
    size(X, 1) == length(y) || throw(ArgumentError("Number of rows in DataFrame must match length of class labels"))

    isnothing(model) && isnothing(models) && throw(ArgumentError("At least one model must be specified"))
    !isnothing(model) && !isnothing(models) && throw(ArgumentError("You can specify either a single model or a vector of models, not both"))

    isnothing(model) ? _symbolic_analysis(X, y; models=models, kwargs...) : _symbolic_analysis(X, y; models=[model], kwargs...)
end
