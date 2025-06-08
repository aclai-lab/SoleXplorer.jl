# ---------------------------------------------------------------------------- #
#                              rules extraction                                #
# ---------------------------------------------------------------------------- #
function rules_extraction!(model::Modelset)
    model.rules = EXTRACT_RULES[model.setup.rulesparams.type](model)
end

# ---------------------------------------------------------------------------- #
#                              symbolic_analysis                               #
# ---------------------------------------------------------------------------- #
function symbolic_analysis(
    X::AbstractDataFrame,
    y::AbstractVector;
    model::Union{NamedTuple, Nothing}=nothing,
    resample::Union{NamedTuple, Nothing}=nothing,
    win::Union{NamedTuple, Nothing}=nothing,
    features::Union{Tuple, Nothing}=nothing,
    tuning::Union{NamedTuple, Bool}=false,
    extract_rules::Union{NamedTuple, Bool}=false,
    preprocess::Union{NamedTuple, Nothing}=nothing,
    reducefunc::Union{Base.Callable, Nothing}=nothing
)::Modelset
    # if model is unspecified, use default model setup
    isnothing(model) && (model = DEFAULT_MODEL_SETUP)
    modelset = validate_modelset(model, eltype(y); resample, win, features, tuning, extract_rules, preprocess, reducefunc)
    model = Modelset(modelset, _prepare_dataset(X, y, modelset))
    _traintest!(model)

    if !isa(extract_rules, Bool) || extract_rules
        rules_extraction!(model)
    end

    # save results into model
    model.results = RESULTS[get_algo(model.setup)](model.setup, model.model)

    return model
end

# y is not a vector, but a symbol or a string that identifies a column in X
function symbolic_analysis(
    X::AbstractDataFrame,
    y::Union{Symbol,AbstractString};
    kwargs...
)::Modelset
    symbolic_analysis(X[!, Not(y)], X[!, y]; kwargs...)
end

