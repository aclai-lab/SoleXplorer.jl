# ---------------------------------------------------------------------------- #
#                               compute_results                                #
# ---------------------------------------------------------------------------- #
function compute_results(
    algo::Symbol,
    labels::AbstractVector,
    predictions::AbstractVector, 
)::AbstractResults
    accuracy = sum(predictions .== labels)/length(labels)
    
    return RESULTS[algo](accuracy)
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
    tuning::Union{NamedTuple, Bool, Nothing}=nothing,
    rules::Union{NamedTuple, Nothing}=nothing,
    preprocess::Union{NamedTuple, Nothing}=nothing,
    reducefunc::Union{Base.Callable, Nothing}=nothing,
)::Modelset
    # if model is unspecified, use default model setup
    isnothing(model) && (model = DEFAULT_MODEL_SETUP)
    modelset = validate_modelset(model, eltype(y); resample, win, features, tuning, rules, preprocess, reducefunc)
    model = Modelset(modelset, _prepare_dataset(X, y, modelset))
    _traintest!(model)

    # TODO extract rules, if needed

    # save results into model
    model.results = compute_results(get_algo(model), get_labels(model), get_predictions(model))

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

