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
    X             :: AbstractDataFrame,
    y             :: AbstractVector;
    model         :: NamedTuple     = (;type=:decisiontree),
    resample      :: NamedTuple     = (;type=Holdout),
    win           :: OptNamedTuple  = nothing,
    features      :: OptTuple       = nothing,
    tuning        :: NamedTupleBool = false,
    extract_rules :: NamedTupleBool = false,
    preprocess    :: OptNamedTuple  = nothing,
    reducefunc    :: OptCallable    = nothing
)::Modelset
    modelset = validate_modelset(model, eltype(y); resample, win, features, tuning, extract_rules, preprocess, reducefunc)
    model = Modelset(modelset, _prepare_dataset(X, y, modelset))
    _traintest!(model)

    if !isa(extract_rules, Bool) || extract_rules
        rules_extraction!(model)
    end

    # save results into model
    # model.results = RESULTS[get_algo(model.setup)](model.setup, model.model)

    return model
end

# y is not a vector, but a symbol or a string that identifies a column in X
function symbolic_analysis(
    X::AbstractDataFrame,
    y::SymbolString;
    kwargs...
)::Modelset
    symbolic_analysis(X[!, Not(y)], X[!, y]; kwargs...)
end

