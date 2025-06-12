# ---------------------------------------------------------------------------- #
#                                   get model                                  #
# ---------------------------------------------------------------------------- #
function get_predictor!(modelset::AbstractModelSetup)::MLJ.Model
    predictor = modelset.type(; modelset.params...)

    modelset.tuning == false || begin
        ranges = [r(predictor) for r in modelset.tuning.ranges]

        predictor = MLJ.TunedModel(; 
            model=predictor, 
            tuning=modelset.tuning.method.type(;modelset.tuning.method.params...),
            range=ranges, 
            modelset.tuning.params...
        )
    end

    return predictor
end

# ---------------------------------------------------------------------------- #
#                                  train_test                                  #
# ---------------------------------------------------------------------------- #
function _traintest!(model::AbstractModelset)::Modelset
    # Early stopping is a regularization technique in XGBoost that prevents overfitting by monitoring model performance 
    # on a validation dataset and stopping training when performance no longer improves.
    if haskey(model.setup.params, :watchlist) && model.setup.params.watchlist == makewatchlist
        model.setup.params = merge(model.setup.params, (watchlist = makewatchlist(model.ds),))
    end

    model.predictor = get_predictor!(model.setup)
    model.mach = MLJ.machine(model.predictor, MLJ.table(@views model.ds.X), @views model.ds.y)

    n_folds = length(model.ds.tt)
    model.model = Vector{SoleXplorer.AbstractModel}(undef, n_folds)

    # TODO this can be parallelizable
    @inbounds for i in 1:n_folds
        train = model.ds.tt[i].train
        test  = model.ds.tt[i].test
        X_test  = DataFrame((@views model.ds.X[test, :]), model.ds.info.vnames)
        y_test  = @views model.ds.y[test]
        
        MLJ.fit!(model.mach, rows=train, verbosity=0)
        model.model[i] = model.setup.learn_method(model.mach, X_test, y_test)
    end

    return model
end

function train_test(
    X             :: AbstractDataFrame,
    y             :: AbstractVector;
    model         :: NamedTuple     = (;type=:decisiontree),
    resample      :: NamedTuple     = (;type=Holdout),
    win           :: OptNamedTuple  = nothing,
    features      :: OptTuple       = nothing,
    tuning        :: NamedTupleBool = false,
    extract_rules :: NamedTupleBool = false,
    preprocess    :: OptNamedTuple  = nothing,
    reducefunc    :: OptCallable    = nothing,
)::Modelset
    modelset = validate_modelset(model, eltype(y); resample, win, features, tuning, extract_rules, preprocess, reducefunc)
    model = Modelset(modelset, _prepare_dataset(X, y, modelset))
    _traintest!(model)

    return model
end

train_test(m::AbstractModelset) = _traintest!(m)

# y is not a vector, but a symbol or a string that identifies the column in X
function train_test(
    X::AbstractDataFrame,
    y::SymbolString;
    kwargs...
)::Modelset
    train_test(X[!, Not(y)], X[!, y]; kwargs...)
end