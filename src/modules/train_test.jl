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

    n_folds = length(model.ds.tt)
    model.mach = Vector{MLJ.Machine}(undef, n_folds)
    model.model = Vector{SoleXplorer.AbstractModel}(undef, n_folds)
    
    # Efficient training loop with views
    @inbounds for (i, fold) in enumerate(model.ds.tt)
        X_train = MLJ.table(@views model.ds.X[fold.train, :])
        X_test  = DataFrame((@views model.ds.X[fold.test, :]), model.ds.info.vnames)
        y_train = @views model.ds.y[fold.train]
        y_test  = @views model.ds.y[fold.test]
        
        model.mach[i] = MLJ.machine(model.predictor, X_train, y_train) |> m -> MLJ.fit!(m, verbosity=0)
        model.model[i] = model.setup.learn_method(model.mach[i], X_test, y_test)
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