# ---------------------------------------------------------------------------- #
#                                   get model                                  #
# ---------------------------------------------------------------------------- #
function get_predictor(modelset::AbstractModelSetup)::MLJ.Model
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
#                                     train                                    #
# ---------------------------------------------------------------------------- #
function _train_machine!(model::AbstractModelset, ds::AbstractDataset)::MLJ.Machine
    # Early stopping is a regularization technique in XGBoost that prevents overfitting by monitoring model performance 
    # on a validation dataset and stopping training when performance no longer improves.
    if haskey(model.setup.params, :watchlist) && model.setup.params.watchlist == makewatchlist
        model.setup.params = merge(model.setup.params, (watchlist = makewatchlist(ds),))
    end

    model.type = get_predictor(model.setup)

    MLJ.machine(
        model.type,
        MLJ.table(@views ds.X; names=ds.info.vnames),
        @views ds.y
    )
end

# ---------------------------------------------------------------------------- #
#                                     test                                     #
# ---------------------------------------------------------------------------- #
function _test_model!(model::AbstractModelset, mach::MLJ.Machine, ds::AbstractDataset)
    n_folds     = length(ds.tt)
    model.model = Vector{AbstractModel}(undef, n_folds)
    yhat        = Vector(undef, n_folds)

    # TODO this can be parallelizable
    @inbounds for i in 1:n_folds
        train   = ds.tt[i].train
        test    = ds.tt[i].test
        X_test  = DataFrame((@views ds.X[test, :]), ds.info.vnames)
        y_test  = @views ds.y[test]
        
        MLJ.fit!(mach, rows=train, verbosity=0)
        model.model[i] = apply(mach, model, X_test, y_test)
    end

    # model.measures = Measures(yhat)
end

function train_test(args...; kwargs...)
    model, ds = _prepare_dataset(args...; kwargs...)
    mach = _train_machine!(model, ds)
    _test_model!(model, mach, ds)

    return model, mach, ds
end
