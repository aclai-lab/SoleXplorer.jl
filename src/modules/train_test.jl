# ---------------------------------------------------------------------------- #
#                                   get model                                  #
# ---------------------------------------------------------------------------- #
function get_predictor(model::AbstractModelSetup)::MLJ.Model
    predictor = model.type(;model.params...)

    model.tuning === false || begin
        ranges = [r(predictor) for r in model.tuning.ranges]

        predictor = MLJ.TunedModel(; 
            model=predictor, 
            tuning=model.tuning.method.type(;model.tuning.method.params...),
            range=ranges, 
            model.tuning.params...
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
        MLJ.table(ds.X; names=ds.info.vnames),
        ds.y
    )
end

# ---------------------------------------------------------------------------- #
#                                     test                                     #
# ---------------------------------------------------------------------------- #
function _test_model!(model::AbstractModelset, mach::MLJ.Machine, ds::AbstractDataset)
    n_folds     = length(ds.tt)
    model.model = Vector{AbstractModel}(undef, n_folds)

    # TODO this can be parallelizable
    @inbounds for i in 1:n_folds
        train   = ds.tt[i].train
        test    = ds.tt[i].test
        X_test  = DataFrame((@views ds.X[test, :]), ds.info.vnames)
        y_test  = @views ds.y[test]

        # xgboost reg:squarederror default base_score is mean(y_train)
        if model.setup.type == MLJXGBoostInterface.XGBoostRegressor 
            base_score = get_base_score(model) == -Inf ? mean(ds.y[train]) : 0.5
            get_tuning(model) === false ?
                (mach.model.base_score = base_score) :
                (mach.model.model.base_score = base_score)
            MLJ.fit!(mach, rows=train, verbosity=0)
            model.model[i] = apply(mach, X_test, y_test, base_score)
        else
            MLJ.fit!(mach, rows=train, verbosity=0)
            model.model[i] = apply(mach, X_test, y_test)
        end
    end
end

function train_test(args...; kwargs...)
    model, ds = _prepare_dataset(args...; kwargs...)
    mach = _train_machine!(model, ds)
    _test_model!(model, mach, ds)

    return model, mach, ds
end
