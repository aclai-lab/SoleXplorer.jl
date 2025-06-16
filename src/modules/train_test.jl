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
function _traintest!(model::AbstractModelset, ds::AbstractDataset)::Modelset
    n_folds = length(ds.tt)
    model.model = Vector{AbstractModel}(undef, n_folds)
    model.setup.tt = Vector{Tuple}(undef, n_folds)

    # Early stopping is a regularization technique in XGBoost that prevents overfitting by monitoring model performance 
    # on a validation dataset and stopping training when performance no longer improves.
    if haskey(model.setup.params, :watchlist) && model.setup.params.watchlist == makewatchlist
        # @inbounds for i in 1:n_folds
        #     watchlist = makewatchlist(ds[i])
        #     @show watchlist
            model.setup.params = merge(model.setup.params, (watchlist = makewatchlist(ds),))
        #     model.setup.params = merge(model.setup.params, (watchlist,))
        # end
    end

    model.predictor = get_predictor!(model.setup)
    model.mach = MLJ.machine(model.predictor, MLJ.table(@views ds.X), @views ds.y)

    # TODO this can be parallelizable
    @inbounds for i in 1:n_folds
        train = ds.tt[i].train
        test  = ds.tt[i].test
        X_test  = DataFrame((@views ds.X[test, :]), ds.info.vnames)
        y_test  = @views ds.y[test]
        
        MLJ.fit!(model.mach, rows=train, verbosity=0)
        model.model[i] = model.setup.learn_method(model.mach, X_test, y_test)
        model.setup.tt[i] = (ds.tt[i].test, ds.tt[i].valid)
    end

    return model
end

function train_test(args...; kwargs...)
    model, ds = _prepare_dataset(args...; kwargs...)
    _traintest!(model, ds)

    return model
end

# function train_test(
#     X             :: AbstractDataFrame,
#     y             :: AbstractVector;
#     model         :: NamedTuple     = (;type=:decisiontree),
#     resample      :: NamedTuple     = (;type=Holdout),
#     win           :: OptNamedTuple  = nothing,
#     features      :: OptTuple       = nothing,
#     tuning        :: NamedTupleBool = false,
#     extract_rules :: NamedTupleBool = false,
#     preprocess    :: OptNamedTuple  = nothing,
#     reducefunc    :: OptCallable    = nothing,
# )::Modelset
#     modelset = validate_modelset(model, eltype(y); resample, win, features, tuning, extract_rules, preprocess, reducefunc)
#     model = Modelset(modelset, _prepare_dataset(X, y, modelset))
#     _traintest!(model)

#     return model
# end

# train_test(m::AbstractModelset) = _traintest!(m)

# # y is not a vector, but a symbol or a string that identifies the column in X
# function train_test(
#     X::AbstractDataFrame,
#     y::SymbolString;
#     kwargs...
# )::Modelset
#     train_test(X[!, Not(y)], X[!, y]; kwargs...)
# end