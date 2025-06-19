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
# function _traintest!(model::AbstractModelset, ds::AbstractDataset)::Modelset
#     n_folds         = length(ds.tt)
#     model.fitresult = Vector{Tuple}(undef, n_folds)
#     model.model     = Vector{AbstractModel}(undef, n_folds)
#     # model.setup.tt  = Vector{Tuple}(undef, n_folds)

#     # Early stopping is a regularization technique in XGBoost that prevents overfitting by monitoring model performance 
#     # on a validation dataset and stopping training when performance no longer improves.
#     if haskey(model.setup.params, :watchlist) && model.setup.params.watchlist == makewatchlist
#         # @inbounds for i in 1:n_folds
#         #     watchlist = makewatchlist(ds[i])
#             model.setup.params = merge(model.setup.params, (watchlist = makewatchlist(ds),))
#         #     model.setup.params = merge(model.setup.params, (watchlist,))
#         # end
#     end

#     # model.predictor = get_predictor!(model.setup)
#     mach = MLJ.machine(get_predictor!(model.setup), MLJ.table(@views ds.X; names=ds.info.vnames), @views ds.y)
#     # mach = MLJ.machine(get_predictor!(model.setup), DataFrame(ds.X, ds.info.vnames), @views ds.y)

#     # TODO this can be parallelizable
#     @inbounds for i in 1:n_folds
#         train = ds.tt[i].train
#         test  = ds.tt[i].test
#         X_test  = DataFrame((@views ds.X[test, :]), ds.info.vnames)
#         y_test  = @views ds.y[test]
        
#         MLJ.fit!(mach, rows=train, verbosity=0)
#         model.fitresult[i] = mach.fitresult
#         model.model[i] = model.setup.learn_method(mach, X_test, y_test)
#         # model.setup.tt[i] = (ds.tt[i].test, ds.tt[i].valid)
#     end

#     return model, mach
# end

function _train_machine(model::AbstractModelset, ds::AbstractDataset)::MLJ.Machine
    MLJ.machine(
        get_predictor!(model.setup),
        MLJ.table(@views ds.X; names=ds.info.vnames),
        @views ds.y
    )
end

function _test_model!(model::AbstractModelset, mach::MLJ.Machine, ds::AbstractDataset)
    n_folds         = length(ds.tt)
    model.fitresult = Vector{Tuple}(undef, n_folds)
    model.model     = Vector{AbstractModel}(undef, n_folds)

    # Early stopping is a regularization technique in XGBoost that prevents overfitting by monitoring model performance 
    # on a validation dataset and stopping training when performance no longer improves.
    if haskey(model.setup.params, :watchlist) && model.setup.params.watchlist == makewatchlist
        model.setup.params = merge(model.setup.params, (watchlist = makewatchlist(ds),))
    end

    # TODO this can be parallelizable
    @inbounds for i in 1:n_folds
        train = ds.tt[i].train
        test  = ds.tt[i].test
        X_test  = DataFrame((@views ds.X[test, :]), ds.info.vnames)
        y_test  = @views ds.y[test]
        
        MLJ.fit!(mach, rows=train, verbosity=0)
        model.fitresult[i] = mach.fitresult
        model.model[i] = model.setup.learn_method(mach, X_test, y_test)

    end
end

function train_test(args...; kwargs...)
    model, ds = _prepare_dataset(args...; kwargs...)
    mach = _train_machine(model, ds)
    _test_model!(model, mach, ds)

    return model, mach
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
#     modalreduce    :: OptCallable    = nothing,
# )::Modelset
#     modelset = validate_modelset(model, eltype(y); resample, win, features, tuning, extract_rules, preprocess, modalreduce)
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