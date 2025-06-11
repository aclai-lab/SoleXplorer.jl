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

    # if model.ds.Xtrain isa AbstractVector
    #     Xtrain = DataFrame.(model.ds.Xtrain, model.ds.info.vnames)
    #     Xtest = DataFrame.(model.ds.Xtest, model.ds.info.vnames)
    # else
    #     Xtrain = DataFrame(model.ds.Xtrain, model.ds.info.vnames)
    #     Xtest = DataFrame(model.ds.Xtest, model.ds.info.vnames)
    # end

    # model.mach = MLJ.machine(model.predictor, Xtrain, model.ds.ytrain) |> m -> fit!(m, verbosity=0)
    # model.model = model.setup.learn_method(model.mach, Xtest, model.ds.ytest)

    # return model

    # convert data to DataFrame based on its structure
    if model.ds.Xtrain isa AbstractVector
        # case 1: Xtrain is a vector of datasets (for cross-validation or multiple folds)
        Xtrain = [MLJ.table(x) for x in model.ds.Xtrain]
        Xtest = [DataFrame(x, model.ds.info.vnames) for x in model.ds.Xtest]

        model.mach = Vector{MLJ.Machine}(undef, length(Xtrain))
        model.model = Vector{SoleXplorer.AbstractModel}(undef, length(Xtrain))
        for i in 1:length(Xtrain)
            model.mach[i] = MLJ.machine(model.predictor, Xtrain[i], model.ds.ytrain[i]) |> m -> MLJ.fit!(m, verbosity=0)
            model.model[i] = model.setup.learn_method(model.mach[i], Xtest[i], model.ds.ytest[i])
        end
    else
        # case 2: Xtrain is a single dataset
        Xtrain = MLJ.table(model.ds.Xtrain)
        Xtest = DataFrame(model.ds.Xtest, model.ds.info.vnames)

        model.mach = MLJ.machine(model.predictor, Xtrain, model.ds.ytrain) |> m -> MLJ.fit!(m, verbosity=0)
        model.model = model.setup.learn_method(model.mach, Xtest, model.ds.ytest)
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