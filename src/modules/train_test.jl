# ---------------------------------------------------------------------------- #
#                                   get model                                  #
# ---------------------------------------------------------------------------- #
function get_classifier!(modelset::AbstractModelSetup)::MLJ.Model
    classifier = modelset.type(; modelset.params...)

    isnothing(modelset.tuning) || begin
        ranges = [r(classifier) for r in modelset.tuning.ranges]

        classifier = MLJ.TunedModel(; 
            model=classifier, 
            tuning=modelset.tuning.method.type(;modelset.tuning.method.params...),
            range=ranges, 
            modelset.tuning.params...
        )
    end

    return classifier
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

    model.classifier = get_classifier!(model.setup)

    # if model.ds.Xtrain isa AbstractVector
    #     Xtrain = DataFrame.(model.ds.Xtrain, model.ds.info.vnames)
    #     Xtest = DataFrame.(model.ds.Xtest, model.ds.info.vnames)
    # else
    #     Xtrain = DataFrame(model.ds.Xtrain, model.ds.info.vnames)
    #     Xtest = DataFrame(model.ds.Xtest, model.ds.info.vnames)
    # end

    # model.mach = MLJ.machine(model.classifier, Xtrain, model.ds.ytrain) |> m -> fit!(m, verbosity=0)
    # model.model = model.setup.learn_method(model.mach, Xtest, model.ds.ytest)

    # return model

    # convert data to DataFrame based on its structure
    if model.ds.Xtrain isa AbstractVector
        # case 1: Xtrain is a vector of datasets (for cross-validation or multiple folds)
        Xtrain = [DataFrame(x, model.ds.info.vnames) for x in model.ds.Xtrain]
        Xtest = [DataFrame(x, model.ds.info.vnames) for x in model.ds.Xtest]

        model.mach = Vector{MLJ.Machine}(undef, length(Xtrain))
        model.model = Vector{SoleXplorer.AbstractModel}(undef, length(Xtrain))
        for i in 1:length(Xtrain)
            model.mach[i] = MLJ.machine(model.classifier, Xtrain[i], model.ds.ytrain[i]) |> m -> fit!(m, verbosity=0)
            model.model[i] = model.setup.learn_method(model.mach[i], Xtest[i], model.ds.ytest[i])
        end
    else
        # case 2: Xtrain is a single dataset
        Xtrain = DataFrame(model.ds.Xtrain, model.ds.info.vnames)
        Xtest = DataFrame(model.ds.Xtest, model.ds.info.vnames)

        model.mach = MLJ.machine(model.classifier, Xtrain, model.ds.ytrain) |> m -> fit!(m, verbosity=0)
        model.model = model.setup.learn_method(model.mach, Xtest, model.ds.ytest)
    end

    return model
end

function train_test(
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

    return model
end

# y is not a vector, but a symbol or a string that identifies the column in X
function train_test(
    X::AbstractDataFrame,
    y::Union{Symbol,AbstractString};
    kwargs...
)::Modelset
    train_test(X[!, Not(y)], X[!, y]; kwargs...)
end