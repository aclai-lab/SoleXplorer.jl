# ---------------------------------------------------------------------------- #
#                                   fit model                                  #
# ---------------------------------------------------------------------------- #
function fitmodel(modelset::AbstractModelSetup, classifier::MLJ.Model, ds::Dataset; kwargs...)
    Xtrain, ytrain = ds.Xtrain isa AbstractDataFrame ? ([ds.Xtrain], [ds.ytrain]) : (ds.Xtrain, ds.ytrain)
    # if haskey(modelset.params, :watchlist) && modelset.params.watchlist == makewatchlist
    #     modelset.params = merge(modelset.params, (watchlist = makewatchlist(ds.Xtrain, ds.ytrain, ds.Xvalid, ds.yvalid),))
    # end

    mach = [MLJ.machine(classifier, x, y; kwargs...) |> m -> fit!(m, verbosity=0) for (x, y) in zip(Xtrain, ytrain)]

    return length(mach) == 1 ? only(mach) : mach
end

# ---------------------------------------------------------------------------- #
#                                  test model                                  #
# ---------------------------------------------------------------------------- #
function testmodel(modelset::AbstractModelSetup, mach::Union{T, AbstractVector{<:T}}, ds::Dataset) where T<:MLJ.Machine
    mach isa AbstractVector || (mach = [mach])
    Xtrain, ytrain = ds.Xtrain isa AbstractDataFrame ? ([ds.Xtrain], [ds.ytrain]) : (ds.Xtrain, ds.ytrain)
    tmodel = [modelset.learn_method(m, x, y) for (m, x, y) in zip(mach, Xtrain, ytrain)]

    return length(tmodel) == 1 ? only(tmodel) : tmodel
end

# ---------------------------------------------------------------------------- #
#                                  train_test                                  #
# ---------------------------------------------------------------------------- #
function _traintest(
    X::AbstractDataFrame,
    y::AbstractVector;
    models::AbstractVector{<:NamedTuple},
    globals::Union{NamedTuple, Nothing}=nothing,
    preprocess::Union{NamedTuple, Nothing}=nothing,
)::AbstractVector{Modelset}
    modelsets = validate_modelset(models, typeof(y), globals, preprocess)

    map(m -> begin
        ds = prepare_dataset(X, y, m)

        # TODO document this
        if haskey(m.params, :watchlist) && m.params.watchlist == makewatchlist
            m.params = merge(m.params, (watchlist = makewatchlist(ds),))
        end

        classifier = getmodel(m)
        mach = fitmodel(m, classifier, ds)
        model = testmodel(m, mach, ds)
        Modelset(m, ds, classifier, mach, model)
    end, modelsets)
end

"""
    train_test(
        X::AbstractDataFrame, 
        y::AbstractVector; 
        models::Union{NamedTuple, AbstractVector{<:NamedTuple}, Nothing}=nothing, 
        globals::Union{NamedTuple, Nothing}=nothing,
        preprocess::Union{NamedTuple, Nothing}=nothing,
    )

Train and test machine learning models on datasets.

This module provides functionality for training and testing machine learning models
on datasets using a flexible configuration system. It supports both single and multiple
model training with customizable preprocessing and global parameters.

Main functions:
- `traintest`: Primary interface for training and testing models
- `_traintest`: Internal implementation handling core training/testing logic

The module validates input data, handles model configuration, split dataset in train and test
partitions and returns results as Modelset objects containing the trained models, dataset and
test the models.

# Arguments
- `X::AbstractDataFrame`: Input features as a DataFrame containing only numeric values
- `y::AbstractVector`   : Target class labels or regression targets
- `models::Union{NamedTuple, AbstractVector{<:NamedTuple}, Nothing}=nothing`
                        : Model configuration(s) to train and test
- `globals::Union{NamedTuple, Nothing}=nothing`
                        : Global parameters applied across all models
- `preprocess::Union{NamedTuple, Nothing}=nothing`
                        : Preprocessing configuration parameters for the dataset

# Returns
- Single `Modelset` if one model is provided
- Vector of `Modelset` if multiple models are provided

# Throws
- `ArgumentError` if DataFrame contains non-numeric values
- `ArgumentError` if number of rows in X doesn't match length of y
- `ArgumentError` if no models are specified

Examples:
```julia
result = traintest(X, y;
    models=(
        # Define the core model type - required field
        type=:decisiontree_classifier,

        # Fine-tune model hyperparameters
        params=(; max_depth=5, min_samples_leaf=1),

        # Configure windowing strategy:
        # Splits data vectors into 2 windows to enable modal-like behavior
        # even for propositional models that don't natively handle data vectors
        winparams=(; type=adaptivewindow, nwindows=2),
        
        # Specify feature extractors to apply on each window
        # mode_5 is imported from Catch22 package
        features=[minimum, mean, cov, mode_5],

        # Enable automated hyperparameter optimization
        # Uses default tuning settings for the selected model
        tuning=true
    )
)

result = traintest(X, y;
    models=(
        type=:randomforest_classifier,

        # Single parameter configuration requires semicolon or trailing comma
        # Example: (;param=value) or (param=value,)
        params=(; n_trees=25),
        features=[minimum, mean, std],

        # MLJ hyperparameter optimization configuration
        tuning=(
            # you can choose the tuning method and adjust the parameters
            # specific for the choosen method
            method=(type=latinhypercube, rng=rng), 

            # Specify global tuning parameters
            params=(repeats=10, n=5),

            # every model has default ranges for tuning
            # but it's highly recommended to choose which parameters ranges to tune
            ranges=[
                SoleXplorer.range(:sampling_fraction, lower=0.3, upper=0.9),
                SoleXplorer.range(:feature_importance, values=[:impurity, :split])
            ]
        ),   
    )
)

result = traintest(X, y;
    models=(
        type=:decisiontree_classifier,
        params=(; max_depth=5, min_samples_leaf=1),
        winparams=(; type=adaptivewindow, nwindows=2),
        features=[minimum, mean, cov, mode_5],
        tuning=true
    ),
    # Specify preprocessing parameters to fine tuning train test split
    preprocess=(
        train_ratio = 0.7,
        stratified=true,
        nfolds=3,
        rng=rng
    )
)

results = traintest(X, y;
    # you can stack multiple models in a vector
    models=[(
            type=:decisiontree_classifier,
            params=(max_depth=3, min_samples_leaf=14),
            features=[minimum, mean, cov, mode_5]
        ),
        (
            type=:adaboost_classifier,
            winparams=(type=movingwindow, window_size=6),
            tuning=true
        ),
        (; type=:modaldecisiontree)],
    # Specify global parameters applied across all models
    # note that if you specify them also in model definitions, they will be overwritten.
    # for example, this could be very useful for passing 'rng' parameter to all models
    globals=(
        params=(; rng=rng),
        features=[std],
        tuning=false
    )
)

# xgboost classification with early stopping
result = traintest(X, y; models=(type=:xgboost_classifier,
        params=(
        num_round=100,
        max_depth=6,
        eta=0.1, 
        objective="multi:softprob",
        # early_stopping parameters
        early_stopping_rounds=20,
        watchlist=makewatchlist)
    ),
    # with early stopping a validation set is required
    preprocess=(; valid_ratio = 0.7)
)
```
See also [`Modelset`](@ref), [`prepare_dataset`](@ref), [`getmodel`](@ref), [`fitmodel`](@ref), [`testmodel`](@ref).
"""
function train_test(
    X::AbstractDataFrame, 
    y::AbstractVector; 
    models::Union{NamedTuple, AbstractVector{<:NamedTuple}, Nothing}=nothing,
    kwargs...
)::Union{Modelset, AbstractVector{Modelset}}
    check_dataframe_type(X) || throw(ArgumentError("DataFrame must contain only numeric values"))
    size(X, 1) == length(y) || throw(ArgumentError("Number of rows in DataFrame must match length of class labels"))
    isnothing(models) && throw(ArgumentError("At least one type must be specified"))

    if isa(models, NamedTuple)
        first(_traintest(X, y; models=[models], kwargs...))
    else
        _traintest(X, y; models=models, kwargs...)
    end
end