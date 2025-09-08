"""
train_test.jl - Cross-Validation and Model Training Interface

This module provides training workflows that:
1. Execute cross-validation across multiple data folds
2. Train ML models on each fold using MLJ
3. Convert trained models to symbolic representations
4. Return SoleModel containers with symbolic models for analysis

Key components:
- SoleModel: Container for collections of symbolic models from CV folds
- train_test: Main interface for training and symbolic conversion
"""

# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
"""
    AbstractSoleModel

Base abstract type for all symbolic model containers in SoleXplorer.
"""
abstract type AbstractSoleModel end

# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
"""
Alias for XGBoost model types (classifier and regressor).
"""
const XGBoostModel = Union{XGBoostClassifier, XGBoostRegressor}

# ---------------------------------------------------------------------------- #
#                                  utilities                                   #
# ---------------------------------------------------------------------------- #
"""
    get_X(ds::AbstractDataSet) -> DataFrame

Extract feature DataFrame from dataset's MLJ machine.
"""
get_X(ds::AbstractDataSet)::DataFrame = ds.mach.args[1].data

"""
    get_y(ds::AbstractDataSet) -> Vector

Extract target vector from dataset's MLJ machine.
"""
get_y(ds::AbstractDataSet)::Vector    = ds.mach.args[2].data

"""
    has_xgboost_model(ds::AbstractDataSet) -> Bool
    has_xgboost_model(model) -> Bool

Check if dataset or model uses XGBoost.
Used to determine if XGBoost-specific setup (watchlist) is needed.
"""
has_xgboost_model(ds::AbstractDataSet) = has_xgboost_model(ds.mach.model)
has_xgboost_model(model::MLJTuning.EitherTunedModel) = has_xgboost_model(model.model)
has_xgboost_model(::XGBoostModel) = true
has_xgboost_model(::Any) = false

"""
    is_tuned_model(ds::AbstractDataSet) -> Bool
    is_tuned_model(model) -> Bool

Check if dataset uses hyperparameter tuning.
"""
is_tuned_model(ds::AbstractDataSet) = is_tuned_model(ds.mach.model)
is_tuned_model(::MLJTuning.EitherTunedModel) = true
is_tuned_model(::Any) = false

"""
    get_early_stopping_rounds(ds::AbstractDataSet) -> Int

Extract early stopping rounds parameter from XGBoost models.
"""
function get_early_stopping_rounds(ds::AbstractDataSet)
    if is_tuned_model(ds)
        return ds.mach.model.model.early_stopping_rounds
    else
        return ds.mach.model.early_stopping_rounds
    end
end

"""
    makewatchlist!(ds::AbstractDataSet, train::Vector{Int}, valid::Vector{Int})

Create XGBoost watchlist for early stopping validation.

Throws `ArgumentError` if validation set is empty.
"""
function makewatchlist!(ds::AbstractDataSet, train::Vector{Int}, valid::Vector{Int})
    isempty(valid) && throw(ArgumentError("No validation data provided, use preprocess valid_ratio parameter"))

    X = get_X(ds)
    y = get_y(ds)
    y_train = @views y[train]
    y_valid = @views y[valid]
    feature_names = String.(propertynames(X))
    if eltype(y) <: CLabel
        y_train = @. MLJ.levelcode(y[train]) - 1 # convert to 0-based indexing
        y_valid = @. MLJ.levelcode(y[valid]) - 1 # convert to 0-based indexing
    end
    dtrain        = XGBoost.DMatrix((@views X[train, :], y_train); feature_names)
    dvalid        = XGBoost.DMatrix((@views X[valid, :], y_valid); feature_names)

    watchlist = XGBoost.OrderedDict(["train" => dtrain, "eval" => dvalid])

    if is_tuned_model(ds)
        ds.mach.model.model.watchlist = watchlist
    else
        ds.mach.model.watchlist = watchlist
    end
end

"""
    set_watchlist!(ds::AbstractDataSet, i::Int)

Configure XGBoost watchlist for fold `i` if early stopping is enabled.
"""
function set_watchlist!(ds::AbstractDataSet, i::Int)
    # xgboost ha la funzione di earlystopping. per farla funzionare occorre passargli una makewatchlist e la valid_ratio
    if get_early_stopping_rounds(ds) > 0
        train = get_train(ds.pidxs[i])
        valid = get_valid(ds.pidxs[i])
        makewatchlist!(ds, train, valid)
    end
end

# ---------------------------------------------------------------------------- #
#                                  solemodel                                   #
# ---------------------------------------------------------------------------- #
"""
    SoleModel{D} <: AbstractSoleModel

Container for collections of symbolic models from cross-validation.

# Fields
- `sole::Vector{AbstractModel}`: Vector of symbolic models, one per CV fold

# Type Parameter
- `D`: Dataset type that generated these models (e.g., PropositionalDataSet)

# Constructor
    SoleModel(ds::D, sole::Vector{AbstractModel}) where D<:AbstractDataSet
"""
mutable struct SoleModel{D} <: AbstractSoleModel
    sole   :: Vector{AbstractModel}

    function SoleModel(::D, sole::Vector{AbstractModel}) where D<:AbstractDataSet
        new{D}(sole)
    end
end

function Base.show(io::IO, solem::SoleModel{D}) where D
    n_models = length(solem.sole)
    dataset_type = D <: AbstractDataSet ? string(D) : "Unknown"
    
    print(io, "SoleModel{$dataset_type}")
    print(io, "\n  Number of models: $n_models")
end

function Base.show(io::IO, ::MIME"text/plain", solem::SoleModel{D}) where D
    show(io, solem)
end

# ---------------------------------------------------------------------------- #
#                                 constructors                                 #
# ---------------------------------------------------------------------------- #
"""
    solemodels(solem::SoleModel) -> Vector{AbstractModel}

Extract the vector of symbolic models from a SoleModel container.
"""
solemodels(solem::SoleModel) = solem.sole

# ---------------------------------------------------------------------------- #
#                                  train_test                                  #
# ---------------------------------------------------------------------------- #
function _train_test(ds::AbstractDataSet)::SoleModel
    n_folds   = length(ds.pidxs)
    solemodel = Vector{AbstractModel}(undef, n_folds)

    # TODO this can be parallelizable
    @inbounds @views for i in 1:n_folds
        train, test = get_train(ds.pidxs[i]), get_test(ds.pidxs[i])
        X_test, y_test = get_X(ds)[test, :], get_y(ds)[test]

        has_xgboost_model(ds) && set_watchlist!(ds, i)

        mach = get_mach(ds)
        MLJ.fit!(mach, rows=train, verbosity=0)
        solemodel[i] = apply(mach, X_test, y_test)
    end

    return SoleModel(ds, solemodel)
end

"""
    _train_test(ds::AbstractDataSet) -> SoleModel

Internal cross-validation training implementation.

Workflow for each fold:
1. Extract train/test indices and data
2. Configure XGBoost watchlist if needed (early stopping)
3. Fit MLJ machine on training data
4. Apply trained model to test data (converts to symbolic model)
5. Store symbolic model in results vector

Returns SoleModel containing all fold models.

See [`setup_dataset`](@ref) for dataset setup parameter descriptions.
"""
function train_test(args...; kwargs...)::SoleModel
    ds = _setup_dataset(args...; kwargs...)
    _train_test(ds)
end

"""
    train_test(ds::AbstractDataSet) -> SoleModel

Direct training interface for pre-configured datasets.
"""
train_test(ds::AbstractDataSet)::SoleModel = _train_test(ds)
