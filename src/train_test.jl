# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
abstract type AbstractSoleModel end

# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
const XGBoostModel = Union{XGBoostClassifier, XGBoostRegressor}

# ---------------------------------------------------------------------------- #
#                                  utilities                                   #
# ---------------------------------------------------------------------------- #
get_X(ds::AbstractDataSet)::DataFrame = ds.mach.args[1].data
get_y(ds::AbstractDataSet)::Vector    = ds.mach.args[2].data

has_xgboost_model(ds::AbstractDataSet) = has_xgboost_model(ds.mach.model)
has_xgboost_model(model::MLJTuning.EitherTunedModel) = has_xgboost_model(model.model)
has_xgboost_model(::XGBoostModel) = true
has_xgboost_model(::Any) = false

is_tuned_model(ds::AbstractDataSet) = is_tuned_model(ds.mach.model)
is_tuned_model(::MLJTuning.EitherTunedModel) = true
is_tuned_model(::Any) = false

function get_early_stopping_rounds(ds::AbstractDataSet)
    if is_tuned_model(ds)
        return ds.mach.model.model.early_stopping_rounds
    else
        return ds.mach.model.early_stopping_rounds
    end
end

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
solemodels(solem::SoleModel) = solem.sole

# ---------------------------------------------------------------------------- #
#                                  train_test                                  #
# ---------------------------------------------------------------------------- #
function _train_test(ds::EitherDataSet)::SoleModel
    n_folds   = length(ds.pidxs)
    solemodel = Vector{AbstractModel}(undef, n_folds)

    # TODO this can be parallelizable
    @inbounds @views for i in 1:n_folds
        train, test = get_train(ds.pidxs[i]), get_test(ds.pidxs[i])
        X_test, y_test = get_X(ds)[test, :], get_y(ds)[test]

        has_xgboost_model(ds) && set_watchlist!(ds, i)

        MLJ.fit!(ds.mach, rows=train, verbosity=0)
        solemodel[i] = apply(ds, X_test, y_test)
    end

    return SoleModel(ds, solemodel)
end

function train_test(args...; kwargs...)::SoleModel
    ds = _prepare_dataset(args...; kwargs...)
    _train_test(model)
end

train_test(ds::AbstractDataSet)::SoleModel = _train_test(ds)
