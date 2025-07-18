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
get_X(model::AbstractDataSet)::DataFrame = model.mach.args[1].data
get_y(model::AbstractDataSet)::Vector = model.mach.args[2].data

has_xgboost_model(model::AbstractDataSet) = has_xgboost_model(model.mach.model)
has_xgboost_model(model::MLJTuning.EitherTunedModel) = has_xgboost_model(model.model)
has_xgboost_model(::XGBoostModel) = true
has_xgboost_model(::Any) = false

is_tuned_model(model::AbstractDataSet) = is_tuned_model(model.mach.model)
is_tuned_model(::MLJTuning.EitherTunedModel) = true
is_tuned_model(::Any) = false

function get_early_stopping_rounds(model::AbstractDataSet)
    if is_tuned_model(model)
        return model.mach.model.model.early_stopping_rounds
    else
        return model.mach.model.early_stopping_rounds
    end
end

function makewatchlist!(model::AbstractDataSet, train::Vector{Int}, valid::Vector{Int})
    isempty(valid) && throw(ArgumentError("No validation data provided, use preprocess valid_ratio parameter"))

    X = get_X(model)
    y = get_y(model)
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

    if is_tuned_model(model)
        model.mach.model.model.watchlist = watchlist
    else
        model.mach.model.watchlist = watchlist
    end
end

function set_watchlist!(model::AbstractDataSet, i::Int)
    # xgboost ha la funzione di earlystopping. per farla funzionare occorre passargli una makewatchlist e la valid_ratio
    if get_early_stopping_rounds(model) > 0
        train = get_train(model.pidxs[i])
        valid = get_valid(model.pidxs[i])
        makewatchlist!(model, train, valid)
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

# ---------------------------------------------------------------------------- #
#                                 constructors                                 #
# ---------------------------------------------------------------------------- #
solemodels(solem::SoleModel) = solem.sole

# ---------------------------------------------------------------------------- #
#                                  train_test                                  #
# ---------------------------------------------------------------------------- #
function _train_test(model::EitherDataSet)::SoleModel
    n_folds   = length(model.pidxs)
    solemodel = Vector{AbstractModel}(undef, n_folds)

    # TODO this can be parallelizable
    @inbounds @views for i in 1:n_folds
        train, test = get_train(model.pidxs[i]), get_test(model.pidxs[i])
        X_test, y_test = get_X(model)[test, :], get_y(model)[test]

        has_xgboost_model(model) && set_watchlist!(model, i)

        MLJ.fit!(model.mach, rows=train, verbosity=0)
        solemodel[i] = apply(model, X_test, y_test)
    end

    return SoleModel(model, solemodel)
end

function train_test(args...; kwargs...)::SoleModel
    model = _prepare_dataset(args...; kwargs...)
    _train_test(model)
end

train_test(model::AbstractDataSet)::SoleModel = _train_test(model)
