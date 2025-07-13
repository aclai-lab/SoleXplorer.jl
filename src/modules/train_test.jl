# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
abstract type AbstractModelSet end

# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
# const DataSetType = Union{
#     PropositionalDataSet{<:MLJ.Model},
#     ModalDataSet{<:Modal},
# }

# ---------------------------------------------------------------------------- #
#                                  utilities                                   #
# ---------------------------------------------------------------------------- #
get_X(model::AbstractDataSet)::DataFrame = model.mach.args[1].data
get_y(model::AbstractDataSet)::Vector = model.mach.args[2].data

function makewatchlist(model::AbstractDataSet, train::Vector{Int}, valid::Vector{Int})
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

    XGBoost.OrderedDict(["train" => dtrain, "eval" => dvalid])
end

# ---------------------------------------------------------------------------- #
#                                   modelset                                   #
# ---------------------------------------------------------------------------- #
mutable struct ModelSet{D} <: AbstractModelSet
    sole   :: Vector{AbstractModel}

    function ModelSet(::D, sole::Vector{AbstractModel}) where D<:AbstractDataSet
        new{D}(sole)
    end
end

# ---------------------------------------------------------------------------- #
#                                  train_test                                  #
# ---------------------------------------------------------------------------- #
function _train_test(model::AbstractDataSet)
    n_folds     = length(model.pidxs)
    solemodel = Vector{AbstractModel}(undef, n_folds)

    # TODO this can be parallelizable
    @inbounds @views for i in 1:n_folds
        train, test = get_train(model.pidxs[i]), get_test(model.pidxs[i])
        X_test, y_test = get_X(model)[test, :], get_y(model)[test]

        # xgboost ha la funzione di earlystopping. per farla funzionare occorre passargli una makewatchlist e la valid_ratio
        # TODO sposta tutto in una funzione a parte
        if model.mach.model isa MLJ.MLJTuning.EitherTunedModel
            model.mach.model.model.early_stopping_rounds > 0 && begin
                valid = get_valid(model.pidxs[i])
                model.mach.model.model.watchlist = makewatchlist(model, train, valid)
            end
        else
            model.mach.model.early_stopping_rounds > 0 && begin
                valid = get_valid(model.pidxs[i])
                model.mach.model.watchlist = makewatchlist(model, train, valid)
            end
        end

        MLJ.fit!(model.mach, rows=train, verbosity=0)
        solemodel[i] = apply(model, X_test, y_test)
    end

    return ModelSet(model, solemodel)
end

function train_test(args...; kwargs...)
    model = _prepare_dataset(args...; kwargs...)
    _train_test(model)
end

train_test(model::AbstractDataSet) = _train_test(model)
