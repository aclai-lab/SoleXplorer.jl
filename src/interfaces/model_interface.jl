# ---------------------------------------------------------------------------- #
#                                   models                                     #
# ---------------------------------------------------------------------------- #
decisiontreeclassifier(; kwargs...) = DecisionTreeClassifier(; kwargs...)
randomforestclassifier(; kwargs...) = RandomForestClassifier(; kwargs...)
adaboostclassifier(; kwargs...)     = AdaBoostStumpClassifier(; kwargs...)

decisiontreeregressor(; kwargs...)  = DecisionTreeRegressor(; kwargs...)
randomforestregressor(; kwargs...)  = RandomForestRegressor(; kwargs...)

modaldecisiontree(; kwargs...)      = ModalDecisionTree(; kwargs...)
modalrandomforest(; kwargs...)      = ModalRandomForest(; kwargs...)
modaladaboost(; kwargs...)          = ModalAdaBoost(; kwargs...)

xgboostclassifier(; kwargs...)      = XGBoostClassifier(; kwargs...)
xgboostregressor(; kwargs...)       = XGBoostRegressor(; kwargs...)

const AVAIL_MODELS = (
    decisiontreeclassifier, randomforestclassifier, adaboostclassifier,
    decisiontreeregressor, randomforestregressor,
    modaldecisiontree, modalrandomforest, modaladaboost,
    xgboostclassifier, xgboostregressor,
)::Tuple{Vararg{Function}}

# used to pass features to modaltype models in params
# const MODAL_MODELS = (
#     modaldecisiontree, modalrandomforest, modaladaboost,
# )::Tuple{Vararg{Function}}

set_rng!(m::MLJ.Model, rng::AbstractRNG) = m.rng = rng
set_features!(m::ModelSetup, features::Vector{<:Base.Callable}) = m.features = features

# ---------------------------------------------------------------------------- #
#                               validate model                                 #
# ---------------------------------------------------------------------------- #
function validate_model(m::NamedTuple, rng::AbstractRNG)::MLJ.Model
    issubset(keys(m), (:type, :params)) || throw(ArgumentError("Unknown fields."))

    modeltype = get(m, :type, nothing)
    isnothing(modeltype) && throw(ArgumentError("Each model specification must contain a 'type' field"))
    
    modelparams = get(m, :params, NamedTuple())

    # check if modeltype is available
    if modeltype ∈ AVAIL_MODELS
        model = modeltype(;modelparams...)
        # set rng if the model supports it
        hasproperty(model, :rng) && set_rng!(model, rng)
        # ModalDecisionTrees package needs features to be passed also in model params
        hasproperty(model, :features) && set_features!(model, get(m, :features, DEFAULT_FEATS))
    else
        throw(ArgumentError("Model $model not found in available models"))
    end

    return model
end