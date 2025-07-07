# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
abstract type AbstractMLJModel <: MLJType end

# ---------------------------------------------------------------------------- #
#                                   models                                     #
# ---------------------------------------------------------------------------- #
struct MLJModel{T} <: AbstractMLJModel
    type::Type{T}
end
(m::MLJModel)(; kwargs...) = m.type(; kwargs...)

# classification models
const decisiontreeclassifier = MLJModel(DecisionTreeClassifier)
const randomforestclassifier = MLJModel(RandomForestClassifier)
const adaboostclassifier     = MLJModel(AdaBoostStumpClassifier)

# regression models
const decisiontreeregressor  = MLJModel(DecisionTreeRegressor)
const randomforestregressor  = MLJModel(RandomForestRegressor)

# modal models
const modaldecisiontree      = MLJModel(ModalDecisionTree)
const modalrandomforest      = MLJModel(ModalRandomForest)
const modaladaboost          = MLJModel(ModalAdaBoost)

# XGBoost models
const xgboostclassifier      = MLJModel(XGBoostClassifier)
const xgboostregressor       = MLJModel(XGBoostRegressor)

# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
const DEFAULT_FEATS = [maximum, minimum, MLJ.mean, std]

# ---------------------------------------------------------------------------- #
#                                 utilities                                    #
# ---------------------------------------------------------------------------- #
function set_rng(m::MLJ.Model, rng::AbstractRNG)::MLJ.Model
    m.rng = rng
    return m
end

function set_features(m::MLJ.Model, features::Vector{<:Base.Callable})::MLJ.Model
    m.features = features
    return m
end

# ---------------------------------------------------------------------------- #
#                               validate model                                 #
# ---------------------------------------------------------------------------- #
function mljmodel(m::NamedTuple, rng::AbstractRNG)::MLJ.Model
    issubset(keys(m), (:type, :params)) || throw(ArgumentError("Unknown fields."))

    modeltype = get(m, :type, nothing)
    isnothing(modeltype) && throw(ArgumentError("Each model specification must contain a 'type' field"))
    
    modelparams = get(m, :params, NamedTuple())

    # check if modeltype is available
    if modeltype isa MLJModel
        model = modeltype(;modelparams...)
        # set rng if the model supports it
        hasproperty(model, :rng) && (model = set_rng(model, rng))
        # ModalDecisionTrees package needs features to be passed in model params
        hasproperty(model, :features) && (model = set_features(model, get(m, :features, DEFAULT_FEATS)))
    else
        throw(ArgumentError("Model $model not found in available models"))
    end

    return model
end