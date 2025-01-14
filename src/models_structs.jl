# ---------------------------------------------------------------------------- #
#                           model consts & structs                             #
# ---------------------------------------------------------------------------- #
const DEFAULT_FEATS = [maximum, minimum, mean, std]

const TUNEDMODEL_PARAMS = (;
    resampling              = Holdout(),
    measure                 = LogLoss(tol = 2.22045e-16),
    weights                 = nothing,
    class_weights           = nothing,
    repeats                 = 1,
    operation               = nothing,
    selection_heuristic     = MLJTuning.NaiveSelection(nothing),
    n                       = nothing,
    train_best              = true,
    acceleration            = default_resource(),
    acceleration_resampling = CPU1(),
    check_measure           = true,
    cache                   = true,
)

abstract type AbstractModelSet end

mutable struct ModelSet <: AbstractModelSet
    model        :: Base.Callable
    type         :: NamedTuple
    params       :: NamedTuple
    features     :: AbstractVector{<:Base.Callable}
    winparams    :: NamedTuple
    learn_method :: Union{Base.Callable, Tuple{Base.Callable, Base.Callable}}
    ranges       :: Union{Nothing, Base.Callable, AbstractVector{<:Base.Callable}}
    rules_method :: SoleModels.RuleExtractor
end

DecisionTreeModel(dtmodel::ModelSet) = dtmodel


const AVAIL_MODELS = Dict(
    :decision_tree       => DecisionTreeModel,
    :decision_forest     => MLJDecisionTreeInterface.RandomForestClassifier,
    :adaboost            => MLJDecisionTreeInterface.AdaBoostStumpClassifier,

    :modal_decision_tree => ModalDecisionTrees.ModalDecisionTree,
    :modal_adaboost      => ModalDecisionTrees.ModalAdaBoost,

    :modal_decision_list => ModalDecisionLists.MLJInterface.ExtendedSequentialCovering,

    :regression_tree     => MLJDecisionTreeInterface.DecisionTreeRegressor,
    :regression_forest   => MLJDecisionTreeInterface.RandomForestRegressor,

    # :xgboost => MLJXGBoostInterface.XGBoostClassifier,
)

const AVAIL_WINS = (movingwindow, wholewindow, splitwindow, adaptivewindow)

const WIN_PARAMS = Dict(
    movingwindow   => (window_size = 1024, window_step = 512),
    wholewindow    => NamedTuple(),
    splitwindow    => (nwindows = 20),
    adaptivewindow => (nwindows = 20, relative_overlap = 0.5)
)

function range(
    field  :: Union{Expr, Symbol};
    lower  :: Union{AbstractFloat, Int, Nothing} = nothing,
    upper  :: Union{AbstractFloat, Int, Nothing} = nothing,
    origin :: Union{AbstractFloat, Int, Nothing} = nothing,
    unit   :: Union{AbstractFloat, Int, Nothing} = nothing,
    scale  :: Union{Symbol, Nothing}             = nothing,
    values :: Union{AbstractVector, Nothing}     = nothing,
)
    return function(model)
        MLJ.range(
            model,
            field;
            lower=lower,
            upper=upper,
            origin=origin,
            unit=unit,
            scale=scale,
            values=values,
        )
    end
end

