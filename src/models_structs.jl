# ---------------------------------------------------------------------------- #
#                                    dataset                                   #
# ---------------------------------------------------------------------------- #
struct TT_indexes
    train :: AbstractVector{<:Int}
    test  :: AbstractVector{<:Int}
end

struct Dataset
    X     :: AbstractDataFrame
    y     :: Union{CategoricalArray, Vector{<:Number}}
    tt    :: Union{SoleXplorer.TT_indexes, AbstractVector{<:SoleXplorer.TT_indexes}}
end

# ---------------------------------------------------------------------------- #
#                           model consts & structs                             #
# ---------------------------------------------------------------------------- #
abstract type AbstractModelSet end
abstract type AbstractModelConfig end

mutable struct SymbolicModelSet <: AbstractModelSet
    type         :: Base.Callable
    config       :: NamedTuple
    params       :: NamedTuple
    features     :: Union{AbstractVector{<:Base.Callable}, Nothing}
    winparams    :: NamedTuple
    learn_method :: Union{Base.Callable, Tuple{Base.Callable, Base.Callable}}
    tuning       :: NamedTuple
    rules_method :: SoleModels.RuleExtractor
    preprocess   :: NamedTuple
end

DecisionTreeModel(dtmodel      :: SymbolicModelSet) = dtmodel
RandomForestModel(dtmodel      :: SymbolicModelSet) = dtmodel
AdaBoostModel(dtmodel          :: SymbolicModelSet) = dtmodel

ModalDecisionTreeModel(dtmodel :: SymbolicModelSet) = dtmodel
ModalRandomForestModel(dtmodel :: SymbolicModelSet) = dtmodel
ModalAdaBoostModel(dtmodel     :: SymbolicModelSet) = dtmodel

mutable struct ModelConfig <: AbstractModelConfig
    setup      :: AbstractModelSet
    ds         :: Dataset
    classifier :: MLJ.Model
    mach       :: MLJ.Machine
    model      :: AbstractModel
    rules      :: AbstractDataFrame
    accuracy   :: AbstractFloat
end

const DEFAULT_FEATS = [maximum, minimum, mean, std]

const DEFAULT_PREPROC = (
    train_ratio         = 0.8,
    shuffle             = true,
    rng                 = TaskLocalRNG(),
    stratified_sampling = false,
    nfolds              = 6
)

const AVAIL_MODELS = Dict(
    :decisiontree      => DecisionTreeModel,
    :randomforest      => RandomForestModel,
    :adaboost          => AdaBoostModel,

    :modaldecisiontree => ModalDecisionTreeModel,
    :modalrandomforest => ModalRandomForestModel,
    :modaladaboost     => ModalAdaBoostModel,

    # :modal_decision_list => ModalDecisionLists.MLJInterface.ExtendedSequentialCovering,

    # :regression_tree     => MLJDecisionTreeInterface.DecisionTreeRegressor,
    # :regression_forest   => MLJDecisionTreeInterface.RandomForestRegressor,

    # :xgboost => MLJXGBoostInterface.XGBoostClassifier,
)

const AVAIL_WINS = (movingwindow, wholewindow, splitwindow, adaptivewindow)

const WIN_PARAMS = Dict(
    movingwindow   => (window_size = 1024, window_step = 512),
    wholewindow    => NamedTuple(),
    splitwindow    => (nwindows = 20),
    adaptivewindow => (nwindows = 20, relative_overlap = 0.5)
)

# ---------------------------------------------------------------------------- #
#                                   tuning                                     #
# ---------------------------------------------------------------------------- #
const AVAIL_TUNING_METHODS = (grid, randomsearch, latinhypercube, treeparzen, particleswarm, adaptiveparticleswarm)

const TUNING_METHODS_PARAMS = Dict(
    grid                  => (
        goal                   = nothing,
        resolution             = 10,
        shuffle                = true,
        rng                    = TaskLocalRNG()
    ),
    randomsearch          => (
        bounded                = Distributions.Uniform,
        positive_unbounded     = Distributions.Gamma,
        other                  = Distributions.Normal,
        rng                    = TaskLocalRNG()
    ),
    latinhypercube        => (
        gens                   = 1, 
        popsize                = 100, 
        ntour                  = 2, 
        ptour                  = 0.8, 
        interSampleWeight      = 1.0, 
        ae_power               = 2, 
        periodic_ae            = false, 
        rng                    = TaskLocalRNG()
    ),
    treeparzen            => (
        config                 = Config(0.25, 25, 24, 20, 1.0), 
        max_simultaneous_draws = 1
    ),
    particleswarm         => (
        n_particles            = 3, 
        w                      = 1.0, 
        c1                     = 2.0, 
        c2                     = 2.0, 
        prob_shift             = 0.25, 
        rng                    = TaskLocalRNG()
    ),
    adaptiveparticleswarm => (
        n_particles            = 3, 
        c1                     = 2.0, 
        c2                     = 2.0, 
        prob_shift             = 0.25, 
        rng                    = TaskLocalRNG()
    )
)

const TUNING_PARAMS = (;
    resampling              = Holdout(),
    measure                 = LogLoss(tol = 2.22045e-16),
    weights                 = nothing,
    class_weights           = nothing,
    repeats                 = 1,
    operation               = nothing,
    selection_heuristic     = MLJTuning.NaiveSelection(nothing),
    n                       = 25,
    train_best              = true,
    acceleration            = default_resource(),
    acceleration_resampling = CPU1(),
    check_measure           = true,
    cache                   = true,
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
            lower  = lower,
            upper  = upper,
            origin = origin,
            unit   = unit,
            scale  = scale,
            values = values,
        )
    end
end
