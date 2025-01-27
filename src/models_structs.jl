# ---------------------------------------------------------------------------- #
#                                    dataset                                   #
# ---------------------------------------------------------------------------- #
struct DatasetInfo
    algo        :: Symbol
    treatment   :: Symbol
    features    :: AbstractVector{<:Base.Callable}
    train_ratio :: Float64
    shuffle     :: Bool
    stratified  :: Bool
    nfolds      :: Int
    rng         :: AbstractRNG
    winparams   :: Union{NamedTuple, Nothing}
    vnames      :: Union{AbstractVector{<:Union{AbstractString, Symbol}}, Nothing}
end

function Base.show(io::IO, info::DatasetInfo)
    println(io, "DatasetInfo:")
    println(io, "  Algorithm:      ", info.algo)
    println(io, "  Treatment:      ", info.treatment)
    println(io, "  Features:       ", info.features)
    println(io, "  Train ratio:    ", info.train_ratio)
    println(io, "  Shuffle:        ", info.shuffle)
    println(io, "  Stratified:     ", info.stratified)
    println(io, "  N-folds:        ", info.nfolds)
    println(io, "  RNG:            ", info.rng)
    println(io, "  Win params:     ", info.winparams)
    println(io, "  Variable names: ", info.vnames)
end

struct TT_indexes
    train       :: AbstractVector{<:Int}
    test        :: AbstractVector{<:Int}
end

struct Dataset{T<:AbstractDataFrame,S}
    X           :: T
    y           :: S
    tt          :: Union{TT_indexes, AbstractVector{<:TT_indexes}}
    info        :: DatasetInfo
    Xtrain      :: Union{SubDataFrame{T}, Vector{<:SubDataFrame{T}}}
    Xtest       :: Union{SubDataFrame{T}, Vector{<:SubDataFrame{T}}}
    ytrain      :: Union{SubArray{<:eltype(S)}, Vector{<:SubArray{<:eltype(S)}}}
    ytest       :: Union{SubArray{<:eltype(S)}, Vector{<:SubArray{<:eltype(S)}}}

    function Dataset(X::T, y::S, tt, info) where {T<:AbstractDataFrame,S}
        Xtrain, Xtest, ytrain, ytest = if info.stratified
            (
            view.(Ref(X), getfield.(tt, :train), Ref(:)),
            view.(Ref(X), getfield.(tt, :test), Ref(:)),
            view.(Ref(y), getfield.(tt, :train)),
            view.(Ref(y), getfield.(tt, :test))
            )
        else
            (
            view(X, tt.train, :),
            view(X, tt.test, :),
            view(y, tt.train),
            view(y, tt.test)
            )
        end

        new{T,S}(X, y, tt, info, Xtrain, Xtest, ytrain, ytest)
    end
end

function Base.show(io::IO, ds::Dataset)
    println(io, "Dataset:")
    println(io, "  X shape:        ", size(ds.X))
    println(io, "  y length:       ", length(ds.y))
    if ds.tt isa AbstractVector
        println(io, "  Train/Test:     ", length(ds.tt), " folds")
    else
        println(io, "  Train indices:  ", length(ds.tt.train))
        println(io, "  Test indices:   ", length(ds.tt.test))
    end
    print(io, ds.info)
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
    train_ratio = 0.8,
    shuffle     = true,
    stratified  = false,
    nfolds      = 6,
    rng         = TaskLocalRNG()
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

const AVAIL_WINS       = (movingwindow, wholewindow, splitwindow, adaptivewindow)
const AVAIL_TREATMENTS = (:aggregate, :reducesize)

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
