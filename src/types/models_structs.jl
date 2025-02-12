# ---------------------------------------------------------------------------- #
#                                    dataset                                   #
# ---------------------------------------------------------------------------- #
struct DatasetInfo
    algo        :: Symbol
    treatment   :: Symbol
    features    :: AbstractVector{<:Base.Callable}
    train_ratio :: Float64
    valid_ratio :: Float64
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
    println(io, "  Valid ratio:    ", info.valid_ratio)
    println(io, "  Shuffle:        ", info.shuffle)
    println(io, "  Stratified:     ", info.stratified)
    println(io, "  N-folds:        ", info.nfolds)
    println(io, "  RNG:            ", info.rng)
    println(io, "  Win params:     ", info.winparams)
    println(io, "  Variable names: ", info.vnames)
end

struct TT_indexes
    train       :: Vector{Int}
    valid       :: Vector{Int}
    test        :: Vector{Int}
end

Base.show(io::IO, t::TT_indexes) = print(io, "TT_indexes(train=", t.train, ", validation=", t.valid, ", test=", t.test, ")")

struct Dataset{T<:AbstractDataFrame,S}
    X           :: T
    y           :: S
    tt          :: Union{TT_indexes, AbstractVector{<:TT_indexes}}
    info        :: DatasetInfo
    Xtrain      :: Union{SubDataFrame{T}, Vector{<:SubDataFrame{T}}}
    Xvalid      :: Union{SubDataFrame{T}, Vector{<:SubDataFrame{T}}}
    Xtest       :: Union{SubDataFrame{T}, Vector{<:SubDataFrame{T}}}
    ytrain      :: Union{SubArray{<:eltype(S)}, Vector{<:SubArray{<:eltype(S)}}}
    yvalid      :: Union{SubArray{<:eltype(S)}, Vector{<:SubArray{<:eltype(S)}}}
    ytest       :: Union{SubArray{<:eltype(S)}, Vector{<:SubArray{<:eltype(S)}}}

    function Dataset(X::T, y::S, tt, info) where {T<:AbstractDataFrame,S}
        if info.stratified
            Xtrain = view.(Ref(X), getfield.(tt, :train), Ref(:))
            Xvalid = view.(Ref(X), getfield.(tt, :valid), Ref(:))
            Xtest  = view.(Ref(X), getfield.(tt, :test), Ref(:))
            ytrain = view.(Ref(y), getfield.(tt, :train))
            yvalid = view.(Ref(y), getfield.(tt, :valid))
            ytest  = view.(Ref(y), getfield.(tt, :test))
        else
            Xtrain = @views X[tt.train, :]
            Xvalid = @views X[tt.valid, :]
            Xtest  = @views X[tt.test, :]
            ytrain = @views y[tt.train]
            yvalid = @views y[tt.valid]
            ytest  = @views y[tt.test]
        end

        new{T,S}(X, y, tt, info, Xtrain, Xvalid, Xtest, ytrain, yvalid, ytest)
    end
end

function Base.show(io::IO, ds::Dataset)
    println(io, "Dataset:")
    println(io, "  X shape:        ", size(ds.X))
    println(io, "  y length:       ", length(ds.y))
    if ds.tt isa AbstractVector
        println(io, "  Train/Valid/Test:     ", length(ds.tt), " folds")
    else
        println(io, "  Train indices:  ", length(ds.tt.train))
        println(io, "  Valid indices:  ", length(ds.tt.valid))
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

function Base.show(io::IO, ::MIME"text/plain", m::SymbolicModelSet)
    println(io, "SymbolicModelSet")
    println(io, "  Model type: ", m.type)
    println(io, "  Features: ", isnothing(m.features) ? "None" : "$(length(m.features)) features")
    println(io, "  Learning method: ", typeof(m.learn_method))
    println(io, "  Rule extraction: ", typeof(m.rules_method))
end

function Base.show(io::IO, m::SymbolicModelSet)
    print(io, "SymbolicModelSet(type=$(m.type), features=$(isnothing(m.features) ? "None" : length(m.features)))")
end

DecisionTreeClassifierModel(dtmodel :: SymbolicModelSet) = dtmodel
RandomForestClassifierModel(dtmodel :: SymbolicModelSet) = dtmodel
AdaBoostClassifierModel(dtmodel     :: SymbolicModelSet) = dtmodel

DecisionTreeRegressorModel(dtmodel  :: SymbolicModelSet) = dtmodel
RandomForestRegressorModel(dtmodel  :: SymbolicModelSet) = dtmodel

ModalDecisionTreeModel(dtmodel      :: SymbolicModelSet) = dtmodel
ModalRandomForestModel(dtmodel      :: SymbolicModelSet) = dtmodel
ModalAdaBoostModel(dtmodel          :: SymbolicModelSet) = dtmodel

XGBoostClassifierModel(dtmodel      :: SymbolicModelSet) = dtmodel
XGBoostRegressorModel(dtmodel       :: SymbolicModelSet) = dtmodel

mutable struct ModelConfig <: AbstractModelConfig
    setup      :: AbstractModelSet
    ds         :: Dataset
    classifier :: MLJ.Model
    mach       :: Union{MLJ.Machine, AbstractVector{<:MLJ.Machine}}
    model      :: Union{AbstractModel, AbstractVector{<:AbstractModel}}
    rules      :: Union{AbstractDataFrame, AbstractVector{<:AbstractDataFrame}, Nothing}
    accuracy   :: Union{AbstractFloat, AbstractVector{<:AbstractFloat}, Nothing}

    function ModelConfig(
        setup::AbstractModelSet,
        ds::Dataset,
        classifier::MLJ.Model,
        mach::Union{MLJ.Machine, AbstractVector{<:MLJ.Machine}},
        model::Union{AbstractModel, AbstractVector{<:AbstractModel}},
    )
        new(setup, ds, classifier, mach, model, nothing, nothing)
    end
end

function Base.show(io::IO, mc::ModelConfig)
    println(io, "ModelConfig:")
    println(io, "    setup      =", mc.setup)
    println(io, "    classifier =", mc.classifier)
    println(io, "    rules      =", isnothing(mc.rules) ? "nothing" : string(mc.rules))
    println(io, "    accuracy   =", isnothing(mc.accuracy) ? "nothing" : string(mc.accuracy))
end

const DEFAULT_FEATS = [maximum, minimum, mean, std]

const DEFAULT_PREPROC = (
    train_ratio = 0.8,
    valid_ratio = 1.0,
    shuffle     = true,
    stratified  = false,
    nfolds      = 6,
    rng         = TaskLocalRNG()
)

const AVAIL_MODELS = Dict(
    :decisiontree_classifier => DecisionTreeClassifierModel,
    :randomforest_classifier => RandomForestClassifierModel,
    :adaboost_classifier     => AdaBoostClassifierModel,

    :decisiontree_regressor  => DecisionTreeRegressorModel,
    :randomforest_regressor  => RandomForestRegressorModel,

    :modaldecisiontree       => ModalDecisionTreeModel,
    :modalrandomforest       => ModalRandomForestModel,
    :modaladaboost           => ModalAdaBoostModel,

    :xgboost_classifier      => XGBoostClassifierModel,
    :xgboost_regressor       => XGBoostRegressorModel,

    # :modal_decision_list => ModalDecisionLists.MLJInterface.ExtendedSequentialCovering,
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

const TUNING_PARAMS = Dict(
    :classification => (;
        resampling              = Holdout(),
        measure                 = LogLoss(tol = 2.22045e-16),
        weights                 = nothing,
        class_weights           = nothing,
        repeats                 = 1,
        operation               = nothing,
        selection_heuristic     = MLJ.MLJTuning.NaiveSelection(nothing),
        n                       = 25,
        train_best              = true,
        acceleration            = default_resource(),
        acceleration_resampling = CPU1(),
        check_measure           = true,
        cache                   = true,
    ),
    :regression => (;
        resampling              = Holdout(),
        measure                 = MLJ.RootMeanSquaredError(),
        weights                 = nothing,
        class_weights           = nothing,
        repeats                 = 1,
        operation               = nothing,
        selection_heuristic     = MLJ.MLJTuning.NaiveSelection(nothing),
        n                       = 25,
        train_best              = true,
        acceleration            = default_resource(),
        acceleration_resampling = CPU1(),
        check_measure           = true,
        cache                   = true,
    ),
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
