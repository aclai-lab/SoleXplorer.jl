# ---------------------------------------------------------------------------- #
#                                    dataset                                   #
# ---------------------------------------------------------------------------- #
"""
Abstract type for dataset configuration outputs
"""
abstract type AbstractDatasetConfig end

"""
Abstract type for dataset outputs
"""
abstract type AbstractDataset end

"""
Abstract type for dataset train, test and validation indexing
"""
abstract type AbstractIndexCollection end

"""
    DatasetInfo{F<:Base.Callable, R<:Real, I<:Integer, RNG<:AbstractRNG} <: AbstractDatasetConfig

An immutable struct containing dataset configuration and metadata.
It is included in ModelConfig and Dataset structs,
In a ModelConfig object, it is reachable through the `ds.info` field. 

# Fields
- `algo::Symbol`:
    Algorithm type, can be :classification, or :regression.
- `treatment::Symbol`: 
    Data treatment method, specify the behaviour of data reducing if dataset is composed of time-series.
    :aggregate, time-series will be reduced to a scalar (propositional case).
    :reducesize, time-series will be windowed to reduce size.
- `features::Vector{F}`: 
    Features functions applied to the dataset.
- `train_ratio::R`: 
    Ratio of training data (0-1), specify the ratio between train and test partitions,
    the higher the ratio, the more data will be used for training.
- `valid_ratio::R`: 
    Ratio of validation data (0-1), spoecify the ratio between train and validation partitions,
    the higher the ratio, the more data will be used for validation.
    If `valid_ratio` is unspecified, no validation data will be used.
- `shuffle::Bool`: 
    Whether to shuffle data during train, validation and test partitioning.
- `stratified::Bool`: 
    Whether to use cross-validation stratified sampling technique.
- `nfolds::I`: 
    Number of cross-validation folds.
- `rng::RNG`: 
    Random number generator.
- `winparams::Union{NamedTuple, Nothing}`: 
    Window parameters: NamedTuple should have the following fields:
    whole window (; type=wholewindow)
    adaptive window (type=adaptivewindow, nwindows, relative_overlap),
    moving window (type=movingwindow, nwindows, relative_overlap, window_size, window_step)
    split window (type=splitwindow, nwindows).
- `vnames::Union{Vector{Symbol}, Nothing}`: 
    Variable names, usually dataset column names.
"""
struct DatasetInfo{F<:Base.Callable, R<:Real, I<:Integer, RNG<:AbstractRNG} <: AbstractDatasetConfig
    algo        :: Symbol
    treatment   :: Symbol
    features    :: Vector{F}
    train_ratio :: R
    valid_ratio :: R
    shuffle     :: Bool
    stratified  :: Bool
    nfolds      :: I
    rng         :: RNG
    winparams   :: Union{NamedTuple, Nothing}
    vnames      :: Union{Vector{Symbol}, Nothing}
end

function DatasetInfo(
    algo::Symbol,
    treatment::Symbol,
    features::AbstractVector{F},
    train_ratio::R,
    valid_ratio::R,
    shuffle::Bool,
    stratified::Bool,
    nfolds::I,
    rng::RNG,
    winparams::Union{NamedTuple, Nothing},
    vnames::Union{AbstractVector{<:Union{AbstractString,Symbol}}, Nothing}
) where {F<:Base.Callable, R<:Real, I<:Integer, RNG<:AbstractRNG}
    # Validate ratios
    0 ≤ train_ratio ≤ 1 || throw(ArgumentError("train_ratio must be between 0 and 1"))
    0 ≤ valid_ratio ≤ 1 || throw(ArgumentError("valid_ratio must be between 0 and 1"))

    converted_vnames = isnothing(vnames) ? nothing : Vector{Symbol}(Symbol.(vnames))

    DatasetInfo{F,R,I,RNG}(
        algo, treatment, features, train_ratio, valid_ratio,
        shuffle, stratified, nfolds, rng, winparams, converted_vnames
    )
end

function Base.show(io::IO, info::DatasetInfo)
    println(io, "DatasetInfo:")
    for field in fieldnames(DatasetInfo)
        value = getfield(info, field)
        println(io, "  ", rpad(String(field) * ":", 15), value)
    end
end

"""
    TT_indexes{T<:Integer} <: AbstractVector{T}

A struct that stores indices for train-validation-test splits of a dataset,
used in Dataset struct.

# Fields
- `train::Vector{T}`: Vector of indices for the training set
- `valid::Vector{T}`: Vector of indices for the validation set
- `test::Vector{T}`:  Vector of indices for the test set
"""
struct TT_indexes{T<:Integer} <: AbstractIndexCollection
    train       :: Vector{T}
    valid       :: Vector{T}
    test        :: Vector{T}
end

function TT_indexes(
    train::AbstractVector{T},
    valid::AbstractVector{T},
    test::AbstractVector{T}
) where {T<:Integer}
    TT_indexes{T}(train, valid, test)
end

Base.show(io::IO, t::TT_indexes) = print(io, "TT_indexes(train=", t.train, ", validation=", t.valid, ", test=", t.test, ")")
Base.length(t::TT_indexes) = length(t.train) + length(t.valid) + length(t.test)

function _create_views(X, y, tt, stratified::Bool)
    if stratified
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
    return Xtrain, Xvalid, Xtest, ytrain, yvalid, ytest
end

"""
    Dataset{T<:AbstractDataFrame,S} <: AbstractDataset

An immutable struct that efficiently stores dataset splits for machine learning.

# Fields
- `X::T`: The feature matrix as a DataFrame
- `y::S`: The target vector
- `tt::Union{TT_indexes{I}, Vector{TT_indexes{I}}}`: Train-test split indices
- `info::DatasetInfo`: Dataset metadata and configuration
- `Xtrain`, `Xvalid`, `Xtest`: Data views for features
- `ytrain`, `yvalid`, `ytest`: Data views for targets
"""
struct Dataset{T<:AbstractDataFrame,S} <: AbstractDataset
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

# const WIN_PARAMS = Dict(
#     movingwindow   => (window_size = 1024, window_step = 512),
#     wholewindow    => NamedTuple(),
#     splitwindow    => (nwindows = 20),
#     adaptivewindow => (nwindows = 20, relative_overlap = 0.5)
# )

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
