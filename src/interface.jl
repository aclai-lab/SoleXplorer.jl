# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
"""
Abstract type for dataset configuration outputs
"""
abstract type AbstractDatasetSetup end

"""
Abstract type for dataset outputs
"""
abstract type AbstractDataset end

"""
Abstract type for dataset train, test and validation indexing
"""
abstract type AbstractIndexCollection end

"""
Abstract type for model configuration and parameters
"""
abstract type AbstractModelSetup end

"""
Abstract type for fitted model configurations
"""
abstract type AbstractModelset end

# ---------------------------------------------------------------------------- #
#                                     types                                    #
# ---------------------------------------------------------------------------- #
const Cat_Value = Union{AbstractString, Symbol, CategoricalValue}
const Reg_Value = Number
const Y_Value   = Union{Cat_Value, Reg_Value}

const Rule      = Union{SoleModels.ClassificationRule, SoleModels.DecisionSet}

struct RulesParams
    type         :: SoleModels.RuleExtractor
    params       :: NamedTuple
end

# ---------------------------------------------------------------------------- #
#                                  dataset info                                #
# ---------------------------------------------------------------------------- #
"""
    DatasetInfo <: AbstractDatasetSetup

An immutable struct containing dataset configuration and metadata for machine learning tasks.
`DatasetInfo` provides all the necessary information about how a dataset is processed,
partitioned, and what features are extracted from it.

# Fields
- `algo::Symbol`: Algorithm type:
  - `:classification`: For categorical target variables
  - `:regression`: For numerical target variables

- `treatment::Symbol`: Data treatment method for time-series data:
  - `:aggregate`: Reduces time-series to scalar features (propositional approach)
  - `:reducesize`: Windows time-series to reduce dimensions while preserving temporal structure

- `features::Vector{<:Base.Callable}`: Feature extraction functions applied to the dataset.
  Each function should accept a vector/array and return a scalar value (e.g., `mean`, `std`, `maximum`).

- `train_ratio::Real`: Proportion of data used for training (range: 0-1).
  Controls the train/test split ratio: higher values allocate more data for training.

- `valid_ratio::Real`: Proportion of training data used for validation (range: 0-1).
  - When `1.0`: No separate validation set is created (empty array)
  - When `< 1.0`: Creates validation set from the training portion

- `shuffle::Bool`: Whether to randomly shuffle data before partitioning:
  - `true`: Randomizes data order for better generalization
  - `false`: Preserves original data order (useful for time-series with temporal dependencies)

- `stratified::Bool`: Whether to use stratified sampling for cross-validation:
  - `true`: Maintains class distribution across folds (for classification tasks)
  - `false`: Simple random sampling without preserving class ratios

- `nfolds::Int`: Number of cross-validation folds when `stratified=true`.
  Higher values give more robust performance estimates but increase computation time.

- `rng::AbstractRNG`: Random number generator for reproducible partitioning and shuffling.

- `winparams::SoleFeatures.WinParams`: Windowing parameters for time-series processing:
  - `type`: Window function type (`wholewindow`, `adaptivewindow`, `movingwindow`, `splitwindow`)
  - Additional parameters specific to each window type:
    - `wholewindow`: Uses entire time-series (no parameters needed)
    - `adaptivewindow`: Uses `nwindows` and `relative_overlap`
    - `movingwindow`: Uses `window_size` and `window_step`
    - `splitwindow`: Uses `nwindows` for equal divisions

- `vnames::Union{Vector{<:AbstractString}, Nothing}`: Variable/column names.
  When `nothing`, column indices are used as identifiers.
"""
struct DatasetInfo <: AbstractDatasetSetup
    algo        :: Symbol
    treatment   :: Symbol
    reducefunc  :: Union{Nothing, <:Base.Callable}
    features    :: Vector{<:Base.Callable}
    train_ratio :: Real
    valid_ratio :: Real
    shuffle     :: Bool
    stratified  :: Bool
    nfolds      :: Int
    rng         :: AbstractRNG
    winparams   :: SoleFeatures.WinParams
    vnames      :: Union{Vector{<:AbstractString}, Nothing}

    function DatasetInfo(
        algo        :: Symbol,
        treatment   :: Symbol,
        reducefunc  :: Union{Nothing, <:Base.Callable},
        features    :: Vector{<:Base.Callable},
        train_ratio :: Real,
        valid_ratio :: Real,
        shuffle     :: Bool,
        stratified  :: Bool,
        nfolds      :: Int,
        rng         :: AbstractRNG,
        winparams   :: SoleFeatures.WinParams,
        vnames      :: Union{Vector{<:AbstractString}, Nothing}
    )::DatasetInfo
        # Validate ratios
        0 ≤ train_ratio ≤ 1 || throw(ArgumentError("train_ratio must be between 0 and 1"))
        0 ≤ valid_ratio ≤ 1 || throw(ArgumentError("valid_ratio must be between 0 and 1"))

        new(
            algo, treatment, reducefunc, features, train_ratio, valid_ratio,
            shuffle, stratified, nfolds, rng, winparams, vnames
        )
    end
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
    train :: Vector{T}
    valid :: Vector{T}
    test  :: Vector{T}
end

function TT_indexes(
    train :: AbstractVector{T},
    valid :: AbstractVector{T},
    test  :: AbstractVector{T}
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
    Dataset{T<:AbstractMatrix,S} <: AbstractDataset

An immutable struct that efficiently stores dataset splits for machine learning.

# Fields
- `X::T`: The feature matrix as a DataFrame
- `y::S`: The target vector
- `tt::Union{TT_indexes{I}, Vector{TT_indexes{I}}}`: Train-test split indices
- `info::DatasetInfo`: Dataset metadata and configuration
- `Xtrain`, `Xvalid`, `Xtest`: Data views for features
- `ytrain`, `yvalid`, `ytest`: Data views for targets
"""
struct Dataset{T<:AbstractMatrix,S} <: AbstractDataset
    X           :: T
    y           :: S
    tt          :: Union{TT_indexes, AbstractVector{<:TT_indexes}}
    info        :: DatasetInfo
    Xtrain      :: Union{AbstractMatrix, Vector{<:AbstractMatrix}}
    Xvalid      :: Union{AbstractMatrix, Vector{<:AbstractMatrix}}
    Xtest       :: Union{AbstractMatrix, Vector{<:AbstractMatrix}}
    ytrain      :: Union{SubArray{<:eltype(S)}, Vector{<:SubArray{<:eltype(S)}}}
    yvalid      :: Union{SubArray{<:eltype(S)}, Vector{<:SubArray{<:eltype(S)}}}
    ytest       :: Union{SubArray{<:eltype(S)}, Vector{<:SubArray{<:eltype(S)}}}

    function Dataset(X::T, y::S, tt, info) where {T<:AbstractMatrix,S}
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
#                                   Modelset                                   #
# ---------------------------------------------------------------------------- #
mutable struct ModelSetup <: AbstractModelSetup
    type         :: Base.Callable
    config       :: NamedTuple
    params       :: NamedTuple
    features     :: Union{AbstractVector{<:Base.Callable}, Nothing}
    winparams    :: SoleFeatures.WinParams
    learn_method :: Union{Base.Callable, Tuple{Base.Callable, Base.Callable}}
    tuning       :: NamedTuple
    rulesparams  :: RulesParams
    preprocess   :: NamedTuple
end

function Base.show(io::IO, ::MIME"text/plain", m::ModelSetup)
    println(io, "ModelSetup")
    println(io, "  Model type: ", m.type)
    println(io, "  Features: ", isnothing(m.features) ? "None" : "$(length(m.features)) features")
    println(io, "  Learning method: ", typeof(m.learn_method))
    println(io, "  Rules extraction: ", typeof(m.rulesparams.type))
end

function Base.show(io::IO, m::ModelSetup)
    print(io, "ModelSetup(type=$(m.type), features=$(isnothing(m.features) ? "None" : length(m.features)))")
end

# ---------------------------------------------------------------------------- #
#                              default parameters                              #
# ---------------------------------------------------------------------------- #
DecisionTreeClassifierModel(dtmodel :: ModelSetup) = dtmodel
RandomForestClassifierModel(dtmodel :: ModelSetup) = dtmodel
AdaBoostClassifierModel(dtmodel     :: ModelSetup) = dtmodel

DecisionTreeRegressorModel(dtmodel  :: ModelSetup) = dtmodel
RandomForestRegressorModel(dtmodel  :: ModelSetup) = dtmodel

ModalDecisionTreeModel(dtmodel      :: ModelSetup) = dtmodel
ModalRandomForestModel(dtmodel      :: ModelSetup) = dtmodel
ModalAdaBoostModel(dtmodel          :: ModelSetup) = dtmodel

XGBoostClassifierModel(dtmodel      :: ModelSetup) = dtmodel
XGBoostRegressorModel(dtmodel       :: ModelSetup) = dtmodel

const DEFAULT_FEATS = [maximum, minimum, mean, std]

const DEFAULT_PREPROC = (
    train_ratio = 0.8,
    valid_ratio = 1.0,
    shuffle     = true,
    stratified  = false,
    nfolds      = 6,
    rng         = TaskLocalRNG()
)

const MODEL_KEYS   = (:type, :params, :features, :winparams, :rulesparams)
const PREPROC_KEYS = (:train_ratio, :valid_ratio, :shuffle, :stratified, :nfolds, :rng)

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

# const AVAIL_WINS       = (movingwindow, wholewindow, splitwindow, adaptivewindow)
const AVAIL_TREATMENTS = (:aggregate, :reducesize)

# const WIN_PARAMS = Dict(
#     movingwindow   => (window_size = 1024, window_step = 512),
#     wholewindow    => NamedTuple(),
#     splitwindow    => (nwindows = 20),
#     adaptivewindow => (nwindows = 20, relative_overlap = 0.5)
# )

# ---------------------------------------------------------------------------- #
#                                    rules                                     #
# ---------------------------------------------------------------------------- #
const AVAIL_RULES = (PlainRuleExtractor(), InTreesRuleExtractor())

const RULES_PARAMS = Dict{SoleModels.RuleExtractor, NamedTuple}(
    PlainRuleExtractor()   => (
        compute_metrics         = false,
        metrics_kwargs          = (;),
        use_shortforms          = true,
        use_leftmostlinearform  = nothing,
        normalize               = false,
        normalize_kwargs        = (allow_atom_flipping=true, rotate_commutatives=false),
        scalar_simplification   = false,
        force_syntaxtree        = false,
        min_coverage            = nothing,
        min_ncovered            = nothing,
        min_ninstances          = nothing,
        min_confidence          = nothing,
        min_lift                = nothing,
        metric_filter_callback  = nothing
    ),
    InTreesRuleExtractor() => (
        prune_rules             = true,
        pruning_s               = nothing,
        pruning_decay_threshold = nothing,
        rule_selection_method   = :CBC,
        rule_complexity_metric  = :natoms,
        max_rules               = -1,
        min_coverage            = nothing,
        silent                  = true,
        rng                     = Random.MersenneTwister(1),
        return_info             = false
    )
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

# ---------------------------------------------------------------------------- #
#                              Modelset struct                                 #
# ---------------------------------------------------------------------------- #
mutable struct Modelset <: AbstractModelset
    setup      :: AbstractModelSetup
    ds         :: Dataset
    classifier :: Union{MLJ.Model,     Nothing}
    mach       :: Union{MLJ.Machine,   Nothing}
    model      :: Union{AbstractModel, Nothing}
    rules      :: Union{Rule,          Nothing}
    accuracy   :: Union{AbstractFloat, Nothing}

    function Modelset(
        setup      :: AbstractModelSetup,
        ds         :: Dataset,
        classifier :: MLJ.Model,
        mach       :: MLJ.Machine,
        model      :: AbstractModel
    )
        new(setup, ds, classifier, mach, model, nothing, nothing)
    end

    function Modelset(
        setup      :: AbstractModelSetup,
        ds         :: Dataset
    )
        new(setup, ds, nothing, nothing, nothing, nothing, nothing)
    end
end

function Base.show(io::IO, mc::Modelset)
    println(io, "Modelset:")
    println(io, "    setup      =", mc.setup)
    println(io, "    classifier =", mc.classifier)
    println(io, "    rules      =", isnothing(mc.rules) ? "nothing" : string(mc.rules))
    println(io, "    accuracy   =", isnothing(mc.accuracy) ? "nothing" : string(mc.accuracy))
end
