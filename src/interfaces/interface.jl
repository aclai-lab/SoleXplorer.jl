# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
"""
Abstract type for model type
"""
abstract type AbstractModelType end

"""
Abstract type for model configuration and parameters
"""
abstract type AbstractModelSetup{T<:AbstractModelType} end

modeltype(::AbstractModelSetup{T}) where {T} = T

"""
Abstract type for fitted model configurations
"""
abstract type AbstractModelset{T<:AbstractModelType} end

modeltype(::AbstractModelset{T}) where {T} = T

"""
Abstract type for results output
"""
abstract type AbstractResults end

"""
Abstract type for type/params structs
"""
abstract type AbstractTypeParams end

# ---------------------------------------------------------------------------- #
#                                     types                                    #
# ---------------------------------------------------------------------------- #
const Cat_Value = Union{AbstractString, Symbol, CategoricalValue}
const Reg_Value = Number
const Y_Value   = Union{Cat_Value, Reg_Value}

# TODO remove this
const Rule      = Union{DecisionList, DecisionEnsemble, DecisionSet}

struct Resample <: AbstractTypeParams
    type        :: Base.Callable
    params      :: NamedTuple
end

struct TuningStrategy <: AbstractTypeParams
    type        :: Base.Callable
    params      :: NamedTuple
end

struct TuningParams <: AbstractTypeParams
    method      :: TuningStrategy
    params      :: NamedTuple
    ranges      :: Tuple{Vararg{Base.Callable}}
end

struct RulesParams <: AbstractTypeParams
    type        :: Symbol
    params      :: NamedTuple
end

# ---------------------------------------------------------------------------- #
#                                   Modelset                                   #
# ---------------------------------------------------------------------------- #
mutable struct ModelSetup{T<:AbstractModelType} <: AbstractModelSetup{T}
    type         :: Base.Callable
    config       :: NamedTuple
    params       :: NamedTuple
    features     :: Union{AbstractVector{<:Base.Callable}, Nothing}
    resample     :: Union{Resample, Nothing}
    winparams    :: SoleFeatures.WinParams
    rawmodel     :: Union{Base.Callable, Tuple{Base.Callable, Base.Callable}}
    learn_method :: Union{Base.Callable, Tuple{Base.Callable, Base.Callable}}
    tuning       :: Union{TuningParams, Bool}
    rulesparams  :: Union{RulesParams, Bool}
    preprocess   :: NamedTuple
end

get_config(m::ModelSetup)                 = m.config
get_params(m::ModelSetup)                 = m.params
get_features(m::ModelSetup)               = m.features
get_winparams(m::ModelSetup)              = m.winparams
get_tuning(m::ModelSetup)                 = m.tuning
get_resample(m::ModelSetup)               = m.resample
get_preprocess(m::ModelSetup)             = m.preprocess
get_rulesparams(m::ModelSetup)            = m.rulesparams

get_pfeatures(m::ModelSetup)              = m.params.features
get_treatment(m::ModelSetup)              = m.config.treatment
get_algo(m::ModelSetup)                   = m.config.algo

get_rawmodel(m::ModelSetup)               = m.rawmodel[1]
get_resampled_rawmodel(m::ModelSetup)     = m.rawmodel[2]
get_learn_method(m::ModelSetup)           = m.learn_method[1]
get_resampled_learn_method(m::ModelSetup) = m.learn_method[2]

set_config!(m::ModelSetup,       config::NamedTuple)                               = m.config = config
set_params!(m::ModelSetup,       params::NamedTuple)                               = m.params = params
set_features!(m::ModelSetup,     features::Union{AbstractVector{<:Base.Callable}}) = m.features = features
set_winparams!(m::ModelSetup,    winparams::SoleFeatures.WinParams)                = m.winparams = winparams
set_tuning!(m::ModelSetup,       tuning::Union{TuningParams, Bool})                = m.tuning = tuning
set_resample!(m::ModelSetup,     resample::Union{Resample, Nothing})               = m.resample = resample
set_rulesparams!(m::ModelSetup,  rulesparams::Union{RulesParams, Bool})            = m.rulesparams = rulesparams
set_rawmodel!(m::ModelSetup,     rawmodel::Base.Callable)                          = m.rawmodel = rawmodel
set_learn_method!(m::ModelSetup, learn_method::Base.Callable)                      = m.learn_method = learn_method

function Base.show(io::IO, ::MIME"text/plain", m::ModelSetup)
    println(io, "ModelSetup")
    println(io, "  Model type: ", m.type)
    println(io, "  Features: ", isnothing(m.features) ? "None" : "$(length(m.features)) features")
    println(io, "  Learning method: ", m.learn_method)
    isa(m.rulesparams, RulesParams) && println(io, "  Rules extraction: ", m.rulesparams.type)
end

function Base.show(io::IO, m::ModelSetup)
    print(io, "ModelSetup(type=$(m.type), features=$(isnothing(m.features) ? "None" : length(m.features)))")
end

# ---------------------------------------------------------------------------- #
#                              default parameters                              #
# ---------------------------------------------------------------------------- #
struct TypeDTC <: AbstractModelType end
struct TypeRFC <: AbstractModelType end
struct TypeABC <: AbstractModelType end

DecisionTreeClassifierModel(dtmodel :: ModelSetup) = dtmodel
RandomForestClassifierModel(dtmodel :: ModelSetup) = dtmodel
AdaBoostClassifierModel(dtmodel     :: ModelSetup) = dtmodel

struct TypeDTR <: AbstractModelType end
struct TypeRFR <: AbstractModelType end

DecisionTreeRegressorModel(dtmodel  :: ModelSetup) = dtmodel
RandomForestRegressorModel(dtmodel  :: ModelSetup) = dtmodel

struct TypeMDT <: AbstractModelType end
struct TypeMRF <: AbstractModelType end
struct TypeMAB <: AbstractModelType end

ModalDecisionTreeModel(dtmodel      :: ModelSetup) = dtmodel
ModalRandomForestModel(dtmodel      :: ModelSetup) = dtmodel
ModalAdaBoostModel(dtmodel          :: ModelSetup) = dtmodel

struct TypeXGC <: AbstractModelType end
struct TypeXGR <: AbstractModelType end

XGBoostClassifierModel(dtmodel      :: ModelSetup) = dtmodel
XGBoostRegressorModel(dtmodel       :: ModelSetup) = dtmodel

# grouping for calc results
const TypeTreeForestC = Union{TypeDTC, TypeRFC, TypeABC, TypeMDT, TypeXGC}
const TypeTreeForestR = Union{TypeDTR, TypeRFR}
const TypeModalForest = Union{TypeMRF, TypeMAB}

const DEFAULT_MODEL_SETUP = (type=:decisiontree,)

const DEFAULT_FEATS = [maximum, minimum, StatsBase.mean, std]

const DEFAULT_PREPROC = (
    train_ratio = 0.8,
    valid_ratio = 1.0,
    rng         = TaskLocalRNG()
)

const PREPROC_KEYS = (:train_ratio, :valid_ratio, :rng)

const AVAIL_MODELS = Dict{Symbol,Function}(
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

# const AVAIL_TREATMENTS = (:aggregate, :reducesize)

# ---------------------------------------------------------------------------- #
#                              Results structs                                 #
# ---------------------------------------------------------------------------- #
struct ClassResults <: AbstractResults
    accuracy   :: AbstractFloat
    # rules      :: Union{Rule,          Nothing}
    # metrics    :: Union{AbstractVector, Nothing}
    # feature_importance :: Union{AbstractVector, Nothing}
    # predictions:: Union{AbstractVector, Nothing}
end

# function Base.show(io::IO, ::MIME"text/plain", r::ClassResults)
#     println(io, "Results")
#     println(io, "  Accuracy: ", r.accuracy)
#     println(io, "  Rules: ", r.rules)
#     println(io, "  Feature importance: ", r.feature_importance)
#     println(io, "  Predictions: ", r.predictions)
# end

# Base.show(io::IO, r::ClassResults) = print(io, "Results(accuracy=$(r.accuracy), rules=$(r.rules))")

struct RegResults <: AbstractResults
    accuracy   :: AbstractFloat
    # rules      :: Union{Rule,          Nothing}
    # metrics    :: Union{AbstractVector, Nothing}
    # feature_importance :: Union{AbstractVector, Nothing}
    # predictions:: Union{AbstractVector, Nothing}
end

const RESULTS = Dict{Symbol,DataType}(
    :classification => ClassResults,
    :regression     => RegResults
)

# ---------------------------------------------------------------------------- #
#                              Modelset struct                                 #
# ---------------------------------------------------------------------------- #
mutable struct Modelset{T<:AbstractModelType} <: AbstractModelset{T}
    setup      :: AbstractModelSetup{T}
    ds         :: AbstractDataset
    classifier :: Union{MLJ.Model,       Nothing}
    mach       :: Union{MLJ.Machine,   AbstractVector{<:MLJ.Machine},   Nothing}
    model      :: Union{AbstractModel, AbstractVector{<:AbstractModel}, Nothing}
    rules      :: Union{Rule,          AbstractVector{<:Rule},          Nothing}
    results    :: Union{AbstractResults, Nothing}

    function Modelset(
        setup      :: AbstractModelSetup{T},
        ds         :: AbstractDataset,
        classifier :: MLJ.Model,
        mach       :: MLJ.Machine,
        model      :: AbstractModel
    ) where {T<:AbstractModelType}
        new{T}(setup, ds, classifier, mach, model, nothing, nothing)
    end

    function Modelset(
        setup      :: AbstractModelSetup{T},
        ds         :: Dataset
    ) where {T<:AbstractModelType}
        new{T}(setup, ds, nothing, nothing, nothing, nothing, nothing)
    end
end

function Base.show(io::IO, mc::Modelset)
    println(io, "Modelset:")
    println(io, "    setup      =", mc.setup)
    println(io, "    classifier =", mc.classifier)
    println(io, "    rules      =", isnothing(mc.rules) ? "nothing" : string(mc.rules))
    # println(io, "    accuracy   =", isnothing(mc.accuracy) ? "nothing" : string(mc.accuracy))
end

# ---------------------------------------------------------------------------- #
#                                   resample                                   #
# ---------------------------------------------------------------------------- #
const AVAIL_RESAMPLES = (CV, Holdout, StratifiedCV, TimeSeriesCV)

const RESAMPLE_PARAMS = Dict{DataType,NamedTuple}(
    CV           => (
        nfolds         = 6,
        shuffle        = true,
        rng            = TaskLocalRNG()
    ),
    Holdout      => (
        fraction_train = 0.7,
        shuffle        = true,
        rng            = TaskLocalRNG()
    ),
    StratifiedCV => (
        nfolds         = 6,
        shuffle        = true,
        rng            = TaskLocalRNG()
    ),
    TimeSeriesCV => (
        nfolds         = 4,
    )
)

# ---------------------------------------------------------------------------- #
#                                   tuning                                     #
# ---------------------------------------------------------------------------- #
const AVAIL_TUNING_METHODS = (grid, randomsearch, latinhypercube, treeparzen, particleswarm, adaptiveparticleswarm)

const TUNING_METHODS_PARAMS = Dict{Union{DataType, UnionAll},NamedTuple}(
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

const TUNING_PARAMS = Dict{Symbol,NamedTuple}(
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
