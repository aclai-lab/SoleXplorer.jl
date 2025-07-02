# ---------------------------------------------------------------------------- #
#                                   Modelset                                   #
# ---------------------------------------------------------------------------- #
mutable struct ModelSetup{T<:AbstractModelType} <: AbstractModelSetup{T}
    type          :: Base.Callable
    config        :: NamedTuple
    params        :: NamedTuple
    features      :: Union{AbstractVector{<:Base.Callable}, Nothing}
    resample      :: Union{Resample, Nothing}
    winparams     :: WinParams
    rawmodel      :: Union{Base.Callable, Tuple{Base.Callable, Base.Callable}}
    tuning        :: Union{TuningParams, Bool}
    resultsparams :: Function
    rulesparams   :: Union{RulesParams, Bool}
    preprocess    :: NamedTuple
    measures      :: Union{Tuple, Nothing}
end

get_params(             m :: ModelSetup) = m.params
get_features(           m :: ModelSetup) = m.features
get_resample(           m :: ModelSetup) = m.resample
get_winparams(          m :: ModelSetup) = m.winparams
get_tuning(             m :: ModelSetup) = m.tuning
get_rulesparams(        m :: ModelSetup) = m.rulesparams
get_preprocess(         m :: ModelSetup) = m.preprocess

# raw models
get_rawmodel(           m :: ModelSetup) = m.rawmodel[1]

# config structure
get_treatment(          m :: ModelSetup) = m.config.treatment

# params structure
get_pfeatures(          m :: ModelSetup) = haskey(m.params, :features) ? m.params.features : nothing

# preprocess structure
get_train_ratio(        m :: ModelSetup) = m.preprocess.train_ratio
get_valid_ratio(        m :: ModelSetup) = m.preprocess.valid_ratio
get_rng(                m :: ModelSetup) = m.preprocess.rng
get_vnames(             m :: ModelSetup) = m.preprocess.vnames
get_modalreduce(        m :: ModelSetup) = m.preprocess.modalreduce

# parameters settings, used in prepare_dataset
set_params!(      m :: ModelSetup, params      :: NamedTuple)                = m.params      = params
set_features!(    m :: ModelSetup, features    :: Vector{<:Base.Callable})   = m.features    = features
set_resample!(    m :: ModelSetup, resample    :: Union{Resample, Nothing})  = m.resample    = resample
set_winparams!(   m :: ModelSetup, winparams   :: WinParams)                 = m.winparams   = winparams
set_rawmodel!(    m :: ModelSetup, rawmodel    :: Base.Callable)             = m.rawmodel    = rawmodel
set_tuning!(      m :: ModelSetup, tuning      :: Union{TuningParams, Bool}) = m.tuning      = tuning
set_rulesparams!( m :: ModelSetup, rulesparams :: Union{RulesParams, Bool})  = m.rulesparams = rulesparams
set_measures!(    m :: ModelSetup, measures    :: Union{Tuple, Nothing})     = m.measures    = measures

###
# get_config(        m::ModelSetup)                 = m.config
# get_winparams(     m::ModelSetup)              = m.winparams
# get_resample(      m::ModelSetup)               = m.resample
# get_resultsparams( m :: ModelSetup)          = m.resultsparams
# get_measures(      m::ModelSetup)               = m.measures
# get_rawmodel(m::ModelSetup)               = m.rawmodel[1]
# get_resampled_rawmodel( m :: ModelSetup) = m.rawmodel[2]

# set_config!(m::ModelSetup,       config::NamedTuple)                        = m.config = config
# set_learn_method!(m::ModelSetup, learn_method::Base.Callable)               = m.learn_method = learn_method

# function Base.show(io::IO, ::MIME"text/plain", m::ModelSetup)
#     println(io, "ModelSetup")
#     println(io, "  Model type: ", m.type)
#     println(io, "  Features: ", m.features === nothing ? "None" : "$(length(m.features)) features")
#     println(io, "  Learning method: ", m.learn_method)
#     isa(m.rulesparams, RulesParams) && println(io, "  Rules extraction: ", m.rulesparams.type)
# end

function Base.show(io::IO, m::ModelSetup)
    print(io, "ModelSetup(type=$(m.type), features=$(m.features === nothing ? "None" : length(m.features)))")
end

# ---------------------------------------------------------------------------- #
#                              default parameters                              #
# ---------------------------------------------------------------------------- #
# struct TypeDTC <: AbstractModelType end
# struct TypeRFC <: AbstractModelType end
# struct TypeABC <: AbstractModelType end

DecisionTreeClassifierModel(dtmodel :: ModelSetup) = dtmodel
RandomForestClassifierModel(dtmodel :: ModelSetup) = dtmodel
AdaBoostClassifierModel(dtmodel     :: ModelSetup) = dtmodel

# struct TypeDTR <: AbstractModelType end
# struct TypeRFR <: AbstractModelType end

DecisionTreeRegressorModel(dtmodel  :: ModelSetup) = dtmodel
RandomForestRegressorModel(dtmodel  :: ModelSetup) = dtmodel

# struct TypeMDT <: AbstractModelType end
# struct TypeMRF <: AbstractModelType end
# struct TypeMAB <: AbstractModelType end

ModalDecisionTreeModel(dtmodel      :: ModelSetup) = dtmodel
ModalRandomForestModel(dtmodel      :: ModelSetup) = dtmodel
ModalAdaBoostModel(dtmodel          :: ModelSetup) = dtmodel

# struct TypeXGC <: AbstractModelType end
# struct TypeXGR <: AbstractModelType end

XGBoostClassifierModel(dtmodel      :: ModelSetup) = dtmodel
XGBoostRegressorModel(dtmodel       :: ModelSetup) = dtmodel

# grouping for calc results
# const TypeTreeForestC = Union{TypeDTC, TypeRFC, TypeABC, TypeMDT, TypeXGC}
# const TypeTreeForestR = Union{TypeDTR, TypeRFR}
# const TypeModalForest = Union{TypeMRF, TypeMAB}

const DEFAULT_FEATS = [maximum, minimum, MLJ.mean, std]
# const DEFAULT_MEAS  = (log_loss,)

const DEFAULT_PREPROC = (
    train_ratio = 0.7,
    valid_ratio = 1.0,
    vnames      = nothing,
    modalreduce  = mean,
    rng         = TaskLocalRNG()
)

# const PREPROC_KEYS = (:train_ratio, :valid_ratio, :vnames, :modalreduce, :rng)

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
#                              Modelset struct                                 #
# ---------------------------------------------------------------------------- #
mutable struct Modelset{T<:AbstractModelType} <: AbstractModelset{T}
    setup      :: AbstractModelSetup{T}
    # ds         :: AbstractDataset
    # predictor  :: Union{MLJ.Model,                       Nothing}
    # mach       :: Union{MLJ.Machine,                     Nothing}
    # fitresult  :: OptVecTuple
    type       :: OptModel
    model      :: OptVecAbsModel
    rules      :: OptRules
    measures   :: OptAbsMeas

    # function Modelset(
    #     setup      :: AbstractModelSetup{T},
    #     # ds         :: AbstractDataset,
    #     predictor  :: MLJ.Model,
    #     mach       :: MLJ.Machine,
    #     model      :: AbstractModel
    # )::Modelset where {T<:AbstractModelType}
    #     new{T}(setup, predictor, mach, model, nothing, nothing)
    # end

    function Modelset(
        setup      :: AbstractModelSetup{T},
        # ds         :: Dataset
    )::Modelset where {T<:AbstractModelType}
        new{T}(setup, nothing, nothing, nothing, nothing)
    end
end

# get_mach_model(m::Modelset)  = m.setup.type(;m.setup.params...)
get_solemodel(m::Modelset)       = m.model

get_setup_meas(m::Modelset)      = m.setup.measures
# get_yhat(m::Modelset)            = m.measures.yhat

get_tuning(m::Modelset)          = m.setup.tuning
get_type(m::Modelset)            = m.setup.type
# get_prediction_type(m::Modelset) = supertype(typeof(m.type))
get_base_score(m::Modelset)      = haskey(m.setup.params, :base_score) ? m.setup.params.base_score : nothing

function Base.show(io::IO, mc::Modelset)
    println(io, "Modelset:")
    println(io, "    setup      =", mc.setup)
    # println(io, "    predictor  =", mc.predictor)
    println(io, "    rules      =", mc.rules === nothing ? "nothing" : string(mc.rules))
    # println(io, "    accuracy   =", mc.accuracy === nothing ? "nothing" : string(mc.accuracy))
end

# const tree_warn = Union{Modelset{SoleXplorer.TypeDTC}, Modelset{SoleXplorer.TypeDTR}, Modelset{SoleXplorer.TypeMDT}}