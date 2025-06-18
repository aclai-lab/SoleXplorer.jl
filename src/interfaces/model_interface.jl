# ---------------------------------------------------------------------------- #
#                                   Modelset                                   #
# ---------------------------------------------------------------------------- #
"""
    ModelSetup{T<:AbstractModelType} <: AbstractModelSetup{T}

A mutable structure that defines the configuration for machine learning models in the SoleXplorer framework.

`ModelSetup` encapsulates all parameters and configuration needed to initialize, train, and
evaluate a machine learning model, including hyperparameter tuning, rule extraction, and
data preprocessing options.

# Fields
- `type::Base.Callable`: Model type function, defining what kind of model will be created (e.g., decision tree, random forest).
- `config::NamedTuple`: General configuration parameters for the model and framework.
- `params::NamedTuple`: Model-specific hyperparameters.
- `features::Union{AbstractVector{<:Base.Callable}, Nothing}`: Feature extraction functions to apply, or `nothing` if no feature transformation is needed.
- `resample::Union{Resample, Nothing}`: Resampling strategy for cross-validation or train/test splitting, or `nothing` for default behavior.
- `winparams::WinParams`: Window parameters for time series or sequential data processing.
- `rawmodel::Union{Base.Callable, Tuple{Base.Callable, Base.Callable}}`: Function(s) to create the base model instance(s).
- `learn_method::Union{Base.Callable, Tuple{Base.Callable, Base.Callable}}`: Function(s) that define how model learning/training is performed.
- `tuning::Union{TuningParams, Bool}`: Hyperparameter tuning configuration, or a boolean to enable/disable tuning with default parameters.
- `rulesparams::Union{RulesParams, Bool}`: Rule extraction configuration, or a boolean to enable/disable rule extraction with default parameters.
- `preprocess::NamedTuple`: Data preprocessing configuration with options like train/validation split ratios and random seed.
"""
mutable struct ModelSetup{T<:AbstractModelType} <: AbstractModelSetup{T}
    type          :: Base.Callable
    config        :: NamedTuple
    params        :: NamedTuple
    features      :: Union{AbstractVector{<:Base.Callable}, Nothing}
    resample      :: Union{Resample, Nothing}
    winparams     :: WinParams
    rawmodel      :: Union{Base.Callable, Tuple{Base.Callable, Base.Callable}}
    learn_method  :: Union{Base.Callable, Tuple{Base.Callable, Base.Callable}}
    tuning        :: Union{TuningParams, Bool}
    resultsparams :: Function
    rulesparams   :: Union{RulesParams, Bool}
    preprocess    :: NamedTuple
    measures      :: Union{Tuple, Nothing}
    tt            :: Union{Vector{Tuple}, Nothing}
end

get_config(m::ModelSetup)                 = m.config
get_params(m::ModelSetup)                 = m.params
get_features(m::ModelSetup)               = m.features
get_winparams(m::ModelSetup)              = m.winparams
get_tuning(m::ModelSetup)                 = m.tuning
get_resample(m::ModelSetup)               = m.resample
get_preprocess(m::ModelSetup)             = m.preprocess
get_resultsparams(m::ModelSetup)          = m.resultsparams
get_rulesparams(m::ModelSetup)            = m.rulesparams
get_measures(m::ModelSetup)               = m.measures

get_pfeatures(m::ModelSetup)              = haskey(m.params, :features) ? m.params.features : nothing
get_treatment(m::ModelSetup)              = m.config.treatment

get_rawmodel(m::ModelSetup)               = m.rawmodel[1]
get_resampled_rawmodel(m::ModelSetup)     = m.rawmodel[2]
get_learn_method(m::ModelSetup)           = m.learn_method[1]
get_resampled_learn_method(m::ModelSetup) = m.learn_method[2]

# get_test(m::ModelSetup)                   = m.tt.test
# get_valid(m::ModelSetup)                  = m.tt.valid

set_config!(m::ModelSetup,       config::NamedTuple)                        = m.config = config
set_params!(m::ModelSetup,       params::NamedTuple)                        = m.params = params
set_features!(m::ModelSetup,     features::AbstractVector{<:Base.Callable}) = m.features = features
set_winparams!(m::ModelSetup,    winparams::WinParams)                      = m.winparams = winparams
set_tuning!(m::ModelSetup,       tuning::Union{TuningParams, Bool})         = m.tuning = tuning
set_resample!(m::ModelSetup,     resample::Union{Resample, Nothing})        = m.resample = resample
set_rulesparams!(m::ModelSetup,  rulesparams::Union{RulesParams, Bool})     = m.rulesparams = rulesparams
set_rawmodel!(m::ModelSetup,     rawmodel::Base.Callable)                   = m.rawmodel = rawmodel
set_learn_method!(m::ModelSetup, learn_method::Base.Callable)               = m.learn_method = learn_method
set_measures!(m::ModelSetup,     measures::Union{Tuple, Nothing})           = m.measures = measures

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
const DEFAULT_MEAS  = (log_loss,)

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
"""
    Modelset{T<:AbstractModelType} <: AbstractModelset{T}

A mutable structure that serves as the primary container for machine learning models in the SoleXplorer framework.

`Modelset` encapsulates all components of a machine learning workflow, including model configuration,
dataset, trained model, extracted rules, and performance results. It provides a unified interface
for model training, evaluation, rule extraction, and interpretation.

# Fields
- `setup::AbstractModelSetup{T}`: Model setup and configuration parameters.
- `ds::AbstractDataset`: Dataset containing features and target variables for training/testing.
- `predictor::Union{MLJ.Model, Nothing}`: The underlying MLJ model specification/definition.
- `mach::Union{MLJ.Machine, AbstractVector{<:MLJ.Machine}, Nothing}`: The fitted MLJ machine(s) 
  that contain the trained model state. May be a vector for ensemble or cross-validation models.
- `model::Union{AbstractModel, AbstractVector{<:AbstractModel}, Nothing}`: The trained model 
  instance(s). May be a vector for ensemble or cross-validation models.
- `rules::Union{Rule, AbstractVector{<:Rule}, Nothing}`: Extracted symbolic rules that represent 
  the model's decision logic in an interpretable format. May be multiple rules for ensemble models.
- `results::Union{AbstractResults, Nothing}`: Performance metrics and evaluation results.

# Constructors
```julia
Modelset(
    setup::AbstractModelSetup{T},
    ds::AbstractDataset,
    predictor::MLJ.Model,
    mach::MLJ.Machine,
    model::AbstractModel
) where {T<:AbstractModelType}

Modelset(
    setup::AbstractModelSetup{T},
    ds::Dataset
) where {T<:AbstractModelType}
"""
mutable struct Modelset{T<:AbstractModelType} <: AbstractModelset{T}
    setup      :: AbstractModelSetup{T}
    # ds         :: AbstractDataset
    predictor  :: Union{MLJ.Model,        Nothing}
    mach       :: Union{MLJ.Machine,      Nothing}
    model      :: Union{AbstractModel,    AbstractVector{<:AbstractModel}, Nothing}
    rules      :: Union{Rule,             AbstractVector{<:Rule},          Nothing}
    measures   :: Union{AbstractMeasures, Nothing}

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
        new{T}(setup, nothing, nothing, nothing, nothing, nothing)
    end
end

get_mach(m::Modelset)       = m.mach
get_mach_model(m::Modelset) = m.mach.model
get_solemodel(m::Modelset)  = m.model
get_mach_y(m::Modelset)     = m.mach.args[2]()
get_setup_meas(m::Modelset) = m.setup.measures
get_setup_tt(m::Modelset)   = m.setup.tt

function Base.show(io::IO, mc::Modelset)
    println(io, "Modelset:")
    println(io, "    setup      =", mc.setup)
    println(io, "    predictor  =", mc.predictor)
    println(io, "    rules      =", mc.rules === nothing ? "nothing" : string(mc.rules))
    # println(io, "    accuracy   =", mc.accuracy === nothing ? "nothing" : string(mc.accuracy))
end

# const tree_warn = Union{Modelset{SoleXplorer.TypeDTC}, Modelset{SoleXplorer.TypeDTR}, Modelset{SoleXplorer.TypeMDT}}