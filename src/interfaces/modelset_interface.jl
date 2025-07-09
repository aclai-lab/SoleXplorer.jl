# ---------------------------------------------------------------------------- #
#                                  modelset                                    #
# ---------------------------------------------------------------------------- #
# 'ModelSet' wrappers for storing data as arguments.
# inspired by MLJ's `Machine` interface, but simplified for Sole.
mutable struct ModelSet{M} <: MLJType
    model::M
    args::Tuple{Vararg{AbstractSource}}
    fitresult
    report # dictionary of named tuples keyed on method (:fit, :predict, etc):
    state::Int

    # data # cached model-specific reformatting of args (for C=true):
    # resampled_data # cached subsample of data (for C=true):
    # frozen::Bool
    # old_rows
    # old_upstream_state
    # fit_okay::Channel{Bool} # cleared by fit!(::Node) calls; put! by `fit_only!(machine, true)` calls:

    function ModelSet(model::M, args::AbstractSource...) where M
        mach = new{M}(model, args)
        # mach.frozen = false
        mach.state = 0
        # mach.old_upstream_state = upstream(mach)
        # mach.fit_okay = Channel{Bool}(1)
        return mach
    end
end

# ---------------------------------------------------------------------------- #
#                                 constructors                                 #
# ---------------------------------------------------------------------------- #
function modelset end

function modelset(
    X         :: AbstractDataFrame,
    y         :: AbstractVector,
    model     :: MLJ.Model;
    ts_params :: NamedTuple = NamedTuple(),
    rng       :: AbstractRNG = TaskLocalRNG(),
)::ModelSet
    args = (source(X, ts_params), source(y))

    # prepare mlj model to feed the mlj machine
    # mlj_model = mljmodel(model, rng)
    ModelSet(model, args...)
end

# ---------------------------------------------------------------------------- #
#                                   methods                                    #
# ---------------------------------------------------------------------------- #
params(mach::ModelSet) = params(mach.model)

function Base.show(io::IO, mach::ModelSet)
    model = mach.model
    m = model isa Symbol ? ":$model" : model
    print(io, "machine($m, …)")
end

function Base.show(io::IO, ::MIME"text/plain", mach::ModelSet{M}) where M
    header =
        mach.state == -1 ? "serializable " :
        mach.state ==  0 ? "untrained " :
        "trained "
    header *= "ModelSet"
    println(io, header)
    println(io, "  model: $(mach.model)")
    println(io, "  args: ")
    for i in eachindex(mach.args)
        arg = mach.args[i]
        print(io, "    $i:\t$arg")
        if arg isa Source
            println(io, " \u23CE $(elscitype(arg))")
        else
            println(io)
        end
    end
end

# ---------------------------------------------------------------------------- #
#                                   Modelset                                   #
# ---------------------------------------------------------------------------- #
mutable struct ModelSetup{T<:AbstractModelType} <: AbstractModelSetup{T}
    type          :: Base.Callable
    config        :: NamedTuple
    # params        :: NamedTuple
    # features      :: Union{AbstractVector{<:Base.Callable}, Nothing}
    resample      :: Union{Resample, Nothing}
    winparams     :: WinParams
    rawmodel      :: Union{Base.Callable, Tuple{Base.Callable, Base.Callable}}
    # learn_method  :: Union{Base.Callable, Tuple{Base.Callable, Base.Callable}}
    tuning        :: Union{TuningParams, Bool}
    resultsparams :: Function
    rulesparams   :: Union{RulesParams, Bool}
    preprocess    :: NamedTuple
    measures      :: Union{Tuple, Nothing}
    # tt            :: Union{Vector{Tuple}, Nothing}
end

get_config(m::ModelSetup)                 = m.config
# get_params(m::ModelSetup)                 = m.params
# get_features(m::ModelSetup)               = m.features
get_winparams(m::ModelSetup)              = m.winparams
get_tuning(m::ModelSetup)                 = m.tuning
get_resample(m::ModelSetup)               = m.resample
get_preprocess(m::ModelSetup)             = m.preprocess
get_resultsparams(m::ModelSetup)          = m.resultsparams
get_rulesparams(m::ModelSetup)            = m.rulesparams
get_measures(m::ModelSetup)               = m.measures

get_treatment(m::ModelSetup)              = m.config.treatment

get_rawmodel(m::ModelSetup)               = m.rawmodel[1]
get_resampled_rawmodel(m::ModelSetup)     = m.rawmodel[2]
# get_learn_method(m::ModelSetup)           = m.learn_method[1]
# get_resampled_learn_method(m::ModelSetup) = m.learn_method[2]

# get_test(m::ModelSetup)                   = m.tt.test
# get_valid(m::ModelSetup)                  = m.tt.valid

set_config!(m::ModelSetup,       config::NamedTuple)                        = m.config = config
# set_params!(m::ModelSetup,       params::NamedTuple)                        = m.params = params

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

const DEFAULT_PREPROC = (
    train_ratio = 0.7,
    valid_ratio = 1.0,
    vnames      = nothing,
    modalreduce  = mean,
    rng         = TaskLocalRNG()
)

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