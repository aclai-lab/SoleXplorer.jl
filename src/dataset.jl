# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
abstract type AbstractDataSet end

# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
const Modal  = Union{ModalDecisionTree, ModalRandomForest, ModalAdaBoost}
const Tuning = Union{Nothing, MLJTuning.TuningStrategy}

const OptAggregationInfo = Optional{AggregationInfo}
const OptVector = Optional{AbstractVector}

# ---------------------------------------------------------------------------- #
#                                  defaults                                    #
# ---------------------------------------------------------------------------- #
# utilizzato in caso non venga specificato il modello da utilizzare
# restituisce un modello di classificazione o di regressione
# a seconda del tipo di y.
function _DefaultModel(y::AbstractVector)::MLJ.Model
    if     eltype(y) <: CLabel
        return DecisionTreeClassifier()
    elseif eltype(y) <: RLabel
        return DecisionTreeRegressor()
    else
        throw(ArgumentError("Unsupported type for y: $(eltype(y))"))
    end
end

# ---------------------------------------------------------------------------- #
#                                 utilities                                    #
# ---------------------------------------------------------------------------- #
function set_rng!(m::MLJ.Model, rng::AbstractRNG)::MLJ.Model
    m.rng = rng
    return m
end

function set_rng!(r::MLJ.ResamplingStrategy, rng::AbstractRNG)::ResamplingStrategy
    typeof(r)(merge(MLJ.params(r), (rng=rng,))...)
end

function set_tuning_rng!(m::MLJ.Model, rng::AbstractRNG)::MLJ.Model
    hasproperty(m.tuning, :rng) && (m.tuning.rng = rng)
    hasproperty(m.resampling, :rng) && (m.resampling = set_rng!(m.resampling, rng))
    return m
end

function set_fraction_train!(r::ResamplingStrategy, train_ratio::Real)::ResamplingStrategy
    typeof(r)(merge(MLJ.params(r), (fraction_train=train_ratio,))...)
end

function set_conditions!(m::MLJ.Model, conditions::Tuple{Vararg{<:Base.Callable}})::MLJ.Model
    m.conditions = Function[conditions...]
    return m
end

function code_dataset!(X::AbstractDataFrame)
    for (name, col) in pairs(eachcol(X))
        if !(eltype(col) <: Number)
            X_coded = MLJ.levelcode.(categorical(col)) 
            X[!, name] = X_coded
        end
    end
    
    return X
end

function code_dataset!(y::AbstractVector)
    if !(eltype(y) <: Number)
        eltype(y) <: Symbol && (y = string.(y))
        y = MLJ.levelcode.(categorical(y)) 
    end
    
    return y
end

code_dataset!(X::AbstractDataFrame, y::AbstractVector) = code_dataset!(X), code_dataset!(y)

# wrapper per MLJ.range in tuning
Base.range(field::Union{Symbol,Expr}; kwargs...) = field, kwargs...

treat2aggr(t::TreatmentInfo)::AggregationInfo = 
    AggregationInfo(t.features, t.winparams)

# ---------------------------------------------------------------------------- #
#                          multidimensional dataset                            #
# ---------------------------------------------------------------------------- #
mutable struct PropositionalDataSet{M} <: AbstractDataSet
    mach    :: MLJ.Machine
    pidxs   :: Vector{PartitionIdxs}
    pinfo   :: PartitionInfo
    ainfo   :: OptAggregationInfo
end

mutable struct ModalDataSet{M} <: AbstractDataSet
    mach    :: MLJ.Machine
    pidxs   :: Vector{PartitionIdxs}
    pinfo   :: PartitionInfo
    tinfo   :: TreatmentInfo
end

function DataSet(
    mach    :: MLJ.Machine{M},
    pidxs   :: Vector{PartitionIdxs},
    pinfo   :: PartitionInfo;
    tinfo   :: Union{TreatmentInfo, Nothing}=nothing
) where {M<:MLJ.Model}
    isnothing(tinfo) ?
        PropositionalDataSet{M}(mach, pidxs, pinfo, nothing) : begin
            if tinfo.treatment == :reducesize
                ModalDataSet{M}(mach, pidxs, pinfo, tinfo)
            else
                ainfo = treat2aggr(tinfo)
                PropositionalDataSet{M}(mach, pidxs, pinfo, ainfo)
            end
        end
end

const EitherDataSet = Union{PropositionalDataSet, ModalDataSet}

# ---------------------------------------------------------------------------- #
#                                 constructors                                 #
# ---------------------------------------------------------------------------- #
function _prepare_dataset(
    X             :: AbstractDataFrame,
    y             :: AbstractVector,
    w             :: OptVector               = nothing;
    model         :: MLJ.Model               = _DefaultModel(y),
    resample      :: ResamplingStrategy      = Holdout(shuffle=true),
    train_ratio   :: Real                    = 0.7,
    valid_ratio   :: Real                    = 0.0,
    rng           :: AbstractRNG             = TaskLocalRNG(),
    win           :: WinFunction             = AdaptiveWindow(nwindows=3, relative_overlap=0.1),
    features      :: Tuple{Vararg{<:Base.Callable}} = (maximum, minimum),
    modalreduce   :: Base.Callable           = mean,
    tuning        :: NamedTuple              = NamedTuple()
)::AbstractDataSet
    # propagate user rng to every field that needs it
    # model
    hasproperty(model,    :rng) && (model    = set_rng!(model,    rng))
    hasproperty(resample, :rng) && (resample = set_rng!(resample, rng))

    # ModalDecisionTrees package needs features to be passed in model params
    hasproperty(model, :features) && (model = set_conditions!(model, features))
    # Holdout resampling needs to setup fraction_train parameters
    resample isa Holdout && (resample = set_fraction_train!(resample, train_ratio))

    # questo if è relativo a dataset multidimensionali.
    # qui si decide come trattare tali dataset:
    # abbiamo 2 soluzioni: utilizzare i normali algoritmi di machine learning, che accettano
    # solo dataset numerici, oppure utilizzare logica modale.
    # nel primo caso i dataset verranno ridotti a dataset numerici,
    # applicando una feature (massimo, minimo, media, ...) su un numero definito di finestre.
    # nel secondo caso, per economia di calcolo, verranno ridotti per finestre,
    # secondo un parametro di riduzione 'modalreduce' tipicamente mean, comunque definito dall'utente.
    if X[1, 1] isa AbstractArray
        treat = model isa Modal ? :reducesize : :aggregate
        X, tinfo = treatment(X; win, features, treat, modalreduce)
    else
        X = code_dataset!(X)
        tinfo = nothing
    end

    ttpairs, pinfo = partition(y; resample, train_ratio, valid_ratio, rng)

    isempty(tuning) || begin
        if !(tuning.range isa MLJ.NominalRange)
            # converti i SX.range in MLJ.range, ora che è disponibile il modello
            range = tuning.range isa Tuple{Vararg{<:Tuple}} ? tuning.range : (tuning.range,)
            range = collect(MLJ.range(model, r[1]; r[2:end]...) for r in range)
            tuning = merge(tuning, (range=range,))
        end

        model = MLJ.TunedModel(model; tuning...)

        # set the model to use the same rng as the dataset
        model = set_tuning_rng!(model, rng)
    end

    mach = isnothing(w) ? MLJ.machine(model, X, y) : MLJ.machine(model, X, y, w)
    
    DataSet(mach, ttpairs, pinfo; tinfo)
end

setup_dataset(args...; kwargs...) = _prepare_dataset(args...; kwargs...)

# y is not a vector, but a symbol that identifies a column in X
function setup_dataset(
    X::AbstractDataFrame,
    y::Symbol;
    kwargs...
)::AbstractDataSet
    setup_dataset(X[!, Not(y)], X[!, y]; kwargs...)
end

Base.length(ds::EitherDataSet) = length(ds.pidxs)

get_y_test(ds::EitherDataSet)::AbstractVector = 
    [@views ds.mach.args[2].data[ds.pidxs[i].test] for i in 1:length(ds)]

get_mach_model(ds::EitherDataSet)::MLJ.Model = ds.mach.model
