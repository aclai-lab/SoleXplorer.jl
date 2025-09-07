# Dataset construction and management utilities for SoleXplorer.

# This module handles the creation of specialized dataset structures that encapsulate
# MLJ machines, partitioning information for propositional sets, including also
# treatment details for modal learning sets.

# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
"""
    AbstractDataSet

Abstract supertype for all dataset structures in SoleXplorer.

Concrete subtypes include:
- [`PropositionalDataSet`](@ref): for standard ML algorithms with aggregated features
- [`ModalDataSet`](@ref): for modal logic algorithms with temporal structure preservation
"""
abstract type AbstractDataSet end

# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
"""
    Modal = Union{ModalDecisionTree, ModalRandomForest, ModalAdaBoost}

Type alias for models that support modal logic.
"""
const Modal = Union{ModalDecisionTree, ModalRandomForest, ModalAdaBoost}

const MaybeAggregationInfo = Maybe{AggregationInfo}

# const MaybeInt = Maybe{Int}
const MaybeVector     = Maybe{AbstractVector}
const MaybeResampling = Maybe{MLJ.ResamplingStrategy}
const MaybeBalancing  = Maybe{Tuple{Vararg{<:MLJ.Model}}}
const MaybeTuning     = Maybe{MLJTuning.TuningStrategy}
const MaybeRange      = Maybe{RangeSpec}
const MaybeRng        = Maybe{AbstractRNG}

const RobustMeasure   = StatisticalMeasures.StatisticalMeasuresBase.RobustMeasure
const FussyMeasure    = StatisticalMeasures.StatisticalMeasuresBase.FussyMeasure
const EitherMeasure   = Union{RobustMeasure, FussyMeasure}

const MaybeMeasures   = Maybe{Tuple{Vararg{<:EitherMeasure}}}

# ---------------------------------------------------------------------------- #
#                                  defaults                                    #
# ---------------------------------------------------------------------------- #
# This function is used when no explicit model is provided to `model_setup`,
# automatically selecting between classification and regression.
# Return a default model appropriate for the target variable type.
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
# # Set the random number generator for a model that supports it.
# # This function mutates the model's `rng` field if it exists, ensuring
# # reproducible results across training sessions.
# function set_rng!(m::MLJ.Model, rng::AbstractRNG)::MLJ.Model
#     m.rng = rng
#     return m
# end

# # Set the random number generator for a resampling strategy.
# function set_rng!(r::MLJ.ResamplingStrategy, rng::AbstractRNG)::ResamplingStrategy
#     typeof(r)(merge(MLJ.params(r), (rng=rng,))...)
# end

# # Set random number generators for balancing-related components of a model.
# function set_balancing_rng(m::MLJ.Model, rng::AbstractRNG)::MLJ.Model
#     if hasproperty(m, :rng)
#         params = MLJ.params(m)
#         params = merge(params, (rng=rng,))
#         return Base.typename(typeof(m)).wrapper(; params...)
#     else
#         return m
#     end
# end

# Set random number generators for tuning-related components of a model.
function set_tuning_rng(t::MLJTuning.TuningStrategy, rng::AbstractRNG)::MLJTuning.TuningStrategy
    hasproperty(t, :rng) && (t.rng = rng)
    # hasproperty(t.resampling, :rng) && (t.resampling = set_rng!(t.resampling, rng))
    return t
end

# # Set the training fraction for a resampling strategy.
# function set_fraction_train(r::ResamplingStrategy, train_ratio::Real)::ResamplingStrategy
#     typeof(r)(merge(MLJ.params(r), (fraction_train=train_ratio,))...)
# end

# Set logical conditions (features) for modal decision tree models.
function set_conditions(m::MLJ.Model, conditions::Tuple{Vararg{Base.Callable}})::MLJ.Model
    m.conditions = Function[conditions...]
    return m
end

"""
    code_dataset!(X::AbstractDataFrame)

In-place encoding of non-numeric columns in a DataFrame to numeric codes.
"""
function code_dataset!(X::AbstractDataFrame)
    for (name, col) in pairs(eachcol(X))
        if !(eltype(col) <: Number)
            X_coded = MLJ.levelcode.(categorical(col)) 
            X[!, name] = X_coded
        end
    end
    
    return X
end

"""
    code_dataset!(y::AbstractVector)

In-place encoding of non-numeric target vector to numeric codes.
"""
function code_dataset!(y::AbstractVector)
    if !(eltype(y) <: Number)
        eltype(y) <: Symbol && (y = string.(y))
        y = MLJ.levelcode.(categorical(y)) 
    end
    
    return y
end

"""
    code_dataset!(X::AbstractDataFrame, y::AbstractVector)

Convenience method to encode both features and target simultaneously.
"""
code_dataset!(X::AbstractDataFrame, y::AbstractVector) = code_dataset!(X), code_dataset!(y)

# Convert treatment information (features and winparams) to aggregation information.
treat2aggr(t::TreatmentInfo)::AggregationInfo = 
    AggregationInfo(t.features, t.winparams)

# ---------------------------------------------------------------------------- #
#                          multidimensional dataset                            #
# ---------------------------------------------------------------------------- #
"""
    PropositionalDataSet{M} <: AbstractDataSet

Wrapper for standard (propositional) machine learning algorithms.

# Fields
- `mach::MLJ.Machine`: The underlying MLJ machine
- `pidxs::Vector{PartitionIdxs}`: Partition indices for train/validation/test splits
- `pinfo::PartitionInfo`: Information about the partitioning strategy
- `ainfo::MaybeAggregationInfo`: Optional aggregation information for feature extraction

The `ainfo` field is used when a multidimensional dataset is aggregated using windowing 
and feature extraction to convert temporal sequences into tabular format.

"""
mutable struct PropositionalDataSet{M} <: AbstractDataSet
    mach       :: MLJ.Machine
    resampling :: MLJ.ResamplingStrategy
    # pinfo      :: PartitionInfo
    ainfo      :: MaybeAggregationInfo
end

"""
    ModalDataSet{M} <: AbstractDataSet

Wrapper for modal logic algorithms that work with temporal structures.

# Fields
- `mach::MLJ.Machine`: The underlying MLJ machine
- `pidxs::Vector{PartitionIdxs}`: Partition indices for train/validation/test splits
- `pinfo::PartitionInfo`: Information about the partitioning strategy
- `tinfo::TreatmentInfo`: Information about temporal data treatment

The `tinfo` field  to store treatment information, such as features and window parameters,
used to reduce the dataset size while preserving temporal structure.
"""
mutable struct ModalDataSet{M} <: AbstractDataSet
    mach       :: MLJ.Machine
    resampling :: MLJ.ResamplingStrategy
    # pinfo      :: PartitionInfo
    tinfo      :: TreatmentInfo
end

"""
    show(io::IO, ds::PropositionalDataSet)

Display a PropositionalDataSet with key information about the MLJ machine,
resampling strategy, and aggregation details.
"""
function Base.show(io::IO, ds::PropositionalDataSet{M}) where M
    println(io, "PropositionalDataSet")
    println(io, "  Model: $(typeof(ds.mach.model))")
    println(io, "  Resampling: $(typeof(ds.resampling))")
    
    if !isnothing(ds.ainfo)
        println(io, "  Aggregation:")
        println(io, "    Features: $(ds.ainfo.features)")
        println(io, "    Window params: $(ds.ainfo.winparams)")
    else
        println(io, "  Aggregation: none")
    end
end

"""
    show(io::IO, ds::ModalDataSet)

Display a ModalDataSet with key information about the MLJ machine,
resampling strategy, and treatment details.
"""
function Base.show(io::IO, ds::ModalDataSet{M}) where M
    println(io, "ModalDataSet")
    println(io, "  Model: $(typeof(ds.mach.model))")
    println(io, "  Resampling: $(typeof(ds.resampling))")
    println(io, "  Treatment:")
    println(io, "    Type: $(ds.tinfo.treatment)")
    println(io, "    Features: $(ds.tinfo.features)")
    println(io, "    Window params: $(ds.tinfo.winparams)")
    if hasproperty(ds.tinfo, :modalreduce)
        println(io, "    Modal reduce: $(ds.tinfo.modalreduce)")
    end
end

"""
    DataSet(mach, pidxs, pinfo; tinfo=nothing)

Construct an appropriate dataset wrapper based on treatment information.

# Arguments
- `mach::MLJ.Machine{M}`: The underlying MLJ machine
- `pidxs::Vector{PartitionIdxs}`: Partition indices
- `pinfo::PartitionInfo`: Partition information
- `tinfo::Union{TreatmentInfo, Nothing}`: Optional treatment information

# Returns
- `PropositionalDataSet{M}` if no treatment info or aggregation treatment
- `ModalDataSet{M}` if treatment is `:reducesize`

This constructor automatically determines the appropriate dataset type based on
whether temporal structure should be preserved (modal) or aggregated (propositional).
"""
function DataSet(
    mach       :: MLJ.Machine{M},
    resampling :: MLJ.ResamplingStrategy;
    # pinfo      :: PartitionInfo;
    tinfo      :: Union{TreatmentInfo, Nothing}=nothing
) where {M<:MLJ.Model}
    isnothing(tinfo) ?
        PropositionalDataSet{M}(mach, resampling, nothing) : begin
            if tinfo.treatment == :reducesize
                ModalDataSet{M}(mach, resampling, tinfo)
            else
                ainfo = treat2aggr(tinfo)
                PropositionalDataSet{M}(mach, resampling, ainfo)
            end
        end
end

# ---------------------------------------------------------------------------- #
#                           MLJ models's extra setup                           #
# ---------------------------------------------------------------------------- #
function set_balancing(
    model     :: MLJ.Model,
    balancing :: Tuple{Vararg{<:MLJ.Model}},
    # rng       :: AbstractRNG
)::MLJ.Model
    pairs = map(enumerate(balancing)) do (i, b)
        # b = set_balancing_rng(b, rng)
        Symbol(:balancer, i) => b
    end
    MLJ.BalancedModel(; model, pairs...)
end

function set_tuning(
    model  :: MLJ.Model,
    tuning :: MLJTuning.TuningStrategy,
    range  :: MaybeRange,
    rng    :: MaybeRng
)::MLJ.Model
        if !(range isa MLJ.NominalRange)
            # Convert SX.range to MLJ.range now that model is available
            range = make_mlj_ranges(range, model)
        end

        # set the model to use the same rng as the dataset
        # tuning = set_tuning_rng(tuning, rng)

        MLJ.TunedModel(
            model; 
            tuning,
            range,
        )
end

# ---------------------------------------------------------------------------- #
#                                 constructors                                 #
# ---------------------------------------------------------------------------- #
function _setup_dataset(
    X             :: AbstractDataFrame,
    y             :: AbstractVector,
    w             :: MaybeVector                  = nothing;
    model         :: MLJ.Model                    = _DefaultModel(y),
    # resampling    :: MLJ.ResamplingStrategy       = Holdout(shuffle=true),
    balancing     :: MaybeBalancing               = nothing,
    tuning        :: MaybeTuning                  = nothing,
    range         :: MaybeRange                   = nothing,
    rng           :: MaybeRng                     = nothing,
    win           :: WinFunction                  = AdaptiveWindow(nwindows=3, relative_overlap=0.1),
    features      :: Tuple{Vararg{Base.Callable}} = (maximum, minimum),
    modalreduce   :: Base.Callable                = mean,

)::AbstractDataSet
    # propagate user rng to every field that needs it
    # model
    if !isnothing(rng)
        hasproperty(model, :rng) && (model = set_rng!(model,    rng))
        hasproperty(resampling, :rng) && (resampling = set_rng!(resampling, rng))
    end

    # ModalDecisionTrees package needs features to be passed in model params
    hasproperty(model, :features) && (model = set_conditions(model, features))
    # Holdout resampling needs to setup fraction_train parameters
    # resampling isa Holdout && (resampling = set_fraction_train(resampling, train_ratio))

    # Handle multidimensional datasets:
    # Decision point: use standard ML algorithms (requiring feature aggregation)
    # or modal logic algorithms (optionally reducing data size).
    if X[1, 1] isa AbstractArray
        treat = model isa Modal ? :reducesize : :aggregate
        X, tinfo = treatment(X; win, features, treat, modalreduce)
    else
        X = code_dataset!(X)
        tinfo = nothing
    end

    # ttpairs, pinfo = partition(y; resampling, train_ratio, valid_ratio, rng)

    # isnothing(balancing) || (model = set_balancing(model, balancing, rng))
    isnothing(balancing) || (model = set_balancing(model, balancing))
    isnothing(tuning)    || (model = set_tuning(model, tuning, range, rng))

    mach = isnothing(w) ? MLJ.machine(model, X, y) : MLJ.machine(model, X, y, w)
    
    # DataSet(mach, resampling, pinfo; tinfo)
    DataSet(mach, resampling; tinfo)
end

"""
    model_setup(X, y, w=nothing; kwargs...)::AbstractDataSet

Internal function to prepare and construct a dataset warper.

# Arguments
- `X::AbstractDataFrame`: Feature data
- `y::AbstractVector`: Target variable
- `w::MaybeVector=nothing`: Optional sample weights

# Keyword Arguments
- `model::MLJ.Model=_DefaultModel(y)`: MLJ model to use
- `resampling::ResamplingStrategy=Holdout(shuffle=true)`: Resampling strategy
- `train_ratio::Real=0.7`: Fraction of data for training
- `valid_ratio::Real=0.0`: Fraction of data for validation
- `rng::AbstractRNG=TaskLocalRNG()`: Random number generator
- `win::WinFunction=AdaptiveWindow(nwindows=3, relative_overlap=0.1)`: Windowing function
- `features::Tuple{Vararg{Base.Callable}}=(maximum, minimum)`: Feature extraction functions
- `modalreduce::Base.Callable=mean`: Reduction function for modal algorithms
- `tuning::NamedTuple=NamedTuple()`: Hyperparameter tuning specification

# Returns
- `AbstractDataSet`: Either `PropositionalDataSet` or `ModalDataSet`

This function handles the complete pipeline of dataset preparation including:
1. Model configuration and RNG propagation
2. Multidimensional data treatment (aggregation vs. modal reduction)
3. Data partitioning and resampling setup
4. Hyperparameter tuning configuration
5. MLJ Machine construction

# Example
```julia
# Standard classification dataset
using MLJ, DataFrames, SoleXplorer
Xc, yc = @load_iris
Xc = DataFrame(Xc)
range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
dsc = model_setup(
    Xc, yc;
    model=DecisionTreeClassifier(),
    resampling=CV(nfolds=5, shuffle=true),
    rng=Xoshiro(1),
    tuning=(tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=accuracy, repeats=2)    
)

# Modal time series dataset 
using SoleXplorer
using SoleData.Artifacts: load
natopsloader = NatopsLoader()
Xts, yts = load(natopsloader)
modelts = symbolic_analysis(
    Xts, yts;
    model=ModalRandomForest(),
    resampling=Holdout(shuffle=true),
    train_ratio=0.75,
    rng=Xoshiro(1),
    features=(minimum, maximum),
    measures=(log_loss, accuracy, confusion_matrix, kappa)
)
```
"""
model_setup(args...; kwargs...) = _setup_dataset(args...; kwargs...)

"""
    model_setup(X::AbstractDataFrame, y::Symbol; kwargs...)::AbstractDataSet

Convenience method when target variable is a column in the feature DataFrame.

See [`model_setup`](@ref) for detailed parameter descriptions.
"""
function model_setup(
    X::AbstractDataFrame,
    y::Symbol;
    kwargs...
)::AbstractDataSet
    model_setup(X[!, Not(y)], X[!, y]; kwargs...)
end

"""
    length(ds::AbstractDataSet)

Return the number of partitions in the dataset.
"""
Base.length(ds::AbstractDataSet) = length(ds.pidxs)

"""
    get_y_test(ds::AbstractDataSet)::AbstractVector

Extract test target values for each partition in the dataset.
"""
get_y_test(ds::AbstractDataSet)::AbstractVector = 
    [@views ds.mach.args[2].data[ds.pidxs[i].test] for i in 1:length(ds)]



"""
    get_mach(ds::AbstractDataSet)::Machine

Extract the MLJ machine from the dataset.
"""
get_mach(ds::AbstractDataSet)::Machine = ds.mach

"""
    get_mach_model(ds::AbstractDataSet)::MLJ.Model

Extract the model from the dataset's MLJ machine.
"""
get_mach_model(ds::AbstractDataSet)::MLJ.Model = ds.mach.model

"""
    get_mach_model(ds::ModalDataSet)::SupportedLogiset

Extract the logiset (if present) from the dataset's MLJ machine.
"""
get_logiset(ds::ModalDataSet)::SupportedLogiset = ds.mach.data[1].modalities[1]
