# dataset.jl

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
const Modal  = Union{ModalDecisionTree, ModalRandomForest, ModalAdaBoost}
const MaybeAggregationInfo = Maybe{AggregationInfo}
const MaybeTuning = Maybe{Tuning}

# ---------------------------------------------------------------------------- #
#                                  defaults                                    #
# ---------------------------------------------------------------------------- #
# Return a default model appropriate for the target variable type.
# This function is used when no explicit model is provided to `setup_dataset`,
# automatically selecting between classification and regression.
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
# Set the random number generator for a model that supports it.
# This function mutates the model's `rng` field if it exists, ensuring
# reproducible results across training sessions.
function set_rng!(m::MLJ.Model, rng::AbstractRNG)::MLJ.Model
    m.rng = rng
    return m
end

# Set the random number generator for a resampling strategy.
function set_rng(r::MLJ.ResamplingStrategy, rng::AbstractRNG)::ResamplingStrategy
    typeof(r)(merge(MLJ.params(r), (rng=rng,))...)
end

# Set random number generators for tuning-related components of a model.
function set_tuning_rng!(m::MLJ.Model, rng::AbstractRNG)::MLJ.Model
    hasproperty(m.tuning, :rng) && (m.tuning.rng = rng)
    hasproperty(m.resampling, :rng) && (m.resampling = set_rng(m.resampling, rng))
    return m
end

# Set the training fraction for a resampling strategy.
function set_fraction_train(r::ResamplingStrategy, train_ratio::Real)::ResamplingStrategy
    typeof(r)(merge(MLJ.params(r), (fraction_train=train_ratio,))...)
end

# Set logical conditions (features) for modal decision tree models.
function set_conditions!(m::MLJ.Model, conditions::Tuple{Vararg{Base.Callable}})::MLJ.Model
    m.conditions = Function[conditions...]
    return m
end

"""
    code_dataset(X::AbstractDataFrame)

In-place encoding of non-numeric columns in a DataFrame to numeric codes.
"""
function code_dataset(X::AbstractDataFrame)
    for (name, col) in pairs(eachcol(X))
        if !(eltype(col) <: Number)
            X_coded = MLJ.levelcode.(categorical(col)) 
            X[!, name] = X_coded
        end
    end
    
    return X
end

"""
    code_dataset(y::AbstractVector)

In-place encoding of non-numeric target vector to numeric codes.
"""
function code_dataset(y::AbstractVector)
    if !(eltype(y) <: Number)
        eltype(y) <: Symbol && (y = string.(y))
        y = MLJ.levelcode.(categorical(y)) 
    end
    
    return y
end

"""
    code_dataset(X::AbstractDataFrame, y::AbstractVector)

Convenience method to encode both features and target simultaneously.
"""
code_dataset(X::AbstractDataFrame, y::AbstractVector) = code_dataset(X), code_dataset(y)

"""
    range(field::Union{Symbol,Expr}; kwargs...)

Wrapper for MLJ.range in hyperparameter tuning contexts.

# Arguments
- `field::Union{Symbol,Expr}`: Model field to tune
- `kwargs...`: Range specification arguments

# Returns
- Tuple of (field, kwargs) for later processing by tuning setup

This function provides a more convenient syntax for specifying hyperparameter
ranges that will be converted to proper MLJ ranges once the model is available.
"""
Base.range(field::Union{Symbol,Expr}; kwargs...) = field, kwargs...

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
    mach    :: MLJ.Machine
    pidxs   :: Vector{PartitionIdxs}
    pinfo   :: PartitionInfo
    ainfo   :: MaybeAggregationInfo
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
    mach    :: MLJ.Machine
    pidxs   :: Vector{PartitionIdxs}
    pinfo   :: PartitionInfo
    tinfo   :: TreatmentInfo
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

# ---------------------------------------------------------------------------- #
#                                 constructors                                 #
# ---------------------------------------------------------------------------- #
function _setup_dataset(
    X             :: AbstractDataFrame,
    y             :: AbstractVector,
    w             :: MaybeVector                  = nothing;
    model         :: MLJ.Model                    = _DefaultModel(y),
    resample      :: ResamplingStrategy           = Holdout(shuffle=true),
    train_ratio   :: Real                         = 0.7,
    valid_ratio   :: Real                         = 0.0,
    rng           :: AbstractRNG                  = TaskLocalRNG(),
    win           :: WinFunction                  = AdaptiveWindow(nwindows=3, relative_overlap=0.1),
    features      :: Tuple{Vararg{Base.Callable}} = (maximum, minimum),
    modalreduce   :: Base.Callable                = mean,
    tuning        :: MaybeTuning                  = nothing
)::AbstractDataSet
    # propagate user rng to every field that needs it
    hasproperty(model, :rng) && set_rng!(model, rng)
    hasproperty(resample, :rng) && (resample = set_rng(resample, rng))

    # ModalDecisionTrees package needs features to be passed in model params
    hasproperty(model, :features) && set_conditions!(model, features)
    # Holdout resampling needs to setup fraction_train parameters
    resample isa Holdout && (resample = set_fraction_train(resample, train_ratio))

    # Handle multidimensional datasets:
    # Decision point: use standard ML algorithms (requiring feature aggregation)
    # or modal logic algorithms (optionally reducing data size).
    if X[1, 1] isa AbstractArray
        treat = model isa Modal ? :reducesize : :aggregate
        X, tinfo = treatment(X; win, features, treat, modalreduce)
    else
        X = code_dataset(X)
        tinfo = nothing
    end

    ttpairs, pinfo = partition(y; resample, train_ratio, valid_ratio, rng)

    isnothing(tuning) || begin
        t_range = get_range(tuning)
        if !(t_range isa MLJ.NominalRange)
            # Convert SX.range to MLJ.range now that model is available
            range = t_range isa Tuple{Vararg{Tuple}} ? t_range : (t_range,)
            range = collect(MLJ.range(model, r[1]; r[2:end]...) for r in range)
            tuning.range = range
        end

        model = MLJ.TunedModel(
            model; 
            tuning=tuning.strategy,
            range=tuning.range,
            resampling=tuning.resampling,
            measure=tuning.measure,
            repeats=tuning.repeats
        )

        # set the model to use the same rng as the dataset
        set_tuning_rng!(model, rng)
    end

    mach = isnothing(w) ? MLJ.machine(model, X, y) : MLJ.machine(model, X, y, w)
    
    DataSet(mach, ttpairs, pinfo; tinfo)
end

"""
    setup_dataset(X, y, w=nothing; kwargs...)::AbstractDataSet

Internal function to prepare and construct a dataset warper.

# Arguments
- `X::AbstractDataFrame`: Feature data
- `y::AbstractVector`: Target variable
- `w::MaybeVector=nothing`: Optional sample weights

# Keyword Arguments
- `model::MLJ.Model=_DefaultModel(y)`: MLJ model to use
- `resample::ResamplingStrategy=Holdout(shuffle=true)`: Resampling strategy
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
dsc = setup_dataset(
    Xc, yc;
    model=DecisionTreeClassifier(),
    resample=CV(nfolds=5, shuffle=true),
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
    resample=Holdout(shuffle=true),
    train_ratio=0.75,
    rng=Xoshiro(1),
    features=(minimum, maximum),
    measures=(log_loss, accuracy, confusion_matrix, kappa)
)
```
"""
setup_dataset(args...; kwargs...) = _setup_dataset(args...; kwargs...)

"""
    setup_dataset(X::AbstractDataFrame, y::Symbol; kwargs...)::AbstractDataSet

Convenience method when target variable is a column in the feature DataFrame.
"""
function setup_dataset(
    X::AbstractDataFrame,
    y::Symbol;
    kwargs...
)::AbstractDataSet
    setup_dataset(X[!, Not(y)], X[!, y]; kwargs...)
end

"""
    length(ds::AbstractDataSet)

Return the number of partitions in the dataset.
"""
Base.length(ds::AbstractDataSet) = length(ds.pidxs)

"""
    get_X(ds::AbstractDataSet) -> DataFrame

Extract feature DataFrame from dataset's MLJ machine.
"""
get_X(ds::AbstractDataSet)::DataFrame = ds.mach.args[1].data

"""
    get_y(ds::AbstractDataSet) -> Vector

Extract target vector from dataset's MLJ machine.
"""
get_y(ds::AbstractDataSet)::Vector    = ds.mach.args[2].data

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
