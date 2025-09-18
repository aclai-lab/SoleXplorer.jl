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

See also: [`setup_dataset`](@ref)
"""
abstract type AbstractDataSet end

# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
const Modal  = Union{ModalDecisionTree, ModalRandomForest, ModalAdaBoost}
const MaybeAggregationInfo = Maybe{AggregationInfo}
const MaybeTuning = Maybe{Tuning}
const MaybeTreatInfo = Maybe{TreatmentInfo}

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
#                                   set rng                                    #
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

# Set logical conditions (features) for modal decision tree models.
function set_conditions!(m::MLJ.Model, conditions::Tuple{Vararg{Base.Callable}})::MLJ.Model
    m.conditions = Function[conditions...]
    return m
end

# ---------------------------------------------------------------------------- #
#                                 code dataset                                 #
# ---------------------------------------------------------------------------- #
"""
    code_dataset(X::AbstractDataFrame)

In-place encoding of non-numeric columns in a DataFrame to numeric codes.
"""
function code_dataset(X::AbstractDataFrame)
    for (name, col) in pairs(eachcol(X))
        if !(eltype(col) <: Number)
            X[!, name] = MLJ.levelcode.(categorical(col)) 
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

# ---------------------------------------------------------------------------- #
#                          multidimensional dataset                            #
# ---------------------------------------------------------------------------- #
"""
    PropositionalDataSet{M} <: AbstractDataSet

Dataset wrapper for standard machine learning algorithms that work with tabular/propositional data.

This dataset type encapsulates an MLJ machine along with partitioning information and optional
aggregation metadata. It is used when working with traditional ML models that require flattened
feature representations, typically created by aggregating multidimensional data.

# Fields
- `mach::MLJ.Machine`: MLJ machine containing the model, training data, and cache
- `pidxs::Vector{PartitionIdxs}`: Partition indices for train/test splits across folds
- `pinfo::PartitionInfo`: Metadata about the partitioning strategy used
- `ainfo::MaybeAggregationInfo`: Optional aggregation information for multidimensional data

# Type Parameter
- `M`: The type of the MLJ model contained in the machine

See also: [`ModalDataSet`](@ref), [`setup_dataset`](@ref), [`AbstractDataSet`](@ref)
"""
mutable struct PropositionalDataSet{M} <: AbstractDataSet
    mach    :: MLJ.Machine
    pidxs   :: Vector{PartitionIdxs}
    pinfo   :: PartitionInfo
    ainfo   :: MaybeAggregationInfo
end

"""
    ModalDataSet{M} <: AbstractDataSet

Dataset wrapper for modal logic algorithms that preserve temporal/structural relationships.

This dataset type is designed for modal learning algorithms that can work directly with
multidimensional time series or structured data without requiring feature aggregation.
It maintains treatment information that describes how the original data structure is preserved.

# Fields
- `mach::MLJ.Machine`: MLJ machine containing the modal model, training data, and cache
- `pidxs::Vector{PartitionIdxs}`: Partition indices for train/test splits across folds
- `pinfo::PartitionInfo`: Metadata about the partitioning strategy used  
- `tinfo::TreatmentInfo`: Treatment information describing data structure preservation

# Type Parameter
- `M`: The type of the modal MLJ model contained in the machine

See also: [`PropositionalDataSet`](@ref), [`setup_dataset`](@ref), [`AbstractDataSet`](@ref)
"""
mutable struct ModalDataSet{M} <: AbstractDataSet
    mach    :: MLJ.Machine
    pidxs   :: Vector{PartitionIdxs}
    pinfo   :: PartitionInfo
    tinfo   :: TreatmentInfo
end

"""
    DataSet(mach, pidxs, pinfo; tinfo=nothing) -> AbstractDataSet

Constructor function that creates the appropriate dataset type based on treatment information.

This function serves as a smart constructor that automatically determines whether to create
a `PropositionalDataSet` or `ModalDataSet` based on the provided treatment information and
the type of treatment specified.

# Arguments
- `mach::MLJ.Machine{M}`: MLJ machine containing model and data
- `pidxs::Vector{PartitionIdxs}`: Partition indices for cross-validation folds
- `pinfo::PartitionInfo`: Information about the partitioning strategy
- `tinfo::MaybeTreatInfo=nothing`: Optional treatment information for multidimensional data

# Returns
- `PropositionalDataSet{M}`: When `tinfo` is `nothing` or treatment is not `:reducesize`
- `ModalDataSet{M}`: When `tinfo` specifies `:reducesize` treatment

# Decision Logic
1. **No treatment info** (`tinfo = nothing`): Creates `PropositionalDataSet` with no aggregation info
2. **Reduce size treatment** (`get_treatment(tinfo) == :reducesize`): Creates `ModalDataSet` preserving structure
3. **Other treatments** (e.g., `:aggregate`): Creates `PropositionalDataSet` with aggregation info converted from treatment

# Type Parameter
- `M <: MLJ.Model`: The type of the model in the MLJ machine

See also: [`PropositionalDataSet`](@ref), [`ModalDataSet`](@ref), [`AbstractDataSet`](@ref)
"""
function DataSet(
    mach    :: MLJ.Machine{M},
    pidxs   :: Vector{PartitionIdxs},
    pinfo   :: PartitionInfo;
    tinfo   :: MaybeTreatInfo=nothing
) where {M<:MLJ.Model}
    isnothing(tinfo) ?
        PropositionalDataSet{M}(mach, pidxs, pinfo, nothing) : begin
        if get_treatment(tinfo) == :reducesize
            ModalDataSet{M}(mach, pidxs, pinfo, tinfo)
        else
            ainfo = treat2aggr(tinfo)
            PropositionalDataSet{M}(mach, pidxs, pinfo, ainfo)
        end
    end
end

# ---------------------------------------------------------------------------- #
#                                    methods                                   #
# ---------------------------------------------------------------------------- #
Base.length(ds::AbstractDataSet) = length(ds.pidxs)

"""
    get_X(ds::AbstractDataSet) -> DataFrame

Extract feature DataFrame from dataset's MLJ machine.
"""
get_X(ds::AbstractDataSet)::DataFrame = ds.mach.args[1].data

"""
    get_X(ds::AbstractDataSet, part::Symbol) -> Vector{<:AbstractDataFrame}

Extract feature DataFrames for a specific partition (e.g., :train, :test or :valid) across all folds.
"""
get_X(ds::AbstractDataSet, part::Symbol)::Vector{<:AbstractDataFrame} = 
    [@views get_X(ds)[getproperty(ds.pidxs[i], part), :] for i in 1:length(ds)]

"""
    get_y(ds::AbstractDataSet) -> Vector

Extract target vector from dataset's MLJ machine.
"""
get_y(ds::AbstractDataSet)::Vector = ds.mach.args[2].data

"""
    get_y(ds::AbstractDataSet, part::Symbol) -> Vector{<:AbstractVector}

Extract target values for a specific partition (e.g., :train, :test or :valid) across all folds.
"""
get_y(ds::AbstractDataSet, part::Symbol)::AbstractVector = 
    [@views get_y(ds)[getproperty(ds.pidxs[i], part)] for i in 1:length(ds)]

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

# ---------------------------------------------------------------------------- #
#                            internal setup dataset                            #
# ---------------------------------------------------------------------------- #
function _setup_dataset(
    X             :: AbstractDataFrame,
    y             :: AbstractVector,
    w             :: MaybeVector                  = nothing;
    model         :: MLJ.Model                    = _DefaultModel(y),
    resample      :: ResamplingStrategy           = Holdout(shuffle=true),
    valid_ratio   :: Real                         = 0.0,
    rng           :: AbstractRNG                  = TaskLocalRNG(),
    win           :: WinFunction                  = AdaptiveWindow(nwindows=3, relative_overlap=0.1),
    features      :: Tuple{Vararg{Base.Callable}} = (maximum, minimum),
    modalreduce   :: Base.Callable                = mean,
    tuning        :: MaybeTuning                  = nothing
)::AbstractDataSet
    # propagate user rng to every field that needs it
    hasproperty(model, :rng)    && set_rng!(model, rng)
    hasproperty(resample, :rng) && (resample = set_rng(resample, rng))

    # ModalDecisionTrees package needs features to be passed in model params
    hasproperty(model, :features) && set_conditions!(model, features)

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

    ttpairs, pinfo = partition(y; resample, valid_ratio, rng)

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
            tuning_params(tuning)...
        )

        # set the model to use the same rng as the dataset
        set_tuning_rng!(model, rng)
    end

    mach = isnothing(w) ? MLJ.machine(model, X, y) : MLJ.machine(model, X, y, w)
    
    DataSet(mach, ttpairs, pinfo; tinfo)
end

# ---------------------------------------------------------------------------- #
#                                setup dataset                                 #
# ---------------------------------------------------------------------------- #
"""
    setup_dataset(
        X, y, w=nothing;
        model=_DefaultModel(y),
        resample=Holdout(fraction_train=0.7, shuffle=true),
        valid_ratio=0.0,
        rng=TaskLocalRNG(),
        win=AdaptiveWindow(nwindows=3, relative_overlap=0.1),
        features=(maximum, minimum),
        modalreduce=mean,
        tuning=nothing
    ) -> AbstractDataSet

Creates and configures a dataset structure for machine learning.

This is the core implementation function that handles the complete dataset setup pipeline,
including data preprocessing, model configuration, partitioning, hyperparameter tuning,
and MLJ machine creation.

# Arguments
- `X::AbstractDataFrame`: Feature matrix/DataFrame
- `y::AbstractVector`: Target variable vector
- `w::MaybeVector=nothing`: Optional sample weights

# Keyword Arguments

## Model Configuration
- `model::MLJ.Model=_DefaultModel(y)`: Sole compatible MLJ model to use, 
   auto-selected based on target type, if no `model` is subbmitted

## Available Models
- **`DecisionTreeClassifier`**, **`DecisionTreeRegressor`**
  **`RandomForestClassifier`**, **`RandomForestRegressor`**
  **`AdaBoostStumpClassifier`**
  from package [DecisionTree.jl]

## Resample



- `tuning::MaybeTuning=nothing`: Hyperparameter tuning configuration,
   requires `range` vectors.

## Data Partitioning
- `resample::ResamplingStrategy=Holdout(shuffle=true)`: Cross-validation strategy
- `valid_ratio::Real=0.0`: Validation set proportion
- `rng::AbstractRNG=TaskLocalRNG()`: Random number generator for reproducibility

## Multidimensional Data Processing
- `win::WinFunction=AdaptiveWindow(nwindows=3, relative_overlap=0.1)`: Windowing function
- `features::Tuple{Vararg{Base.Callable}}=(maximum, minimum)`: Feature extraction functions
- `modalreduce::Base.Callable=mean`: Reduction function for modal algorithms

# Returns
- `PropositionalDataSet{M}`: For standard ML algorithms with tabular data
- `ModalDataSet{M}`: For modal logic algorithms with structured data

# Processing Pipeline

## 1. Random Number Generator Propagation
```julia
# Ensures reproducible results across all components
hasproperty(model, :rng) && set_rng!(model, rng)
hasproperty(resample, :rng) && (resample = set_rng(resample, rng))
```

## 2. Model-Specific Configuration
- **Modal models**: Sets feature extraction functions via `set_conditions!`
- **Holdout resampling**: Configures training fraction via `set_fraction_train`

## 3. Data Type Detection & Processing
```julia
if X[1, 1] isa AbstractArray
    # Multidimensional data: choose treatment strategy
    treat = model isa Modal ? :reducesize : :aggregate
    X, tinfo = treatment(X; win, features, treat, modalreduce)
else
    # Tabular data: encode categorical variables
    X = code_dataset(X)
    tinfo = nothing
end
```

## 4. Data Partitioning
Creates train/test splits with cross-validation folds using the specified resampling strategy.

## 5. Hyperparameter Tuning Setup
```julia
if !isnothing(tuning)
    # Convert SX.range to MLJ.range specifications
    # Wrap model in TunedModel with tuning configuration
    # Propagate RNG to tuning components
end
```

## 6. MLJ Machine Creation
Creates MLJ machine with or without sample weights, ready for training.

## 7. Dataset Construction
Uses the smart `DataSet` constructor to create the appropriate dataset type.

# Data Type Handling

## Tabular Data (Standard Case)
- **Detection**: `X[1, 1]` is not an `AbstractArray`
- **Processing**: Categorical encoding via `code_dataset`
- **Result**: `PropositionalDataSet` with encoded features

## Multidimensional Data (Time Series/Structured)
- **Detection**: `X[1, 1]` is an `AbstractArray`
- **Modal algorithms**: Use `:reducesize` treatment → `ModalDataSet`
- **Standard algorithms**: Use `:aggregate` treatment → `PropositionalDataSet`

# Tuning Integration
Supports both simple and complex hyperparameter tuning configurations:
```julia
# Simple range tuning
tuning = Tuning(range=(:max_depth, 1:10))

# Complex multi-parameter tuning
tuning = Tuning(
    range=[(:max_depth, 1:10), (:min_samples_split, 2:20)],
    measure=accuracy,
    resampling=CV(nfolds=3)
)
```

# Examples
```julia
# Standard classification
dataset = _setup_dataset(X_tabular, y_class)

# Regression with custom model
dataset = _setup_dataset(X, y_continuous; model=RandomForestRegressor())

# Modal learning with time series
dataset = _setup_dataset(X_timeseries, y; model=ModalDecisionTree())

# With hyperparameter tuning
dataset = _setup_dataset(X, y; tuning=Tuning(range=(:max_depth, 1:10)))

# Custom cross-validation
dataset = _setup_dataset(X, y; resample=CV(nfolds=5), rng=123)
```

# Implementation Details
- **Efficient processing**: Minimal data copying through view-based operations
- **Type stability**: Returns concrete dataset types based on input characteristics
- **Error handling**: Validates inputs and provides informative error messages
- **Memory efficiency**: Uses MLJ's lazy evaluation and caching mechanisms

# See Also
- [`setup_dataset`](@ref): Public interface for this function
- [`DataSet`](@ref): Smart constructor for dataset types
- [`treatment`](@ref): Multidimensional data processing
- [`partition`](@ref): Data partitioning utilities
- [`code_dataset`](@ref): Categorical encoding
- [`PropositionalDataSet`](@ref), [`ModalDataSet`](@ref): Dataset types

# Internal Use
This function is the core implementation behind the public `setup_dataset` interface.
External users should typically use `setup_dataset` instead of calling this directly.
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
