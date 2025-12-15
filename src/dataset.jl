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
#                                    types                                     #
# ---------------------------------------------------------------------------- #
const Balancing = NamedTuple{(:oversampler, :undersampler), <:Tuple{<:MLJ.Model, <:MLJ.Model}}

const MaybeInt             = Maybe{Int64}
const MaybeAggregationInfo = Maybe{AggregationInfo}
const MaybeBalancing       = Maybe{Balancing}
const MaybeTuning          = Maybe{Tuning}
const MaybeTreatInfo       = Maybe{TreatmentInfo}

# ---------------------------------------------------------------------------- #
#                                  defaults                                    #
# ---------------------------------------------------------------------------- #
# return a default model appropriate for the target variable type
# this function is used when no explicit model is provided to `setup_dataset`,
# automatically selecting between classification and regression
function _DefaultModel(y::AbstractVector)::MLJ.Model
    return eltype(y) <: RLabel ?
        DecisionTreeRegressor() :
        DecisionTreeClassifier()
end

# ---------------------------------------------------------------------------- #
#                                   set rng                                    #
# ---------------------------------------------------------------------------- #
# Set the random number generator for a model that supports it
function set_rng!(m::MLJ.Model, rng::Random.AbstractRNG)::MLJ.Model
    m.rng = rng
    return m
end

# set the random number generator for a resampling strategy
function set_rng(r::MLJ.ResamplingStrategy, rng::Random.AbstractRNG)::ResamplingStrategy
    typeof(r)(merge(MLJ.params(r), (rng=rng,))...)
end

# set random number generators for tuning-related components of a model
function set_tuning_rng!(m::MLJ.Model, rng::Random.AbstractRNG)::MLJ.Model
    hasproperty(m.tuning, :rng) && (m.tuning.rng = rng)
    hasproperty(m.resampling, :rng) && (m.resampling = set_rng(m.resampling, rng))
    return m
end

# set the seed for balancing-related components of a model
# originally broken in case you pass a RNG method (lenth mismatch during MLJ.fit!)
function set_balancing_seed(m::MLJ.Model, seed::Int64)::MLJ.Model
    hasproperty(m, :rng) && (m = typeof(m).name.wrapper(merge(MLJ.params(m), (rng=seed,))...))
    return m
end

# set logical conditions (features) for modal models
function set_conditions!(m::MLJ.Model, conditions::Tuple{Vararg{Base.Callable}})::MLJ.Model
    m.conditions = Function[conditions...]
    return m
end

# ---------------------------------------------------------------------------- #
#                   dataset and targets check and conversion                   #
# ---------------------------------------------------------------------------- #
# ensures that the target variable `y` is properly formatted for use with MLJ
# it handles automatic conversion to categorical format when needed for classification tasks
function check_y(y::AbstractVector, model::MLJ.Model)::AbstractVector
    return (eltype(y) === Any) || ((eltype(y) <: RLabel) && !(model isa Regression)) ?
        MLJ.categorical(string.(y)) :
        y
end

# convert all numeric columns in a DataFrame to Float64 type
function to_float_dataset(X::AbstractDataFrame)::AbstractDataFrame
    for (name, col) in pairs(eachcol(X))
        if eltype(col) <: Number && eltype(col) != AbstractFloat
            X[!, name] = Float64.(col)
        end
    end
    
    return X
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
            # handle mixed types by converting to string first
            eltype(col) == AbstractString || (col = string.(coalesce.(col, "missing")))
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
    mach  :: MLJ.Machine
    pidxs :: Vector{PartitionIdxs}
    pinfo :: PartitionInfo
    ainfo :: MaybeAggregationInfo
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
    mach  :: MLJ.Machine
    pidxs :: Vector{PartitionIdxs}
    pinfo :: PartitionInfo
    tinfo :: TreatmentInfo
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
    mach  :: MLJ.Machine{M},
    pidxs :: Vector{PartitionIdxs},
    pinfo :: PartitionInfo;
    tinfo :: MaybeTreatInfo=nothing
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
    get_logiset(ds::ModalDataSet)::SupportedLogiset

Extract the logiset (if present) from the dataset's MLJ machine.
"""
get_logiset(ds::ModalDataSet)::SupportedLogiset = ds.mach.data[1].modalities[1]

# ---------------------------------------------------------------------------- #
#                           MLJ models's extra setup                           #
# ---------------------------------------------------------------------------- #
function set_balancing(
    model     :: MLJ.Model,
    balancing :: Balancing,
    seed      :: Int64=17
)::MLJ.Model
    # regression models don't support balancing
    model isa Regression &&
        throw(ArgumentError("Balancing is not supported for regression models."))

    # set the model to use the same seed
    balancing = map(b -> set_balancing_seed(b, seed), balancing)

    model = MLJ.BalancedModel(model; balancing...)
end

function set_tuning(
    model  :: MLJ.Model,
    tuning :: Tuning,
    rng    :: Random.AbstractRNG
)::MLJ.Model
    t_range = get_range(tuning)
    if !(t_range isa MLJ.NominalRange)
        # convert SX.range to MLJ.range now that model is available
        range = t_range isa Tuple{Vararg{Tuple}} ? t_range : (t_range,)
        range = collect(MLJ.range(model, r[1]; r[2:end]...) for r in range)
        tuning.range = range
    end

    model = MLJ.TunedModel(model; tuning_params(tuning)...)

    # set the model to use the same rng as the dataset
    set_tuning_rng!(model, rng)
end

# ---------------------------------------------------------------------------- #
#                            internal setup dataset                            #
# ---------------------------------------------------------------------------- #
function _setup_dataset(
    X             :: AbstractDataFrame,
    y             :: AbstractVector,
    w             :: MaybeVector                  = nothing;
    model         :: MLJ.Model                    = _DefaultModel(y),
    resampling    :: ResamplingStrategy           = Holdout(fraction_train=0.7, shuffle=true),
    valid_ratio   :: Real                         = 0.0,
    seed          :: MaybeInt                     = nothing,
    balancing     :: MaybeBalancing               = nothing,
    tuning        :: MaybeTuning                  = nothing,
    win           :: WinFunction                  = AdaptiveWindow(nwindows=3, relative_overlap=0.1),
    features      :: Tuple{Vararg{Base.Callable}} = (maximum, minimum),
    modalreduce   :: Base.Callable                = mean
)::AbstractDataSet
    # check y special cases
    y = check_y(y, model)
    eltype(y) <: Label || throw(ArgumentError("Target variable y must have elements of type Label, " *
        "got eltype: $(eltype(y))"))

    # setup rng
    if !isnothing(seed)
        rng = Xoshiro(seed)
        # propagate user rng to every field that needs it
        hasproperty(model, :rng)      && set_rng!(model, rng)
        hasproperty(resampling, :rng) && (resampling = set_rng(resampling, rng))
    else
        rng = TaskLocalRNG()
    end

    # Modal models need features to be passed in model params
    hasproperty(model, :features) && set_conditions!(model, features)
    # MLJ.TunedModels can't automatically assigns measure to Modal models
    if model isa Modal && !isnothing(tuning)
        isnothing(get_measure(tuning)) && (tuning.measure = LogLoss())
    end

    # handle multidimensional datasets:
    # propositional models requiring feature aggregation
    # modal models requiring reducing data size
    if is_multidim_dataframe(X)
        treat = model isa Modal ? :reducesize : :aggregate
        X, tinfo = treatment(X, treat; features, win, modalreduce)
    else
        X = code_dataset(X)
        # some algos, like xgboost, doesnt accept dataset with numeric values, only float
        X = to_float_dataset(X)
        tinfo = nothing
    end

    ttpairs, pinfo = partition(y; resampling, valid_ratio, rng)

    isnothing(seed)      && (seed = 1)
    isnothing(balancing) || (model = set_balancing(model, balancing, seed))
    isnothing(tuning)    || (model = set_tuning(model, tuning, rng))

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
        resampling=Holdout(fraction_train=0.7, shuffle=true),
        valid_ratio=0.0,
        seed=nothing,
        balancing=nothing,
        tuning=nothing,
        win=AdaptiveWindow(nwindows=3, relative_overlap=0.1),
        features=(maximum, minimum),
        modalreduce=mean,
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

## Data resampling
- `resampling::ResamplingStrategy=Holdout(shuffle=true)`: Cross-validation strategy
- `valid_ratio::Real=0.0`: Validation set proportion
- `rng::Random.AbstractRNG=TaskLocalRNG()`: Random number generator for reproducibility

Resampling strategies are taken from the package [MLJ](https://juliaai.github.io/MLJ.jl/stable/).
See official documentation [here](https://juliaai.github.io/MLJBase.jl/stable/resampling/).

`valid_ratio` is used with XGBoost early stop [technique](https://xgboost.readthedocs.io/en/stable/prediction.html).
`rng` can be setted externally (via seed, using internal Xoshiro algo) for convenience.

## Balancing
Balancing strategies are taken from the package [Imbalance](https://github.com/JuliaAI/Imbalance.jl).
See official documentation [here](https://juliaai.github.io/Imbalance.jl/dev/).

## Tuning
`tuning::MaybeTuning=nothing`: Hyperparameter tuning configuration,
requires `range` vectors, i.e:
```
range = SoleXplorer.range(:min_purity_increase; lower=0.1, upper=1.0, scale=:log)
```

Tuning strategies are adapted from the package [MLJ](https://juliaai.github.io/MLJ.jl/stable/)
and package [MLJParticleSwarmOptimization](https://github.com/JuliaAI/MLJParticleSwarmOptimization.jl).


## Multidimensional Data Processing
These parameters are needed only if a time series dataset is used.
With these parameters we can tweak size reduction, in case of **modal** analysis
or aggregation strategy, in case of further **propositional** analysis.
Parameters are the same, SoleXplorer will take care of automatically set the case,
depending on the model choose.

- `win::WinFunction=AdaptiveWindow(nwindows=3, relative_overlap=0.1)`: Windowing function
Available windows strategies: [MovingWindow](@ref), [WholeWindow](@ref), [SplitWindow](@ref), [AdaptiveWindow](@ref).

- `features::Tuple{Vararg{Base.Callable}}=(maximum, minimum)`: Feature extraction functions
Note that beyond standard reduction functions (e.g., maximum, minimum, mean, mode), [Catch22](https://time-series-features.gitbook.io/catch22) time-series features are also available.
- `modalreduce::Base.Callable=mean`: Reduction function for modal algorithms

# Returns
- `PropositionalDataSet{M}`: For standard ML algorithms with tabular data
- `ModalDataSet{M}`: For modal logic algorithms with structured data

# Examples:
```julia
using SoleXplorer
using MLJ
using DataFrames, Random
const SX = SoleXplorer

Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

natopsloader = SX.NatopsLoader()
Xts, yts = SX.load(natopsloader)

# basic setup
dsc = setup_dataset(Xc, yc)

# model type specification
dsc = setup_dataset(
    Xc, yc;
    model=AdaBoostStumpClassifier()
)

# resampling
dsc = setup_dataset(
    Xc, yc;
    resampling=CV(nfolds=10),
)

dsc = setup_dataset(
    Xc, yc;
    resampling=CV(nfolds=10, shuffle=true),
    seed=1
)

# tuning
range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
dsc = setup_dataset(
    Xc, yc;
    model=ModalDecisionTree(),
    resampling=CV(nfolds=5, shuffle=true),
    seed=1,
    tuning=GridTuning(resolution=10, resampling=CV(nfolds=3), range=range, measure=accuracy, repeats=2)
)

# time-series
dts = setup_dataset(
    Xts, yts;
    model=ModalRandomForest(),
    resampling=Holdout(fraction_train=0.7, shuffle=true),
    seed=1,
    win=AdaptiveWindow(nwindows=3, relative_overlap=0.3),
    features=(minimum, maximum),
    modalreduce=mode
)
```

# See also: [`DataSet`](@ref), [`PropositionalDataSet`](@ref), [`ModalDataSet`](@ref), [`symbolic_analysis`](@ref)
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
