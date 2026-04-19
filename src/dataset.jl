# ---------------------------------------------------------------------------- #
#                                  defaults                                    #
# ---------------------------------------------------------------------------- #
# return a default model appropriate for the target variable type
# this function is used when no explicit model is provided to `setup_dataset`,
# automatically selecting between classification and regression
# no supervised (y==nothing) is treated as regression as it is.
function _default_model(y::AbstractVector)
    return y isa CategoricalArray && !isempty(y) ?
        DecisionTreeClassifier() :
        DecisionTreeRegressor()
end

# ---------------------------------------------------------------------------- #
#                                   set rng                                    #
# ---------------------------------------------------------------------------- #
# Set the random number generator for a model that supports it
function set_rng!(m::MLJ.Model, rng::Random.AbstractRNG)
    m.rng = rng
    return m
end

# set the random number generator for a resampling strategy
function set_rng(r::MLJ.ResamplingStrategy, rng::Random.AbstractRNG)
    typeof(r)(merge(MLJ.params(r), (rng=rng,))...)
end

# set random number generators for tuning-related components of a model
function set_tuning_rng!(m::MLJ.Model, rng::Random.AbstractRNG)
    hasproperty(m.tuning, :rng) && (m.tuning.rng = rng)
    hasproperty(m.resampling, :rng) && (m.resampling = set_rng(m.resampling, rng))
    return m
end

# ---------------------------------------------------------------------------- #
#                                DataSet struct                                #
# ---------------------------------------------------------------------------- #
struct DataSet{M,T}
    mach::MLJ.Machine
    pidxs::Vector{PartitionIdxs{T}}
    pinfo::PartitionInfo

    DataSet(
        mach::MLJ.Machine{M},
        pidxs::Vector{PartitionIdxs{T}},
        pinfo::PartitionInfo
    ) where {M<:MLJ.Model,T} = new{M,T}(mach, pidxs, pinfo)
end

# ---------------------------------------------------------------------------- #
#                                    methods                                   #
# ---------------------------------------------------------------------------- #
Base.length(ds::DataSet) = length(ds.pidxs)

get_X(ds::DataSet) = ds.mach.args[1].data
get_X(ds::DataSet, part::Symbol) = 
    [@views get_X(ds)[getproperty(ds.pidxs[i], part), :] for i in 1:length(ds)]
get_y(ds::DataSet) = ds.mach.args[2].data
get_y(ds::DataSet, part::Symbol) = 
    [@views get_y(ds)[getproperty(ds.pidxs[i], part)] for i in 1:length(ds)]

get_mach(ds::DataSet) = ds.mach
get_mach_model(ds::DataSet) = ds.mach.model

get_logiset(ds::DataSet) = ds.mach.data[1].modalities[1]
get_rng(ds::DataSet) = get_rng(ds.pinfo)

# ---------------------------------------------------------------------------- #
#                             MLJ models's tuning                              #
# ---------------------------------------------------------------------------- #
function set_tuning(
    model::MLJ.Model,
    tuning::Tuning,
    rng::Random.AbstractRNG
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
#                                setup dataset                                 #
# ---------------------------------------------------------------------------- #
function setup_dataset(
    X::AbstractDataFrame,
    y::AbstractVector,
    args...;
    kwargs...
)
    throw(ArgumentError(
        "Target variable y must have elements of type Label, " *
        "got eltype: $(eltype(y))"))
end

"""
    setup_dataset(
        dt::DataTreatment;
        w=nothing,
        model=_default_model(get_target(dt)),
        resampling=Holdout(fraction_train=0.7, shuffle=true),
        valid_ratio=0.0,
        seed=nothing,
        tuning=nothing,
    ) -> DataSet

    setup_dataset(
        X::Matrix,
        vnames::Vector{String},
        y=nothing,
        treatments...;
        treatment_ds=true,
        leftover_ds=true,
        float_type=Float64,
        kwargs...
    ) -> DataSet

    setup_dataset(df::AbstractDataFrame, y=nothing, args...; kwargs...) -> DataSet
    setup_dataset(df::AbstractDataFrame, y::Symbol, args...; kwargs...) -> DataSet

Creates and configures a dataset structure for machine learning.

This is the core implementation function that handles the complete dataset setup pipeline,
including data preprocessing, model configuration, partitioning, hyperparameter tuning,
and MLJ machine creation.

# Arguments
- `dt::DataTreatment`: A `DataTreatment` object encapsulating features, target, and
  preprocessing information. Use `DataTreatments.load_dataset` to construct one.
- `X::Matrix`: Raw feature matrix.
- `vnames::Vector{String}`: Column names for the feature matrix.
- `y::Union{Nothing,AbstractVector{<:Label}}=nothing`: Target variable vector.
  If `nothing`, an unsupervised (or regression-only) setup is assumed.
- `df::AbstractDataFrame`: Feature DataFrame, optionally containing the target column.
- `treatments::Vararg{Base.Callable}`: Data treatment functions applied during
  preprocessing (defaults to `DataTreatments.DefaultTreatmentGroup`).

# Keyword Arguments

## Model Configuration
- `model::MLJ.Model=_default_model(y)`: Sole-compatible MLJ model to use,
  auto-selected based on target type if not provided. Classification targets
  (CategoricalArray) default to `DecisionTreeClassifier`, others to `DecisionTreeRegressor`.

## Data Resampling
- `resampling::ResamplingStrategy=Holdout(fraction_train=0.7, shuffle=true)`:
  Cross-validation or holdout strategy. Strategies are taken from
  [MLJ](https://juliaai.github.io/MLJBase.jl/stable/resampling/).
- `valid_ratio::Real=0.0`: Fraction of training data to reserve as a validation set.
  Primarily used with XGBoost [early stopping](https://xgboost.readthedocs.io/en/stable/prediction.html).
- `seed::Union{Nothing,Int}=nothing`: Integer seed for reproducibility. Internally
  initializes a `Xoshiro` RNG and propagates it to the model, resampling strategy,
  and tuning components.


## Hyperparameter Tuning
- `tuning::Union{Nothing,Tuning}=nothing`: Tuning configuration. Requires a `range`
  specification, e.g.:
  ```julia
  range = SoleXplorer.range(:min_purity_increase; lower=0.1, upper=1.0, scale=:log)
  ```
  Tuning strategies are adapted from [MLJ](https://juliaai.github.io/MLJ.jl/stable/)
  and [MLJParticleSwarmOptimization.jl](https://github.com/JuliaAI/MLJParticleSwarmOptimization.jl).

## Data Loading (Matrix/DataFrame methods only)
- `treatment_ds::Bool=true`: Whether to include the treated dataset partition.
- `leftover_ds::Bool=true`: Whether to include the leftover (untreated) partition.
- `float_type::Type=Float64`: Numeric type used for feature conversion.

## Weights
- `w::Union{Nothing,Vector}=nothing`: Optional per-sample weights vector.

# Returns
- `DataSet{M,T}`: A configured dataset struct wrapping an MLJ machine, partition
  indices, and partition metadata.

# Notes
- When calling `setup_dataset(df, y::Symbol, ...)`, the target column `y` is
  automatically removed from the feature set.
- Modal models require a `DataTreatment` built from multidimensional (time-series)
  data; non-modal models require tabular data. Mixing types raises an error.
- For Modal models with tuning, if no `measure` is provided, `LogLoss()` is used
  as a default.

# Examples
```julia
using SoleXplorer, MLJ, DataFrames
const SX = SoleXplorer

Xc, yc = @load_iris
Xc = DataFrame(Xc)

# basic classification setup
ds = setup_dataset(Xc, yc)

# specify model
ds = setup_dataset(Xc, yc; model=AdaBoostStumpClassifier())

# cross-validation with seed
ds = setup_dataset(Xc, yc; resampling=CV(nfolds=10, shuffle=true), seed=1)

# hyperparameter tuning
range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
ds = setup_dataset(
    Xc, yc;
    model=ModalDecisionTree(),
    resampling=CV(nfolds=5, shuffle=true),
    seed=1,
    tuning=GridTuning(
        resolution=10,
        resampling=CV(nfolds=3),
        range=range,
        measure=accuracy,
        repeats=2
    )
)

# target as column symbol
ds = setup_dataset(Xc, :target)
```

# See also: [`DataSet`](@ref), [`solexplorer`](@ref)
"""
function setup_dataset(
    dt::DT.DataTreatment;
    w::Union{Nothing,Vector}=nothing,
    model::MLJ.Model=_default_model(DT.get_target(dt)),
    resampling::ResamplingStrategy=Holdout(fraction_train=0.7, shuffle=true),
    valid_ratio::Real=0.0,
    rng::Union{AbstractRNG,Int}=Xoshiro(42),
    tuning::Union{Nothing,Tuning}=nothing
)
    rng isa Int && (rng = Xoshiro(rng))
    # get the dataset if type is appropriate for the chosen model
    X, vnames = if has_tabular(dt) && !(model isa Modal)
        DT.get_tabular(dt)
    elseif has_multidim(dt) && (model isa Modal)
        DT.get_multidim(dt)
    else
        error("Incompatible dataset and model types: " *
        "use a modal model for multidimensional data, " *
        "and a non-modal model for tabular data.")
    end

    y = DT.get_target(dt)

    # setup rng
    hasproperty(model, :rng) && set_rng!(model, rng)
    hasproperty(resampling, :rng) && (resampling = set_rng(resampling, rng))

    # MLJ.TunedModels can't automatically assigns measure to Modal models
    if model isa Modal && !isnothing(tuning)
        isnothing(get_measure(tuning)) && (tuning.measure = LogLoss())
    end

    ttpairs, pinfo = partition(DT.nrows(dt), y; resampling, valid_ratio, rng)

    isnothing(tuning) || (model = set_tuning(model, tuning, rng))

    Xdf = DataFrame(X, vnames)
    to_mach = isempty(y) ? (Xdf) : (Xdf, y)

    mach = isnothing(w) ?
        MLJ.machine(model, to_mach...) : MLJ.machine(model, to_mach..., w)

    DataSet(mach, ttpairs, pinfo)
end

function setup_dataset(
    X::Matrix,
    vnames::Vector{String}=["V$i" for i in 1:size(X, 2)],
    y::Union{Nothing,AbstractVector{<:Label}}=nothing,
    treatments::Vararg{Base.Callable}=DT.DefaultTreatmentGroup;
    treatment_ds::Bool=true,
    leftover_ds::Bool=false,
    float_type::Type=Float64,
    kwargs...
)
    dt = DT.load_dataset(
        X,
        vnames,
        y,
        treatments...;
        treatment_ds,
        leftover_ds,
        float_type
    )

    setup_dataset(dt; kwargs...)
end

setup_dataset(
    df::AbstractDataFrame,
    y::Union{Nothing,AbstractVector{<:Label}}=nothing,
    args...;
    kwargs...
) = setup_dataset(Matrix(df), names(df), y, args...; kwargs...)

setup_dataset(df::AbstractDataFrame, args...; kwargs...) =
    setup_dataset(Matrix(df), names(df), nothing, args...; kwargs...)

"""
    setup_dataset(X::AbstractDataFrame, y::Symbol; kwargs...)::AbstractDataSet

Convenience method when target variable is a column in the feature DataFrame.
"""
setup_dataset(
    X::AbstractDataFrame,
    y::Symbol,
    args...;
    kwargs...
) = setup_dataset(X[!, Not(y)], X[!, y], args...; kwargs...)
