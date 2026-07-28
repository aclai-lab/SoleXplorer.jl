# ---------------------------------------------------------------------------- #
#                                  defaults                                    #
# ---------------------------------------------------------------------------- #
is_multidim(::Matrix{T}) where T = T <: AbstractArray

# return a default model appropriate for the target variable type and data
# dimensionality.
# this function is used when no explicit model is provided to `setup_dataset`,
# automatically selecting between classification and regression, and between
# modal and non-modal models based on the dataset dimensionality.
# unsupervised (y == nothing) is treated as regression.
function default_model(X::Matrix{T}, y::AbstractVector) where T
    is_multidim(X) && return ModalDecisionTree()
    return y isa CategoricalArray && !isempty(y) ?
        DecisionTreeClassifier() : DecisionTreeRegressor()
end

# if no data treatment is specified, check the model being used and,
# if it is modal, do not aggregate; instead, reduce the dimensionality of the
# multidimensional data.
function default_treatment(model::MLJ.Model)
    return model isa Modal ?
        TreatmentGroup(aggrfunc=reducesize(win=(splitwindow(nwindows=3)))) :
        DT.DefaultTreatmentGroup
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
    hasproperty(m.tuning, :rng) &&
        (m.tuning.rng = rng)
    hasproperty(m.resampling, :rng) &&
        (m.resampling = set_rng(m.resampling, rng))
    return m
end

# ---------------------------------------------------------------------------- #
#                                DataSet struct                                #
# ---------------------------------------------------------------------------- #
"""
    DataSet{M<:MLJ.Model,T} <: Any

A configured dataset structure wrapping an MLJ machine, partition indices,
and partition metadata.

# Type Parameters
- `M<:MLJ.Model`: The MLJ model type (e.g. `DecisionTreeClassifier`).
- `T`: The element type of the partition index vectors.

# Fields
- `mach::MLJ.Machine`: The MLJ machine holding the model, features `X`,
  and target `y`.
- `pidxs::Vector{PartitionIdxs{T}}`: Per-fold partition index sets,
  each containing train/test (and optionally validation) indices.
- `pinfo::PartitionInfo`: Metadata about the partitioning strategy,
  including the resampling type and rng used.

# See also: [`setup_dataset`](@ref), [`solexplorer`](@ref)
"""
mutable struct DataSet{M,T}
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
"""
    Base.length(ds::DataSet) -> Int

Return the number of folds (partitions) in the dataset.
"""
Base.length(ds::DataSet) = length(ds.pidxs)

"""
    get_X(ds::DataSet) -> AbstractMatrix
    get_X(ds::DataSet, part::Symbol) -> Vector{AbstractMatrix}

Return the feature matrix stored in the MLJ machine.

The two-argument form returns a vector of views, one per fold, sliced
according to the partition indices named `part` (e.g. `:train`, `:test`,
`:val`).
"""
get_X(ds::DataSet) = ds.mach.args[1].data
get_X(ds::DataSet, part::Symbol) =
    [@views get_X(ds)[getproperty(ds.pidxs[i], part), :] for i in 1:length(ds)]

"""
    get_y(ds::DataSet) -> AbstractVector
    get_y(ds::DataSet, part::Symbol) -> Vector{AbstractVector}

Return the target vector stored in the MLJ machine.

The two-argument form returns a vector of views, one per fold, sliced
according to the partition indices named `part` (e.g. `:train`, `:test`,
`:val`).
"""
get_y(ds::DataSet) = ds.mach.args[2].data
get_y(ds::DataSet, part::Symbol) =
    [@views get_y(ds)[getproperty(ds.pidxs[i], part)] for i in 1:length(ds)]

"""
    get_mach(ds::DataSet) -> MLJ.Machine

Return the MLJ machine wrapped by the dataset.
"""
get_mach(ds::DataSet) = ds.mach

"""
    get_mach_model(ds::DataSet) -> MLJ.Model

Return the model stored inside the MLJ machine.
"""
get_mach_model(ds::DataSet) = ds.mach.model

"""
    get_logiset(ds::DataSet) -> AbstractLogiset

Return the first modality of the logiset produced after fitting the
machine. Only valid for modal datasets.
"""
get_logiset(ds::DataSet) = ds.mach.data[1].modalities[1]

"""
    get_rng(ds::DataSet) -> AbstractRNG

Return the random number generator used when partitioning the dataset.
"""
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
"""
    setup_dataset(
        X::Matrix,
        y=nothing;
        vnames=["V1", "V2", ...],
        w=nothing,
        model=default_model(y, is_multidim(X)),
        resampling=Holdout(fraction_train=0.7, shuffle=true),
        valid_ratio=0.0,
        tuning=nothing,
        float_type=Float32,
        rng=Xoshiro(42),
        kwargs...
    ) -> DataSet

    setup_dataset(
        df::AbstractDataFrame,
        y=nothing;
        kwargs...
    ) -> DataSet

Create and configure a `DataSet` for machine learning. The function loads or
uses the supplied data treatment, selects the data representation compatible
with the model, creates train/test (and optionally validation) partitions,
and builds an MLJ machine.

# Arguments
- `X::Matrix`: Raw feature matrix. Multidimensional samples are represented
  by matrix elements that are themselves vectors.
- `df::AbstractDataFrame`: Tabular feature data. It is converted to a matrix
  before loading.
- `y::Union{Nothing,AbstractVector{<:Label}}=nothing`: Target values. When
  omitted, the MLJ machine is created without a target.
- `vnames::Vector{String}`: Feature names for `X`. By default, names are
  generated as `"V1"`, `"V2"`, and so on.
- `w::Union{Nothing,Vector}=nothing`: Optional per-observation weights.

# Keyword Arguments

## Model and preprocessing
- `model::MLJ.Model`: Model used to build the MLJ machine. If omitted, a
  `ModalDecisionTree` is selected for multidimensional data; otherwise,
  categorical targets select a `DecisionTreeClassifier` and other targets
  select a `DecisionTreeRegressor`.
- `float_type::Type=Float32`: Numeric type used while loading matrix or
  DataFrame data.

## Resampling
- `resampling::ResamplingStrategy=Holdout(fraction_train=0.7, shuffle=true)`:
  MLJ holdout or cross-validation strategy used to partition observations.
- `valid_ratio::Real=0.0`: Fraction of each training partition reserved for
  validation.
- `rng::Union{AbstractRNG,Int}=Xoshiro(42)`: Random-number generator, or an
  integer seed. The generator is propagated to the model, resampling strategy,
  and tuning components when supported.

## Hyperparameter tuning
- `tuning::Union{Nothing,Tuning}=nothing`: Optional tuning configuration. Its
  range is converted to an MLJ range after the model has been selected. For
  modal models, `LogLoss()` is used when no tuning measure is specified.

# Returns
A `DataSet` containing the MLJ machine, partition indices, and partition
metadata.

# Errors
Throws an error when a modal model is used with tabular data, or when a
non-modal model is used with multidimensional data.

# Examples
```julia
using SoleXplorer, MLJ, DataFrames

X, y = @load_iris
df = DataFrame(X)

# Classification with default model and preprocessing:
ds = setup_dataset(df, y)

# Cross-validation with a reproducible seed:
ds = setup_dataset(
    df,
    y;
    resampling=CV(nfolds=10, shuffle=true),
    rng=1,
)

# Explicit model and tuning:
range = SoleXplorer.range(:gamma; lower=0.001, upper=1.0, scale=:log)
ds = setup_dataset(
    df,
    y;
    model=XGBoostClassifier(),
    tuning=GridTuning(
        resolution=10,
        resampling=CV(nfolds=3),
        range=range,
        measure=accuracy,
        repeats=2,
    ),
)
```

# See also
[`DataSet`](@ref), [`solexplorer`](@ref)
"""
function setup_dataset(
    dt::DT.DataTreatment;
    w::Union{Nothing,Vector}=nothing,
    model::MLJ.Model=default_model(DT.get_target(dt), has_multidim(dt)),
    resampling::ResamplingStrategy=Holdout(fraction_train=0.7, shuffle=true),
    valid_ratio::Real=0.0,
    tuning::Union{Nothing,Tuning}=nothing,
    rng::Union{AbstractRNG,Int}=Xoshiro(42)
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
    X::Matrix{T},
    y::Union{Nothing,AbstractVector{<:Label}}=nothing;
    vnames::Vector{String}=["V$i" for i in 1:size(X, 2)],
    w::Union{Nothing,Vector}=nothing,
    model::MLJ.Model=default_model(X, y),
    resampling::ResamplingStrategy=Holdout(fraction_train=0.7, shuffle=true),
    valid_ratio::Real=0.0,
    tuning::Union{Nothing,Tuning}=nothing,
    float_type::Type=Float32,
    rng::Union{AbstractRNG,Int}=Xoshiro(42),
    kwargs...
) where T
    treatments = isempty(kwargs) ?
        default_treatment(model) :
        TreatmentGroup(; kwargs...)

    dt = DT.load_dataset(
        X,
        vnames,
        y,
        treatments;
        float_type
    )

    setup_dataset(
        dt;
        w,
        model,
        resampling,
        valid_ratio,
        tuning,
        rng,
    )
end

function setup_dataset(
    df::AbstractDataFrame,
    y::Union{Nothing,AbstractVector{<:Label}}=nothing;
    kwargs...
) 
setup_dataset(Matrix(df), y; vnames=names(df), kwargs...)
end

setup_dataset(df::AbstractDataFrame, args...; kwargs...) =
    setup_dataset(Matrix(df); vnames=names(df), kwargs...)
