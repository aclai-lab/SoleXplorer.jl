# ---------------------------------------------------------------------------- #
#                                  defaults                                    #
# ---------------------------------------------------------------------------- #
is_multidim(::Matrix{T}) where T = T <: AbstractArray

# if no data treatment is specified, check the model being used and,
# if it is modal, do not aggregate; instead, reduce the dimensionality of the
# multidimensional data.
function default_treatment(model::MLJ.Model; kwargs...)
    return model isa Modal ?
        TreatmentGroup(
            aggrfunc=reducesize(win=(splitwindow(nwindows=3))); kwargs...) :
        TreatmentGroup(
            aggrfunc=aggregate(win=(wholewindow())); kwargs...)
end

# ---------------------------------------------------------------------------- #
#                                   set rng                                    #
# ---------------------------------------------------------------------------- #
# set `model.rng` to `rng` and return the modified model.
# called only for MLJ models exposing an `rng` property.
function set_rng!(m::MLJ.Model, rng::Random.AbstractRNG)
    m.rng = rng
    return m
end

# return a copy of `resampling` configured with `rng`.
# called only for resampling strategies exposing an `rng` parameter.
function set_rng(r::MLJ.ResamplingStrategy, rng::Random.AbstractRNG)
    typeof(r)(merge(MLJ.params(r), (rng=rng,))...)
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
    Base.length(ds::DataSet)

Return the number of folds (partitions) in the dataset.
"""
Base.length(ds::DataSet) = length(ds.pidxs)

"""
    get_X(ds::DataSet)
    get_X(ds::DataSet, part::Symbol)

Return the feature table stored in the MLJ machine. Datasets created by
[`setup_dataset`](@ref) store features as a `DataFrame`.

The two-argument form returns one view per fold, selecting the rows named by
`part` in each partition (for example, `:train`, `:test`, or `:val`).
"""
get_X(ds::DataSet) = ds.mach.args[1].data
get_X(ds::DataSet, part::Symbol) =
    [@views get_X(ds)[getproperty(ds.pidxs[i], part), :] for i in 1:length(ds)]

"""
    get_y(ds::DataSet)
    get_y(ds::DataSet, part::Symbol)

Return the target vector stored in the MLJ machine.

The two-argument form returns one view per fold, selecting observations named
by `part`. These accessors require a supervised dataset; they are unavailable
when `setup_dataset` was called without a target.
"""
get_y(ds::DataSet) = ds.mach.args[2].data
get_y(ds::DataSet, part::Symbol) =
    [@views get_y(ds)[getproperty(ds.pidxs[i], part)] for i in 1:length(ds)]

"""
    get_mach(ds::DataSet)

Return the MLJ machine wrapped by the dataset.
"""
get_mach(ds::DataSet) = ds.mach

"""
    get_mach_model(ds::DataSet)

Return the model stored inside the MLJ machine.
"""
get_mach_model(ds::DataSet) = ds.mach.model

"""
    get_rng(ds::DataSet) -> AbstractRNG

Return the random number generator used when partitioning the dataset.
"""
get_rng(ds::DataSet) = get_rng(ds.pinfo)

# ---------------------------------------------------------------------------- #
#                                setup dataset                                 #
# ---------------------------------------------------------------------------- #
function _setup_dataset(
    dt::DT.DataTreatment;
    model::MLJ.Model,
    w::Union{Nothing,Vector}=nothing,
    resampling::ResamplingStrategy=Holdout(fraction_train=0.7, shuffle=true),
    valid_ratio::Real=0.0,
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

    ttpairs, pinfo = partition(DT.nrows(dt), y; resampling, valid_ratio, rng)

    Xdf = DataFrame(X, vnames)
    to_mach = isempty(y) ? (Xdf) : (Xdf, y)

    mach = isnothing(w) ?
        MLJ.machine(model, to_mach...) : MLJ.machine(model, to_mach..., w)

    DataSet(mach, ttpairs, pinfo)
end

"""
    setup_dataset(X::Matrix, y=nothing; model, kwargs...) -> DataSet
    setup_dataset(df::AbstractDataFrame, y=nothing; model, kwargs...) -> DataSet

Create an untrained [`DataSet`](@ref) containing an MLJ machine, partition
indices, and partition metadata.

`X` may contain scalar values for tabular learning or array-valued entries for
multidimensional/modal learning. Data-frame inputs are converted to a matrix
before being loaded. The selected treatment determines whether tabular or
multidimensional data are supplied to the machine.

# Arguments
- `X::Matrix`: Feature matrix.
- `df::AbstractDataFrame`: Tabular feature data.
- `y::Union{Nothing,AbstractVector{<:Label}}=nothing`: Optional target vector.
- `model::MLJ.Model`: **Required** MLJ model. Modal models require
  multidimensional data; non-modal models require tabular data.
- `vnames::Vector{String}`: Names for matrix columns. Defaults to `"V1"`,
  `"V2"`, and so on.
- `w::Union{Nothing,Vector}=nothing`: Optional observation weights.

# Keyword arguments
- `resampling::ResamplingStrategy=Holdout(fraction_train=0.7, shuffle=true)`:
  Holdout or cross-validation strategy used to create partitions.
- `valid_ratio::Real=0.0`: Fraction of each training partition reserved for
  validation.
- `balance=nothing`: Optional balancing strategy forwarded to
  `DataTreatments.load_dataset`.
- `float_type::Type=Float32`: Numeric type used while loading data.
- `rng::Union{AbstractRNG,Int}=Xoshiro(42)`: RNG, or integer seed, used for
  partitioning and propagated to models/resampling strategies that support an
  `rng` parameter.
- `aggrfunc` and additional `kwargs`: Treatment options forwarded to
  `TreatmentGroup`. If `aggrfunc` is omitted, modal models use dimensionality
  reduction over three windows and non-modal models aggregate each whole
  window.

# Returns
A [`DataSet`](@ref). This function configures the machine but does not fit it.

# Errors
Throws an error if a modal model is paired with tabular data, or a non-modal
model is paired with multidimensional data.

# Examples
```julia
using SoleXplorer, MLJ, DataFrames

X, y = @load_iris

ds = setup_dataset(
    DataFrame(X),
    y;
    model=DecisionTreeClassifier(),
    resampling=CV(nfolds=10, shuffle=true),
    rng=1,
)
```

# See also
[`DataSet`](@ref), [`get_X`](@ref), [`get_y`](@ref), [`solexplorer`](@ref)
"""
function setup_dataset(
    X::Matrix{T},
    y::Union{Nothing,AbstractVector{<:Label}}=nothing;
    model::MLJ.Model,
    vnames::Vector{String}=["V$i" for i in 1:size(X, 2)],
    w::Union{Nothing,Vector}=nothing,
    resampling::ResamplingStrategy=Holdout(fraction_train=0.7, shuffle=true),
    valid_ratio::Real=0.0,
    balance::Union{Nothing,S,Tuple{Vararg{S}}}=nothing,
    float_type::Type=Float32,
    rng::Union{AbstractRNG,Int}=Xoshiro(42),
    kwargs...
) where {T,S<:DT.AbstractBalance}
    treatment = haskey(kwargs, :aggrfunc) ?
        TreatmentGroup(; kwargs...) :
        default_treatment(model; kwargs...)

    dt = DT.load_dataset(X, vnames, y, treatment; balance, float_type)
    _setup_dataset(dt; model, w, resampling, valid_ratio, rng)
end

setup_dataset(
    df::AbstractDataFrame,
    y::AbstractVector{<:Label},
    args...;
    kwargs...
) = setup_dataset(Matrix(df), y; vnames=names(df), kwargs...)


setup_dataset(df::AbstractDataFrame, args...; kwargs...) =
    setup_dataset(Matrix(df), nothing; vnames=names(df), kwargs...)
