# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
"""
Abstract type for dataset configuration outputs
"""
abstract type AbstractDatasetSetup end

"""
Abstract type for dataset train, test and validation indexing
"""
abstract type AbstractIndexCollection end

"""
Abstract type for dataset outputs
"""
abstract type AbstractDataset end

# ---------------------------------------------------------------------------- #
#                                  dataset info                                #
# ---------------------------------------------------------------------------- #
"""
    DatasetInfo <: AbstractDatasetSetup

An immutable struct containing dataset configuration and metadata for machine learning tasks.
`DatasetInfo` provides all the necessary information about how a dataset is processed,
partitioned, and what features are extracted from it.

# Fields
- `algo::Symbol`: Algorithm type:
  - `:classification`: For categorical target variables
  - `:regression`: For numerical target variables

- `treatment::Symbol`: Data treatment method for time-series data:
  - `:aggregate`: Reduces time-series to scalar features (propositional approach)
  - `:reducesize`: Windows time-series to reduce dimensions while preserving temporal structure

- `features::Vector{<:Base.Callable}`: Feature extraction functions applied to the dataset.
  Each function should accept a vector/array and return a scalar value (e.g., `mean`, `std`, `maximum`).

- `train_ratio::Real`: Proportion of data used for training (range: 0-1).
  Controls the train/test split ratio: higher values allocate more data for training.

- `valid_ratio::Real`: Proportion of training data used for validation (range: 0-1).
  - When `1.0`: No separate validation set is created (empty array)
  - When `< 1.0`: Creates validation set from the training portion

- `shuffle::Bool`: Whether to randomly shuffle data before partitioning:
  - `true`: Randomizes data order for better generalization
  - `false`: Preserves original data order (useful for time-series with temporal dependencies)

- `stratified::Bool`: Whether to use stratified sampling for cross-validation:
  - `true`: Maintains class distribution across folds (for classification tasks)
  - `false`: Simple random sampling without preserving class ratios

- `nfolds::Int`: Number of cross-validation folds when `stratified=true`.
  Higher values give more robust performance estimates but increase computation time.

- `rng::AbstractRNG`: Random number generator for reproducible partitioning and shuffling.

- `winparams::SoleFeatures.WinParams`: Windowing parameters for time-series processing:
  - `type`: Window function type (`wholewindow`, `adaptivewindow`, `movingwindow`, `splitwindow`)
  - Additional parameters specific to each window type:
    - `wholewindow`: Uses entire time-series (no parameters needed)
    - `adaptivewindow`: Uses `nwindows` and `relative_overlap`
    - `movingwindow`: Uses `window_size` and `window_step`
    - `splitwindow`: Uses `nwindows` for equal divisions

- `vnames::Union{Vector{<:AbstractString}, Nothing}`: Variable/column names.
  When `nothing`, column indices are used as identifiers.
"""
struct DatasetInfo <: AbstractDatasetSetup
    algo        :: Symbol
    treatment   :: Symbol
    reducefunc  :: Union{<:Base.Callable, Nothing}
    train_ratio :: Real
    valid_ratio :: Real
    rng         :: AbstractRNG
    resample    :: Bool
    vnames      :: Union{Vector{<:AbstractString}, Nothing}

    function DatasetInfo(
        algo        :: Symbol,
        treatment   :: Symbol,
        reducefunc  :: Union{<:Base.Callable, Nothing},
        train_ratio :: Real,
        valid_ratio :: Real,
        rng         :: AbstractRNG,
        resample    :: Bool,
        vnames      :: Union{Vector{<:AbstractString}, Nothing}
    )::DatasetInfo
        # Validate ratios
        0 ≤ train_ratio ≤ 1 || throw(ArgumentError("train_ratio must be between 0 and 1"))
        0 ≤ valid_ratio ≤ 1 || throw(ArgumentError("valid_ratio must be between 0 and 1"))

        new(algo, treatment, reducefunc, train_ratio, valid_ratio, rng, resample, vnames)
    end
end

get_resample(dsinfo::DatasetInfo)::Bool = dsinfo.resample

function Base.show(io::IO, info::DatasetInfo)
    println(io, "DatasetInfo:")
    for field in fieldnames(DatasetInfo)
        value = getfield(info, field)
        println(io, "  ", rpad(String(field) * ":", 15), value)
    end
end

# ---------------------------------------------------------------------------- #
#                              indexes collection                              #
# ---------------------------------------------------------------------------- #
"""
    TT_indexes{T<:Integer} <: AbstractVector{T}

A struct that stores indices for train-validation-test splits of a dataset,
used in Dataset struct.

# Fields
- `train::Vector{T}`: Vector of indices for the training set
- `valid::Vector{T}`: Vector of indices for the validation set
- `test::Vector{T}`:  Vector of indices for the test set
"""
struct TT_indexes{T<:Integer} <: AbstractIndexCollection
    train :: Vector{T}
    valid :: Vector{T}
    test  :: Vector{T}

    function TT_indexes(
        train :: AbstractVector{T},
        valid :: AbstractVector{T},
        test  :: AbstractVector{T}
    ) where {T<:Integer}
        new{T}(train, valid, test)
    end
end

Base.show(io::IO, t::TT_indexes) = print(io, "TT_indexes(train=", t.train, ", validation=", t.valid, ", test=", t.test, ")")
Base.length(t::TT_indexes) = length(t.train) + length(t.valid) + length(t.test)

# ---------------------------------------------------------------------------- #
#                                   dataset                                    #
# ---------------------------------------------------------------------------- #
"""
    Dataset{T<:AbstractMatrix,S} <: AbstractDataset

An immutable struct that efficiently stores dataset splits for machine learning.

# Fields
- `X::T`: The feature matrix as a DataFrame
- `y::S`: The target vector
- `tt::Union{TT_indexes{I}, Vector{TT_indexes{I}}}`: Train-test split indices
- `info::DatasetInfo`: Dataset metadata and configuration
- `Xtrain`, `Xvalid`, `Xtest`: Data views for features
- `ytrain`, `yvalid`, `ytest`: Data views for targets
"""
struct Dataset{T<:AbstractMatrix,S} <: AbstractDataset
    X           :: T
    y           :: S
    tt          :: Union{TT_indexes, AbstractVector{<:TT_indexes}}
    info        :: DatasetInfo
    Xtrain      :: Union{AbstractMatrix, Vector{<:AbstractMatrix}}
    Xvalid      :: Union{AbstractMatrix, Vector{<:AbstractMatrix}}
    Xtest       :: Union{AbstractMatrix, Vector{<:AbstractMatrix}}
    ytrain      :: Union{SubArray{<:eltype(S)}, Vector{<:SubArray{<:eltype(S)}}}
    yvalid      :: Union{SubArray{<:eltype(S)}, Vector{<:SubArray{<:eltype(S)}}}
    ytest       :: Union{SubArray{<:eltype(S)}, Vector{<:SubArray{<:eltype(S)}}}

    function Dataset(X::T, y::S, tt, info) where {T<:AbstractMatrix,S}
        if get_resample(info)
            Xtrain = view.(Ref(X), getfield.(tt, :train), Ref(:))
            Xvalid = view.(Ref(X), getfield.(tt, :valid), Ref(:))
            Xtest  = view.(Ref(X), getfield.(tt, :test), Ref(:))
            ytrain = view.(Ref(y), getfield.(tt, :train))
            yvalid = view.(Ref(y), getfield.(tt, :valid))
            ytest  = view.(Ref(y), getfield.(tt, :test))
        else
            Xtrain = @views X[tt.train, :]
            Xvalid = @views X[tt.valid, :]
            Xtest  = @views X[tt.test,  :]
            ytrain = @views y[tt.train]
            yvalid = @views y[tt.valid]
            ytest  = @views y[tt.test]
        end

        new{T,S}(X, y, tt, info, Xtrain, Xvalid, Xtest, ytrain, yvalid, ytest)
    end
end

function Base.show(io::IO, ds::Dataset)
    println(io, "Dataset:")
    println(io, "  X shape:        ", size(ds.X))
    println(io, "  y length:       ", length(ds.y))
    if ds.tt isa AbstractVector
        println(io, "  Train/Valid/Test:     ", length(ds.tt), " folds")
    else
        println(io, "  Train indices:  ", length(ds.tt.train))
        println(io, "  Valid indices:  ", length(ds.tt.valid))
        println(io, "  Test indices:   ", length(ds.tt.test))
    end
    print(io, ds.info)
end