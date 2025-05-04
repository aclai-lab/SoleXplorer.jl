# ---------------------------------------------------------------------------- #
#                                  dataset info                                #
# ---------------------------------------------------------------------------- #
"""
    DatasetInfo(
        algo::Symbol,
        treatment::Symbol,
        reducefunc::Union{<:Base.Callable, Nothing},
        train_ratio::Real,
        valid_ratio::Real,
        rng::AbstractRNG,
        resample::Bool,
        vnames::Union{Vector{<:AbstractString}, Nothing}
    ) -> DatasetInfo

Create a configuration for dataset preparation and splitting in machine learning workflows.

# Fields
- `algo::Symbol`: Algorithm to use for dataset processing
- `treatment::Symbol`: Data treatment method (e.g., `:standardize`, `:normalize`)
- `reducefunc::Union{<:Base.Callable, Nothing}`: Optional function for dimensionality reduction
- `train_ratio::Real`: Proportion of data to use for training (must be between 0 and 1)
- `valid_ratio::Real`: Proportion of data to use for validation (must be between 0 and 1)
- `rng::AbstractRNG`: Random number generator for reproducible splits
- `resample::Bool`: Whether to perform resampling for cross-validation
- `vnames::Union{Vector{<:AbstractString}, Nothing}`: Optional feature/variable names
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

get_algo(dsinfo::DatasetInfo)        :: Symbol = dsinfo.algo
get_treatment(dsinfo::DatasetInfo)   :: Symbol = dsinfo.treatment
get_reducefunc(dsinfo::DatasetInfo)  :: Union{<:Base.Callable, Nothing} = dsinfo.reducefunc
get_train_ratio(dsinfo::DatasetInfo) :: Real = dsinfo.train_ratio
get_valid_ratio(dsinfo::DatasetInfo) :: Real = dsinfo.valid_ratio
get_rng(dsinfo::DatasetInfo)         :: AbstractRNG = dsinfo.rng
get_resample(dsinfo::DatasetInfo)    :: Bool = dsinfo.resample
get_vnames(dsinfo::DatasetInfo)      :: Union{Vector{<:AbstractString}, Nothing} = dsinfo.vnames

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

get_train(tt::TT_indexes) = tt.train
get_valid(tt::TT_indexes) = tt.valid
get_test(tt::TT_indexes)  = tt.test

Base.show(io::IO, t::TT_indexes) = print(io, "TT_indexes(train=", t.train, ", validation=", t.valid, ", test=", t.test, ")")
Base.length(t::TT_indexes) = length(t.train) + length(t.valid) + length(t.test)

# ---------------------------------------------------------------------------- #
#                                   dataset                                    #
# ---------------------------------------------------------------------------- #
"""
    Dataset{T<:AbstractMatrix,S} <: AbstractDataset

An immutable struct that efficiently stores and manages data for machine learning, 
including train-validation-test splits with views into the original data.

# Fields
- `X::T`: Original feature matrix
- `y::S`: Original target vector/matrix
- `tt::Union{TT_indexes, AbstractVector{<:TT_indexes}}`: Train-validation-test split indices
- `info::DatasetInfo`: Dataset configuration and metadata
- `Xtrain::Union{AbstractMatrix, Vector{<:AbstractMatrix}}`: Features for training
- `Xvalid::Union{AbstractMatrix, Vector{<:AbstractMatrix}}`: Features for validation
- `Xtest::Union{AbstractMatrix, Vector{<:AbstractMatrix}}`: Features for testing
- `ytrain::Union{SubArray{<:eltype(S)}, Vector{<:SubArray{<:eltype(S)}}}`: Targets for training
- `yvalid::Union{SubArray{<:eltype(S)}, Vector{<:SubArray{<:eltype(S)}}}`: Targets for validation
- `ytest::Union{SubArray{<:eltype(S)}, Vector{<:SubArray{<:eltype(S)}}}`: Targets for testing

# Constructor
    Dataset(X::T, y::S, tt, info) where {T<:AbstractMatrix,S}

Creates a new `Dataset` with views into the data according to the provided indices.
- If `get_resample(info)` is `true`, handles multiple train-validation-test splits (e.g., for cross-validation)
- Otherwise, creates simple views for a single train-validation-test split

# Access the splits
X_train = dataset.Xtrain          # Training features
y_train = dataset.ytrain          # Training targets
```

# Note
All data views are created using Julia's view mechanism, not copies of the data,
providing memory-efficient access to data partitions.

See also: [`DatasetInfo`](@ref), [`TT_indexes`](@ref)
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

"""
    get_X(ds::Dataset) -> AbstractMatrix

Get the original feature matrix from a Dataset structure.
"""
get_X(ds::Dataset)      = ds.X

"""
    get_y(ds::Dataset) -> Any

Get the original target vector/matrix from a Dataset structure.
"""
get_y(ds::Dataset)      = ds.y

"""
    get_tt(ds::Dataset) -> Union{TT_indexes, AbstractVector{<:TT_indexes}}

Get the train-validation-test split indices from a Dataset structure.
"""
get_tt(ds::Dataset)     = ds.tt

"""
    get_info(ds::Dataset) -> DatasetInfo

Get the dataset configuration and metadata from a Dataset structure.
"""
get_info(ds::Dataset)   = ds.info

"""
    get_Xtrain(ds::Dataset) -> Union{AbstractMatrix, Vector{<:AbstractMatrix}}

Get the training feature views from a Dataset structure.
"""
get_Xtrain(ds::Dataset) = ds.Xtrain

"""
    get_Xvalid(ds::Dataset) -> Union{AbstractMatrix, Vector{<:AbstractMatrix}}

Get the validation feature views from a Dataset structure.
"""
get_Xvalid(ds::Dataset) = ds.Xvalid

"""
    get_Xtest(ds::Dataset) -> Union{AbstractMatrix, Vector{<:AbstractMatrix}}

Get the test feature views from a Dataset structure.
"""
get_Xtest(ds::Dataset)  = ds.Xtest

"""
    get_ytrain(ds::Dataset) -> Union{SubArray, Vector{<:SubArray}}

Get the training target views from a Dataset structure.
"""
get_ytrain(ds::Dataset) = ds.ytrain

"""
    get_yvalid(ds::Dataset) -> Union{SubArray, Vector{<:SubArray}}

Get the validation target views from a Dataset structure.
"""
get_yvalid(ds::Dataset) = ds.yvalid

"""
    get_ytest(ds::Dataset) -> Union{SubArray, Vector{<:SubArray}}

Get the test target views from a Dataset structure.
"""
get_ytest(ds::Dataset)  = ds.ytest

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