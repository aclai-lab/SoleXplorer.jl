# ---------------------------------------------------------------------------- #
#                                  dataset info                                #
# ---------------------------------------------------------------------------- #
"""
    DatasetInfo(
        treatment::Symbol,
        modalreduce::OptCallable,
        train_ratio::Real,
        valid_ratio::Real,
        rng::AbstractRNG,
        vnames::OptStringVec
    ) -> DatasetInfo

Create a configuration for dataset preparation and splitting in machine learning workflows.

# Fields
- `treatment::Symbol`: Data treatment method (e.g., `:standardize`, `:normalize`)
- `modalreduce::OptCallable`: Optional function for dimensionality reduction
- `train_ratio::Real`: Proportion of data to use for training (must be between 0 and 1)
- `valid_ratio::Real`: Proportion of data to use for validation (must be between 0 and 1)
- `rng::AbstractRNG`: Random number generator for reproducible splits
- `vnames::OptStringVec`: Optional feature/variable names
"""
struct DatasetInfo <: AbstractDatasetSetup
    treatment   :: Symbol
    modalreduce :: OptCallable
    train_ratio :: Real
    valid_ratio :: Real
    rng         :: AbstractRNG
    vnames      :: OptStringVec

    function DatasetInfo(
        treatment   :: Symbol,
        modalreduce  :: OptCallable,
        train_ratio :: Real,
        valid_ratio :: Real,
        rng         :: AbstractRNG,
        vnames      :: OptStringVec
    )::DatasetInfo
        # Validate ratios
        0 ≤ train_ratio ≤ 1 || throw(ArgumentError("train_ratio must be between 0 and 1"))
        0 ≤ valid_ratio ≤ 1 || throw(ArgumentError("valid_ratio must be between 0 and 1"))

        new(treatment, modalreduce, train_ratio, 1-valid_ratio, rng, vnames)
    end
end

get_treatment(dsinfo::DatasetInfo)   :: Symbol = dsinfo.treatment
get_modalreduce(dsinfo::DatasetInfo) :: OptCallable = dsinfo.modalreduce
get_train_ratio(dsinfo::DatasetInfo) :: Real = dsinfo.train_ratio
get_valid_ratio(dsinfo::DatasetInfo) :: Real = dsinfo.valid_ratio
get_rng(dsinfo::DatasetInfo)         :: AbstractRNG = dsinfo.rng
get_vnames(dsinfo::DatasetInfo)      :: OptStringVec = dsinfo.vnames

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
struct Dataset{T<:AbstractMatrix,S} <: AbstractDataset
    X           :: T
    y           :: S
    tt          :: Vector{<:TT_indexes}
    info        :: DatasetInfo
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

function Base.show(io::IO, ds::Dataset)
    println(io, "Dataset:")
    println(io, "  X shape:        ", size(ds.X))
    println(io, "  y length:       ", length(ds.y))
    println(io, "  Train/Valid/Test:     ", length(ds.tt), " folds")
    print(io, ds.info)
end