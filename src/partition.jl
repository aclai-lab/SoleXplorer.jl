# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
# base type for partition metadata containers
abstract type AbstractPartitionInfo end

# base type for partition index containers
abstract type AbstractPartitionIdxs end

# ---------------------------------------------------------------------------- #
#                                 dataset info                                 #
# ---------------------------------------------------------------------------- #
# container for partition strategy metadata and parameters

# fields
# - type::T: MLJ resampling strategy (e.g., CV, Holdout, StratifiedCV)
# - valid_ratio::Real: Proportion of data for validation (0.0-1.0), optinal for XGBoost
struct PartitionInfo{T} <: AbstractPartitionInfo
    type        :: T
    valid_ratio :: Real
    rng         :: Random.AbstractRNG

    function PartitionInfo(
        type        :: T,
        valid_ratio :: Real,
        rng         :: Random.AbstractRNG,
    )::PartitionInfo where {T<:MLJ.ResamplingStrategy}
        0 ≤ valid_ratio ≤ 1 || throw(ArgumentError("valid_ratio must be between 0 and 1"))

        new{T}(type, valid_ratio, rng)
    end
end

# ---------------------------------------------------------------------------- #
#                                  base show                                   #
# ---------------------------------------------------------------------------- #
function Base.show(io::IO, info::PartitionInfo)
    println(io, "PartitionInfo:")
    for field in fieldnames(PartitionInfo)
        value = getfield(info, field)
        println(io, "  ", rpad(String(field) * ":", 15), value)
    end
end

function Base.show(io::IO, ::MIME"text/plain", info::PartitionInfo)
    show(io, info)
end

# ---------------------------------------------------------------------------- #
#                             partition indexes                                #
# ---------------------------------------------------------------------------- #
# container for train/validation/test index vectors
struct PartitionIdxs{T<:Int} <: AbstractPartitionIdxs
    train :: Vector{T}
    valid :: Vector{T}
    test  :: Vector{T}

    function PartitionIdxs(
        train :: Union{Vector{T}, UnitRange{T}},
        valid :: Union{Vector{T}, UnitRange{T}},
        test  :: Union{Vector{T}, UnitRange{T}},
    ) where T<:Int
    new{T}(
        train isa UnitRange ? collect(train) : train, 
        valid isa UnitRange ? collect(valid) : valid, 
        test  isa UnitRange ? collect(test)  : test
    )
    end
end

# ---------------------------------------------------------------------------- #
#                                 constructors                                 #
# ---------------------------------------------------------------------------- #
"""
    partition(y; resampling, valid_ratio, rng) -> (parts::Vector{PartitionIdxs}, info::PartitionInfo)

Create data partitions from labels `y` using an MLJ `resampling` strategy.

Arguments
- `y::AbstractVector{<:Label}`: Labels used for stratification where applicable.
- `resampling::MLJ.ResamplingStrategy`: MLJ resampling strategy (e.g., `CV`, `Holdout`, `StratifiedCV`).
- `valid_ratio::Real`: Fraction of the training split to allocate to validation (0.0–1.0). If `0.0`,
  no validation indices are produced.
- `rng::Random.AbstractRNG`: RNG controlling reproducibility of the MLJ partitioning.

Returns
- `parts::Vector{PartitionIdxs}`: One `PartitionIdxs` per train/test (and optional validation) split.
  Validation indices are empty when `valid_ratio == 0.0`.
- `info::PartitionInfo`: Captures the resampling strategy, `valid_ratio`, and `rng` used.

Notes
- Train/test indices are obtained via `MLJBase.train_test_pairs`.
- When `valid_ratio > 0`, each train split is further split by `MLJ.partition(train, 1 - valid_ratio)`
  to produce validation indices.
"""
function partition(
    y           :: AbstractVector{<:Label};
    resampling  :: MLJ.ResamplingStrategy,
    valid_ratio :: Real,
    rng         :: Random.AbstractRNG
)::Tuple{Vector{PartitionIdxs}, PartitionInfo}
    pinfo = PartitionInfo(resampling, valid_ratio, rng)

    ttpairs = MLJBase.train_test_pairs(resampling, 1:length(y), y)

    if valid_ratio == 0.0
        return ([PartitionIdxs(train, eltype(train)[], test) for (train, test) in ttpairs], pinfo)
    else
        tvalid = collect((MLJ.partition(t[1], 1-valid_ratio)..., t[2]) for t in ttpairs)
        return ([PartitionIdxs(train, valid, test) for (train, valid, test) in tvalid], pinfo)
    end
end

# ---------------------------------------------------------------------------- #
#                                   methods                                    #
# ---------------------------------------------------------------------------- #
Base.length(t::PartitionIdxs) = length(t.train) + length(t.valid) + length(t.test)

"""
    get_train(t::PartitionIdxs) -> Vector{Int}

Extract training indices from partition.
"""
get_train(t::PartitionIdxs) = t.train

"""
    get_valid(t::PartitionIdxs) -> Vector{Int}

Extract validation indices from partition.
"""
get_valid(t::PartitionIdxs) = t.valid

"""
    get_test(t::PartitionIdxs) -> Vector{Int}

Extract test indices from partition.
"""
get_test(t::PartitionIdxs)  = t.test

# ---------------------------------------------------------------------------- #
#                                  base show                                   #
# ---------------------------------------------------------------------------- #
function Base.show(io::IO, pidx::PartitionIdxs{T}) where T
    n_train = length(pidx.train)
    n_valid = length(pidx.valid)
    n_test  = length(pidx.test)
    total   = n_train + n_valid + n_test
    
    print(io, "PartitionIdxs{$T}")
    print(io, "\n  Total samples: $total, Train: $n_train,Valid: $n_valid, Test: $n_test.")
end

function Base.show(io::IO, ::MIME"text/plain", pidx::PartitionIdxs{T}) where T
    show(io, pidx)
end

# ---------------------------------------------------------------------------- #
#                        parametrized cross validation                         #
# ---------------------------------------------------------------------------- #
"""
    pCV <: MLJ.ResamplingStrategy

Custom MLJ resampling strategy for parametrized cross-validation with configurable train/test split ratio.

Unlike standard k-fold cross-validation where the train/test split is determined by the number of folds,
`pCV` allows you to specify both the number of folds and the exact fraction of data allocated to training
in each fold.

Fields
- `nfolds::Int`: Number of folds (must be > 1).
- `fraction_train::Float64`: Fraction of data to use for training in each fold (0.0–1.0).
- `shuffle::Bool`: Whether to shuffle the data before partitioning.
- `rng::Union{Int,AbstractRNG}`: Random number generator or seed for reproducibility.

Constructor
    pCV(; nfolds=6, fraction_train=0.7, shuffle=nothing, rng=nothing)

Example
```julia
using MLJ, Random

# Create a pCV resampling strategy with 10 folds, 60% training data
resampling = pCV(nfolds=10, fraction_train=0.6, shuffle=true, rng=Xoshiro(42))

# Use with SoleXplorer.partition
partition_idxs, info = SoleXplorer.partition(y; resampling, valid_ratio=0.0, rng=Xoshiro(42))
```

Notes
- Each fold produces a different random train/test split with the specified ratio.
- This differs from standard CV where test sets are disjoint partitions of the data.
- Useful for scenarios requiring repeated random splits with a specific train/test ratio.
"""
struct pCV <: ResamplingStrategy
    nfolds::Int
    fraction_train::Float64
    shuffle::Bool
    rng::Union{Int,AbstractRNG}
    function pCV(nfolds, fraction_train, shuffle, rng)
        nfolds > 1 || throw(ArgumentError("Must have nfolds > 1. "))
        return new(nfolds, fraction_train, shuffle, rng)
    end
end

# Constructor with keywords
pCV(; nfolds::Int=6, fraction_train::Float64=0.7, shuffle=nothing, rng=nothing) =
    pCV(nfolds, fraction_train, MLJBase.shuffle_and_rng(shuffle, rng)...)

function MLJBase.train_test_pairs(pcv::pCV, rows)
    return map(1:pcv.nfolds) do _
        MLJBase.partition(rows, pcv.fraction_train, shuffle=pcv.shuffle, rng=pcv.rng)
    end
end