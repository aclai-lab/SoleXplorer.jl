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
    rng         :: AbstractRNG

    function PartitionInfo(
        type        :: T,
        valid_ratio :: Real,
        rng         :: AbstractRNG,
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
    "Extract training indices from partition."
    train :: Vector{T}
    "Extract validation indices from partition."
    valid :: Vector{T}
    "Extract test indices from partition."
    test  :: Vector{T}

    function PartitionIdxs(
        train :: Union{Vector{T}, UnitRange{T}},
        valid :: Union{Vector{T}, UnitRange{T}},
        test  :: Union{Vector{T}, UnitRange{T}},
    ) where T<:Int
    new{T}(
        train isa UnitRange ? collect(train) : train, 
        valid isa UnitRange ? collect(valid) : valid, 
        test  isa UnitRange ? collect(test) : test
    )
    end
end

# ---------------------------------------------------------------------------- #
#                                 constructors                                 #
# ---------------------------------------------------------------------------- #
function partition end

# create data partitions using MLJ resampling strategies
function partition(
    y           :: AbstractVector{<:Label};
    resampling  :: MLJ.ResamplingStrategy,
    valid_ratio :: Real,
    rng         :: AbstractRNG
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
get_train(t::PartitionIdxs) = t.train
get_valid(t::PartitionIdxs) = t.valid
get_test(t::PartitionIdxs)  = t.test

Base.length(t::PartitionIdxs) = length(get_train(t)) + length(get_valid(t)) + length(get_test(t))

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