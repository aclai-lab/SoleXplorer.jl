# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
abstract type AbstractPartitionInfo end
abstract type AbstractPartitionIdxs end

# ---------------------------------------------------------------------------- #
#                                 dataset info                                 #
# ---------------------------------------------------------------------------- #
struct PartitionInfo{T} <: AbstractPartitionInfo
    type        :: T
    train_ratio :: Real
    valid_ratio :: Real
    rng         :: AbstractRNG

    function PartitionInfo(
        type        :: T,
        train_ratio :: Real,
        valid_ratio :: Real,
        rng         :: AbstractRNG,
    )::PartitionInfo where {T<:MLJ.ResamplingStrategy}
        # Validate ratios
        0 ≤ train_ratio ≤ 1 || throw(ArgumentError("train_ratio must be between 0 and 1"))
        0 ≤ valid_ratio ≤ 1 || throw(ArgumentError("valid_ratio must be between 0 and 1"))

        new{T}(type, train_ratio, valid_ratio, rng)
    end
end

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
        test  isa UnitRange ? collect(test) : test
    )
    end
end

# ---------------------------------------------------------------------------- #
#                                 constructors                                 #
# ---------------------------------------------------------------------------- #
# partiziona il dataset creando gli indici validi per la parte di test e train, opzionalmente anche gli indici di validazione
# restituisce un vettore di `PartitionIdxs` in accordo con il tipo di resampling specificato
function partition end

function partition(
    y           :: AbstractVector{<:Label};
    resample    :: MLJ.ResamplingStrategy,
    train_ratio :: Real,
    valid_ratio :: Real,
    rng         :: AbstractRNG
)::Tuple{Vector{PartitionIdxs}, PartitionInfo}
    pinfo = PartitionInfo(resample, train_ratio, valid_ratio, rng)

    ttpairs = MLJBase.train_test_pairs(resample, 1:length(y), y)

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

Base.length(t::PartitionIdxs) = length(t.train) + length(t.valid) + length(t.test)

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