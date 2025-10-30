using Test
using SoleXplorer
const SX = SoleXplorer

using MLJ, Random

# ---------------------------------------------------------------------------- #
#                                load dataset                                  #
# ---------------------------------------------------------------------------- #
Xc, yc = @load_iris
Xr, yr = @load_boston

natopsloader = NatopsLoader()
Xts, yts = try
    SX.load(natopsloader)
catch
    using SoleData.Artifacts
    @test_nowarn fillartifacts() # fill your Artifacts.toml file
    SX.load(natopsloader)
end

model = symbolic_analysis(Xts, yts, seed=11)

@btime symbolic_analysis(Xts, yts, seed=11)
# 11.728 ms (20570 allocations: 2.09 MiB)

# ---------------------------------------------------------------------------- #
model = DecisionTreeClassifier()
balancing = (oversampler=BorderlineSMOTE1(m=6, k=4), undersampler=ClusterUndersampler())
seed = 11

function partition(
    y           :: AbstractVector,
    resampling  :: MLJ.ResamplingStrategy,
    valid_ratio :: Real,
    rng         :: Random.AbstractRNG
)::Tuple{Vector{SX.PartitionIdxs}, SX.PartitionInfo}
    pinfo = SX.PartitionInfo(resampling, valid_ratio, rng)

    ttpairs = MLJ.MLJBase.train_test_pairs(resampling, 1:length(y), y)

    if valid_ratio == 0.0
        # return ([SX.PartitionIdxs(train, eltype(train)[], test) for (train, test) in ttpairs], pinfo)
        return (fetch.([Threads.@spawn SX.PartitionIdxs(train, eltype(train)[], test) for (train, test) in ttpairs]), pinfo)
    else
        tvalid = collect((MLJ.partition(t[1], 1-valid_ratio)..., t[2]) for t in ttpairs)
        return ([SX.PartitionIdxs(train, valid, test) for (train, valid, test) in tvalid], pinfo)
    end
end

@btime partition(repeat(yts, 100), CV(nfolds=5, shuffle=true), 0.0, Xoshiro(seed))
# 348.039 μs (155 allocations: 2.89 MiB)

using DataFrames
X = DataFrame(Xc)
function code_dataset(X::AbstractDataFrame)
    Threads.@threads for (name, col) in collect(pairs(eachcol(X)))
        if !(eltype(col) <: Number)
            # handle mixed types by converting to string first
            eltype(col) == AbstractString || (col = string.(coalesce.(col, "missing")))
            X[!, name] = MLJ.levelcode.(categorical(col))
        end
    end
    
    return X
end

@btime code_dataset(X)
# 97.398 ns (2 allocations: 96 bytes)
# 116.847 ns (4 allocations: 224 bytes)

Threads.@threads for (i, col) in collect(enumerate(eachcol(X)))
    @show typeof(col)
end


# https://docs.julialang.org/en/v1/manual/multi-threading/
# https://www.lesswrong.com/posts/kPnjPfp2ZMMYfErLJ/julia-tasks-101
# https://discourse.julialang.org/t/looking-for-some-best-practices-for-optimizing-julia-code-performance/124117

function fill_matrix_parallel(
    rows::Int64,
    cols::Int64,
    f::Function
)::Matrix{Int64}
    M = Matrix{Int64}(undef, rows, cols)

    Threads.@threads for idx in CartesianIndices((1:rows, 1:cols)) # multi thread
        M[idx] = f(idx[1], idx[2])
    end
    
    return M
end

@show Threads.nthreads()

@btime fill_matrix_parallel(2000, 2000, (i, j) -> i * j)
# single thread
# 2.260 ms (3 allocations: 30.52 MiB)
# 12 threads
# 1.393 ms (65 allocations: 30.52 MiB)
