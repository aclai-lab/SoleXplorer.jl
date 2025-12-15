using Test
using SoleXplorer
const SX = SoleXplorer

using MLJ, Random, DataFrames

# ---------------------------------------------------------------------------- #
#                                load dataset                                  #
# ---------------------------------------------------------------------------- #
Xc, yc = @load_iris
Xr, yr = @load_boston

natopsloader = SX.NatopsLoader()
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


function treatment(
    X           :: AbstractDataFrame,
    treat       :: Symbol;
    win         :: WinFunction=AdaptiveWindow(nwindows=3, relative_overlap=0.1),
    features    :: Tuple{Vararg{Base.Callable}}=(maximum, minimum),
    modalreduce :: Base.Callable=mean
)
    vnames, intervals = propertynames(X), win(length(X[1,1]))
    nvnames, nfeatures, nintervals = length(vnames), length(features), length(intervals)

    # calculate number of cols after treatment
    treatcols  = nvnames * nfeatures * nintervals
    treatnames, treatX = Vector{Symbol}(undef, treatcols), Matrix{Float64}(undef, nrow(X), treatcols)

    # if treat == :aggregate
        Threads.@threads for f in eachindex(features)
            Threads.@threads for v in eachindex(vnames)
                if nintervals == 1
                    # single window: apply to whole time series
                    idx = (f - 1) * nvnames * nintervals + (v - 1) * nintervals
                    treatnames[idx] = Symbol("$(f)($(v))")
                    # apply_vectorized!(_X, X[!, v], f, col_name)
                else
                    # multiple windows: apply to each interval
                    for i in eachindex(intervals)
                        idx = (f - 1) * nvnames * nintervals + (v - 1) * nintervals + i
                        treatnames[idx] = Symbol("$(f)($(v))w$(i)")
                        # apply_vectorized(treatX, X[!, v], f, col_name, interval)
                        Threads.@threads for row in 1:nrow(X)
                            treatX[row,v] = features[f](@views X[row,v][intervals[i]])
                        end
                    end
                end
            end
        end
treatnames
end

function apply_vectorized(
    X::Matrix{Float64},
    X_col::Vector{<:Vector{<:Real}},
    feature_func::Function,
    col_name::Symbol,
    interval::UnitRange{Int64}
)::Vector{<:Real}
    @views @inbounds X[!, col_name] = collect(feature_func(col[interval]) for col in X_col)
end

treatment(Xts, :aggregate)
# 70.648 μs (2630 allocations: 105.38 KiB)
# 36.336 μs (2575 allocations: 107.94 KiB)
# 20.913 μs (2747 allocations: 120.25 KiB)
# 880.870 μs (119099 allocations: 11.46 MiB)

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
