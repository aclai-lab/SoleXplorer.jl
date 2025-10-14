using Test
using SoleXplorer
const SX = SoleXplorer

using MLJ
using DataFrames, Random

# ---------------------------------------------------------------------------- #
#                                load dataset                                  #
# ---------------------------------------------------------------------------- #/home/paso/Documents/Aclai/Sole/SoleXplorer.jl
Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

natopsloader = NatopsLoader()
Xts, yts = SX.load(natopsloader)


# https://discourse.julialang.org/t/looking-for-some-best-practices-for-optimizing-julia-code-performance/124117
# https://stackoverflow.com/questions/52018024/how-to-kill-a-task-in-julia

function fill_matrix_parallel(
    rows::Int64,
    cols::Int64,
    f::Function
)::Matrix{Int64}
    # M = Matrix{Any}(undef, rows, cols)
    M = Matrix{Int64}(undef, rows, cols)
    
    Threads.@threads for idx in 1:(rows * cols)
    # for idx in 1:(rows * cols)
        i = ((idx - 1) ÷ cols) + 1
        j = ((idx - 1) % cols) + 1
        M[i, j] = f(i, j)
    end
    
    return M
end

@show Threads.nthreads()
@btime fill_matrix_parallel(2000, 2000, (i, j) -> i * j)
# 8 threads
# 12.398 ms (3996775 allocations: 91.51 MiB)
# Matrix{Int}
# 2.249 ms (61 allocations: 30.52 MiB)
# ::Matrix{Int64}
# 2.267 ms (45 allocations: 30.52 MiB)

# 1 threads
# 39.596 ms (3996733 allocations: 91.50 MiB)

# GC.gc()

