using Distributed
addprocs(2)

@everywhere begin
    using Test
    using SoleXplorer
    using MLJ
    using DataFrames, Random
end

const SX = SoleXplorer

function run_tests(list)
    println("\n" * ("#"^50))
    for test in list
        println("TEST: $test")
        include(test)
    end
end

println("Julia version: ", VERSION)

test_suites = [
    ("Setup Dataset",        ["dataset.jl",           ]),
    ("Train and Test",       ["train_test.jl",        ]),
    ("Symbolic Analysis",    ["symbolic_analysis.jl", ]),
    ("Solemodel robustness", ["robustness.jl"          ]),
]

@testset "SoleXplorer.jl" begin
    for ts in eachindex(test_suites)
        name = test_suites[ts][1]
        list = test_suites[ts][2]
        let
            @testset "$name" begin
                run_tests(list)
            end
        end
    end
    println()
end
