using Distributed
addprocs(2)

@everywhere begin
    using SoleXplorer
    using Test
    using Random
    using MLJ
    using DataFrames
    # using MLJDecisionTreeInterface
    # using SoleModels
    # using StatsBase
    # using Catch22
end

function run_tests(list)
    println("\n" * ("#"^50))
    for test in list
        println("TEST: $test")
        include(test)
    end
end

println("Julia version: ", VERSION)

test_suites = [
    ("Prepare Dataset", ["prepare_dataset.jl", ]),
    ("Train and test", ["train_test.jl", ]),
    # ("Interfaces", [
    #     "interfaces/base_interface.jl",
    #     "interfaces/dataset_interface.jl",
    #     "interfaces/resample_interface.jl",
    #     "interfaces/windowing_interface.jl",
    #     "interfaces/tuning_interface.jl",
    #     "interfaces/extractrules_interface.jl",
    #     "interfaces/model_interface.jl"
    # ]),
    # ("models", [
    #     "models/decisiontrees.jl",
    #     "models/modaldecisiontrees.jl",
    #     "models/xgboost.jl"
    # ]),
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
