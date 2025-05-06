using SoleXplorer
using Test

function run_tests(list)
    println("\n" * ("#"^50))
    for test in list
        println("TEST: $test")
        include(test)
    end
end

println("Julia version: ", VERSION)

test_suites = [
    ("Utils", ["utils/featureset.jl", ]),
    ("Interfaces", [
        "interfaces/base_interface.jl",
        "interfaces/dataset_interface.jl",
        "interfaces/resample_interface.jl",
        "interfaces/windowing_interface.jl",
        "interfaces/tuning_interface.jl",
        "interfaces/extractrules_interface.jl",
        "interfaces/model_interface.jl"
    ])
    # ("Prepare Dataset", ["modules/prepare_dataset.jl", ]),
    # ("Train_Test", ["modules/train_test.jl", ]),
    # ("Train_Test", ["modules/symbolic_analisys.jl", ]),
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
