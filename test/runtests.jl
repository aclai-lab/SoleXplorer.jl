using Test
using SoleXplorer
using Random

function run_tests(list)
    println("\n" * ("#"^50))
    for test in list
        println("TEST: $test")
        include(test)
    end
end

println("Julia version: ", VERSION)

test_suites = [
    ("Prepare Dataset", ["modules/prepare_dataset.jl", ]),
    ("Validate Modelset", ["modules/validate_modelset.jl", ]),
    ("Get Models", ["modules/models.jl", ]),
    # ("DecisionForest", ["decision_forest.jl", ]),
    # ("ModalDecisionTree", ["modal_decision_tree.jl", ]),
    # ("AdaBoost", ["adaboost.jl", ]),
    # ("ModalAdaBoost", ["modal_adaboost.jl", ]),
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
