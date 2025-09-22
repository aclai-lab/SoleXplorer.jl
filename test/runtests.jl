using Test
using SoleData.Artifacts
# fill your Artifacts.toml file;
@test_nowarn fillartifacts()

# Loader lists
# abcloader = ABCLoader()
# mitloader = MITESPRESSOLoader()

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
    ("Solemodel robustness", ["robustness.jl"         ]),
    ("Rule extraction",      ["rule_extraction.jl"    ]),
    ("Association Rules",    ["associationrules.jl"   ]),
    ("Serialization",        ["serialize.jl"          ]),
    ("Collection",           ["analysis_collection.jl"]),
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
