using Test
using SoleXplorer
using MLJ
using DataFrames, Random

@testset "RulesParams and Rule Extraction" begin
    
    @testset "RulesParams Struct" begin
        # Test instantiation
        params = SoleXplorer.RulesParams(:intrees, (prune_rules=true, max_rules=10))
        @test params.type == :intrees
        @test params.params.prune_rules == true
        @test params.params.max_rules == 10
    end
    
    @testset "RULES_PARAMS Default Configurations" begin
        # Test that all expected methods have default parameters
        for method in [:intrees, :refne, :trepan, :batrees, :rulecosi, :lumen]
            @test haskey(SoleXplorer.RULES_PARAMS, method)
            @test SoleXplorer.RULES_PARAMS[method] isa NamedTuple
        end
        
        # Test specific parameters
        @test SoleXplorer.RULES_PARAMS[:intrees].prune_rules == true
        @test SoleXplorer.RULES_PARAMS[:refne].L == 10
        @test SoleXplorer.RULES_PARAMS[:trepan].partial_sampling == 0.5
        @test SoleXplorer.RULES_PARAMS[:batrees].num_trees == 10
        @test SoleXplorer.RULES_PARAMS[:lumen].ott_mode == false
    end
    
    @testset "EXTRACT_RULES Methods" begin
        # Test that all expected methods are available
        for method in [:intrees, :refne, :trepan, :batrees, :rulecosi, :lumen]
            @test haskey(SoleXplorer.EXTRACT_RULES, method)
            @test SoleXplorer.EXTRACT_RULES[method] isa Function
        end
    end
    
    @testset "Rule Extraction with Mock Data" begin
        # Create a mock Modelset for testing
        # This requires setting up minimal structure to avoid actual model execution
        
        # First create a mock model, dataset and setup
        mock_model = nothing
        mock_ds = (
            Xtest = DataFrame(x1=[1,2,3], x2=[4,5,6]),
            ytest = [1, 2, 1],
            info = (vnames = [:x1, :x2],)
        )
        mock_setup = (
            resample = nothing,
            rulesparams = SoleXplorer.RulesParams(:intrees, 
                (prune_rules=true, max_rules=5, silent=true)),
            rawmodel = m -> m,
            config = (rawapply = identity,)
        )
        
        # Create a minimal Modelset-like structure
        mock_modelset = (
            model = mock_model,
            ds = mock_ds,
            setup = mock_setup,
            mach = nothing
        )
        
        # Test existence of extraction functions
        # We won't actually call them since they require real models,
        # but we'll verify the functions exist and have the right signature
        intrees_fn = SoleXplorer.EXTRACT_RULES[:intrees]
        @test intrees_fn isa Function
        
        refne_fn = SoleXplorer.EXTRACT_RULES[:refne]
        @test refne_fn isa Function
        
        trepan_fn = SoleXplorer.EXTRACT_RULES[:trepan]
        @test trepan_fn isa Function
    end

    @testset "Test the actual rule extraction process" begin
        X, y = @load_iris
        X = DataFrame(X)
        
        # Helper function to show rules and check basic properties
        function verify_rules(modelset, method_name)
            println("\n--- Rules extracted with $method_name ---")
            println(modelset.rules)
            @test !isnothing(modelset.rules)
            @test !isempty(modelset.rules)
            # Test that rules can be applied to generate predictions
            preds = MLJ.predict(modelset.mach, X)
            @test length(preds) == size(X, 1)
            return modelset
        end
        
        # Test InTrees rule extraction
        @testset "InTrees rule extraction" begin
            modelset = symbolic_analysis(
                X, y;
                model=(type=:decisiontree, params=(;max_depth=3)),
                preprocess=(;rng=Xoshiro(1)),
                extract_rules=(;type=:intrees, params=(prune_rules=true, max_rules=10))
            )
            verify_rules(modelset, "InTrees")
        end
        
        # Test REFNE rule extraction
        @testset "REFNE rule extraction" begin
            modelset = symbolic_analysis(
                X, y;
                model=(type=:randomforest, params=(;n_trees=10, max_depth=3)),
                preprocess=(;rng=Xoshiro(1)),
                extract_rules=(;type=:refne, params=(L=5, partial_sampling=0.7))
            )
            verify_rules(modelset, "REFNE")
        end
        
        # Test TREPAN rule extraction
        @testset "TREPAN rule extraction" begin
            modelset = symbolic_analysis(
                X, y;
                model=(type=:randomforest, params=(;n_trees=10, max_depth=3)),
                preprocess=(;rng=Xoshiro(1)),
                extract_rules=(;type=:trepan, params=(;max_depth=3))
            )
            verify_rules(modelset, "TREPAN")
        end
        
        # Test BATREES rule extraction
        @testset "BATREES rule extraction" begin
            modelset = symbolic_analysis(
                X, y;
                model=(type=:randomforest, params=(;n_trees=10, max_depth=3)),
                preprocess=(;rng=Xoshiro(1)),
                extract_rules=(;type=:batrees, params=(num_trees=20, max_depth=5))
            )
            verify_rules(modelset, "BATREES")
        end
        
        # Test RuleCOSI rule extraction
        @testset "RuleCOSI rule extraction" begin
            modelset = symbolic_analysis(
                X, y;
                model=(type=:randomforest, params=(;n_trees=10, max_depth=3)),
                preprocess=(;rng=Xoshiro(1)),
                extract_rules=(;type=:rulecosi)
            )
            verify_rules(modelset, "RuleCOSI")
        end
        
        # Test LUMEN rule extraction
        @testset "LUMEN rule extraction" begin
            modelset = symbolic_analysis(
                X, y;
                model=(type=:randomforest, params=(;n_trees=10, max_depth=3)),
                preprocess=(;rng=Xoshiro(1)),
                extract_rules=(;type=:lumen, params=(ott_mode=false, return_info=false))
            )
            verify_rules(modelset, "LUMEN")
        end
        
        # Test different model types with same extractor
        @testset "Different model types" begin
            model_types = [:decisiontree, :randomforest, :xgboost]
            
            for model_type in model_types
                println("\nTesting model type: $model_type")
                
                params = if model_type == :decisiontree
                    (;max_depth=3)
                elseif model_type == :randomforest
                    (;n_trees=10, max_depth=3)
                else
                    (;max_depth=3, eta=0.3, num_round=10)
                end
                
                modelset = symbolic_analysis(
                    X, y;
                    model=(type=model_type, params=params),
                    preprocess=(;rng=Xoshiro(1)),
                    extract_rules=(;type=:intrees)  # InTrees works with most model types
                )
                
                println("Rules for $model_type model:")
                println(modelset.rules)
                @test !isnothing(modelset.rules)
            end
        end
    end
end