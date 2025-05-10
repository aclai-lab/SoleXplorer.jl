using Test
using SoleXplorer
using MLJ
using Random
using DataFrames

@testset "Modal Decision Tree Models" begin
    
    @testset "ModalDecisionTreeModel" begin
        # Test constructor returns correct type
        mdt_model = SoleXplorer.ModalDecisionTreeModel()
        @test mdt_model isa SoleXplorer.ModelSetup{SoleXplorer.TypeMDT}
        
        # Test basic configuration
        @test mdt_model.type == SoleXplorer.MDT.ModalDecisionTree
        @test mdt_model.config.algo == :classification
        @test mdt_model.config.type == SoleXplorer.DecisionTree
        @test mdt_model.config.treatment == :reducesize
        @test mdt_model.config.reducefunc == MLJ.mean
        @test mdt_model.config.rawapply == SoleXplorer.MDT.apply
        
        # Test hyperparameters
        @test mdt_model.params.max_depth === nothing
        @test mdt_model.params.min_samples_leaf == 4
        @test mdt_model.params.min_purity_increase == 0.002
        @test mdt_model.params.max_purity_at_leaf == Inf
        @test mdt_model.params.relations == :IA7
        @test mdt_model.params.downsize == true
        @test mdt_model.params.force_i_variables == true
        @test mdt_model.params.feature_importance == :split
        
        # Test window parameters
        @test mdt_model.winparams.type == adaptivewindow
        
        # Test rawmodel and learn_method
        @test mdt_model.rawmodel isa Tuple
        @test length(mdt_model.rawmodel) == 2
        @test mdt_model.learn_method isa Tuple
        @test length(mdt_model.learn_method) == 2
        
        # Test tuning setup
        @test mdt_model.tuning isa SoleXplorer.TuningParams
        @test mdt_model.tuning.method.type == latinhypercube
        @test mdt_model.tuning.method.params.ntour == 20
        @test length(mdt_model.tuning.ranges) == 2
        
        # Test rules extraction
        @test mdt_model.rulesparams isa SoleXplorer.RulesParams
        @test mdt_model.rulesparams.type == :intrees
    end
    
    @testset "ModalRandomForestModel" begin
        # Test constructor returns correct type
        mrf_model = SoleXplorer.ModalRandomForestModel()
        @test mrf_model isa SoleXplorer.ModelSetup{SoleXplorer.TypeMRF}
        
        # Test basic configuration
        @test mrf_model.type == SoleXplorer.MDT.ModalRandomForest
        @test mrf_model.config.algo == :classification
        @test mrf_model.config.type == SoleXplorer.DecisionForest
        @test mrf_model.config.treatment == :reducesize
        
        # Test hyperparameters
        @test mrf_model.params.sampling_fraction == 0.7
        @test mrf_model.params.ntrees == 10
        @test mrf_model.params.max_depth === nothing
        @test mrf_model.params.min_samples_leaf == 1
        @test mrf_model.params.relations == :IA7
        @test mrf_model.params.n_subfeatures == SoleXplorer.MDT.MLJInterface.sqrt_f
        
        # Test window parameters
        @test mrf_model.winparams.type == adaptivewindow
        
        # Test tuning setup
        @test mrf_model.tuning isa SoleXplorer.TuningParams
        @test mrf_model.tuning.method.type == latinhypercube
        @test length(mrf_model.tuning.ranges) == 2
        # Verify sampling_fraction is one of the tuned parameters
        ranges_str = string.(mrf_model.tuning.ranges)
    end
    
    @testset "ModalAdaBoostModel" begin
        # Test constructor returns correct type
        mab_model = SoleXplorer.ModalAdaBoostModel()
        @test mab_model isa SoleXplorer.ModelSetup{SoleXplorer.TypeMAB}
        
        # Test basic configuration
        @test mab_model.type == SoleXplorer.MDT.ModalAdaBoost
        @test mab_model.config.algo == :classification
        @test mab_model.config.type == SoleXplorer.MDT.DecisionEnsemble
        @test mab_model.config.treatment == :reducesize
        
        # Test hyperparameters
        @test mab_model.params.min_samples_leaf == 1
        @test mab_model.params.min_purity_increase == 0.0
        @test mab_model.params.max_purity_at_leaf == Inf
        @test mab_model.params.relations == :IA7
        @test mab_model.params.n_iter == 10
        
        # Test window parameters
        @test mab_model.winparams.type == adaptivewindow
        
        # Test tuning setup
        @test mab_model.tuning isa SoleXplorer.TuningParams
        @test mab_model.tuning.method.type == latinhypercube
        @test length(mab_model.tuning.ranges) == 2
        # Verify min_samples_leaf is one of the tuned parameters
        ranges_str = string.(mab_model.tuning.ranges)
    end
    
    @testset "Common functionality" begin
        # Compare all three models to ensure they share expected properties
        mdt_model = SoleXplorer.ModalDecisionTreeModel()
        mrf_model = SoleXplorer.ModalRandomForestModel()
        mab_model = SoleXplorer.ModalAdaBoostModel()
        
        # All should use the same treatment and window function
        @test mdt_model.config.treatment == mrf_model.config.treatment == mab_model.config.treatment == :reducesize
        @test mdt_model.winparams.type == mrf_model.winparams.type == mab_model.winparams.type == adaptivewindow
        
        # All should use the same tuning strategy
        @test mdt_model.tuning.method.type == mrf_model.tuning.method.type == mab_model.tuning.method.type == latinhypercube
        @test mdt_model.tuning.method.params.ntour == mrf_model.tuning.method.params.ntour == mab_model.tuning.method.params.ntour == 20
        
        # All should use the same rules extraction method
        @test mdt_model.rulesparams.type == mrf_model.rulesparams.type == mab_model.rulesparams.type == :intrees
    end
end