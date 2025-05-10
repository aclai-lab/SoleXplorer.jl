using Test
using SoleXplorer
using MLJ
using Random
using DataFrames

@testset "Decision Trees Models" begin
    
    @testset "DecisionTreeClassifierModel" begin
        # Test constructor returns correct type
        dt_model = SoleXplorer.DecisionTreeClassifierModel()
        @test dt_model isa SoleXplorer.ModelSetup{SoleXplorer.TypeDTC}
        
        # Test basic configuration
        @test dt_model.type == SoleXplorer.MLJDecisionTreeInterface.DecisionTreeClassifier
        @test dt_model.config.algo == :classification
        @test dt_model.config.type == SoleXplorer.DecisionTree
        @test dt_model.config.treatment == :aggregate
        @test dt_model.config.rawapply == SoleXplorer.DT.apply_tree
        
        # Test hyperparameters
        @test dt_model.params.max_depth == -1
        @test dt_model.params.min_samples_leaf == 1
        @test dt_model.params.min_samples_split == 2
        @test dt_model.params.n_subfeatures == 0
        @test dt_model.params.post_prune == false
        
        # Test tuning setup
        @test dt_model.tuning isa SoleXplorer.TuningParams
        @test dt_model.tuning.method.type == latinhypercube
        @test dt_model.tuning.method.params.ntour == 20
        @test length(dt_model.tuning.ranges) == 2
        
        # Test rules extraction
        @test dt_model.rulesparams isa SoleXplorer.RulesParams
        @test dt_model.rulesparams.type == :intrees
    end
    
    @testset "RandomForestClassifierModel" begin
        # Test constructor returns correct type
        rf_model = SoleXplorer.RandomForestClassifierModel()
        @test rf_model isa SoleXplorer.ModelSetup{SoleXplorer.TypeRFC}
        
        # Test basic configuration
        @test rf_model.type == SoleXplorer.MLJDecisionTreeInterface.RandomForestClassifier
        @test rf_model.config.algo == :classification
        @test rf_model.config.type == SoleXplorer.DecisionEnsemble
        @test rf_model.config.rawapply == SoleXplorer.DT.apply_forest
        
        # Test hyperparameters
        @test rf_model.params.n_trees == 10
        @test rf_model.params.sampling_fraction == 0.7
        @test rf_model.params.n_subfeatures == -1
        
        # Test tuning setup
        @test rf_model.tuning isa SoleXplorer.TuningParams
        @test rf_model.tuning.method.type == latinhypercube
        @test rf_model.tuning.ranges isa Tuple
        @test length(rf_model.tuning.ranges) == 2
    end
    
    @testset "AdaBoostClassifierModel" begin
        # Test constructor returns correct type
        ab_model = SoleXplorer.AdaBoostClassifierModel()
        @test ab_model isa SoleXplorer.ModelSetup{SoleXplorer.TypeABC}
        
        # Test basic configuration
        @test ab_model.type == SoleXplorer.MLJDecisionTreeInterface.AdaBoostStumpClassifier
        @test ab_model.config.algo == :classification
        @test ab_model.config.type == SoleXplorer.DecisionEnsemble
        @test ab_model.config.rawapply == SoleXplorer.DT.apply_adaboost_stumps
        
        # Test hyperparameters
        @test ab_model.params.n_iter == 10
        @test ab_model.params.feature_importance == :impurity
        
        # Test tuning setup
        @test ab_model.tuning.method.type == latinhypercube
    end
    
    @testset "DecisionTreeRegressorModel" begin
        # Test constructor returns correct type
        dtr_model = SoleXplorer.DecisionTreeRegressorModel()
        @test dtr_model isa SoleXplorer.ModelSetup{SoleXplorer.TypeDTR}
        
        # Test basic configuration
        @test dtr_model.type == SoleXplorer.MLJDecisionTreeInterface.DecisionTreeRegressor
        @test dtr_model.config.algo == :regression
        
        # Test hyperparameters
        @test dtr_model.params.max_depth == -1
        @test dtr_model.params.min_samples_leaf == 5  # Different from classifier
        
        # Test tuning parameters
        @test dtr_model.tuning.params == SoleXplorer.TUNING_PARAMS[:regression]
    end
    
    @testset "RandomForestRegressorModel" begin
        # Test constructor returns correct type
        rfr_model = SoleXplorer.RandomForestRegressorModel()
        @test rfr_model isa SoleXplorer.ModelSetup{SoleXplorer.TypeRFR}
        
        # Test basic configuration
        @test rfr_model.type == SoleXplorer.MLJDecisionTreeInterface.RandomForestRegressor
        @test rfr_model.config.algo == :regression
        
        # Test hyperparameters
        @test rfr_model.params.n_trees == 100  # Different from classifier
        @test rfr_model.params.sampling_fraction == 0.7
        
        # Test tuning parameters
        @test rfr_model.tuning.params == SoleXplorer.TUNING_PARAMS[:regression]
    end
    
    @testset "rawmodel and learn_method" begin
        # This is a more complex test, we'll create mock objects
        
        # Create a decision tree model
        dt_model = SoleXplorer.DecisionTreeClassifierModel()
        
        # Test that rawmodel and learn_method are tuples
        @test dt_model.rawmodel isa Tuple
        @test dt_model.learn_method isa Tuple
        @test length(dt_model.rawmodel) == 2
        @test length(dt_model.learn_method) == 2
        
        # Test that the functions in these tuples are callable
        # (We can't fully test them without actual model objects)
        @test all(f -> f isa Function, dt_model.rawmodel)
        @test all(f -> f isa Function, dt_model.learn_method)
    end
end