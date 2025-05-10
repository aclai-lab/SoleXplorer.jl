using Test
using SoleXplorer
using MLJ, DataFrames
using Random
# using OrderedCollections: OrderedDict

@testset "XGBoost Models" begin
    
    @testset "Helper Functions" begin
        # Test get_encoding
        test_classes = MLJ.categorical(["A", "B", "C"])
        encoding = SoleXplorer.get_encoding(test_classes)
        @test encoding isa Dict
        @test encoding[1] == "A"
        @test encoding[2] == "B"
        @test encoding[3] == "C"
        
        # Test get_classlabels
        classlabels = SoleXplorer.get_classlabels(encoding)
        @test classlabels isa Vector{String}
        @test classlabels == ["A", "B", "C"]
        
        # Test makewatchlist (only if XGBoost is available)
        X, y = MLJ.@load_iris
        X = DataFrame(X)

        info = SoleXplorer.DatasetInfo(
            :classification,
            :standardize,
            nothing,
            0.7,
            0.1,
            Xoshiro(11),
            false,
            ["f1", "f2", "f3", "f4"]
        )
        tt = SoleXplorer.TT_indexes(1:7, 8:9, [10])
        
        # Create dataset
        ds = Dataset(Matrix(X), y, tt, info)
        
        watchlist = SoleXplorer.makewatchlist(ds)
        @test watchlist isa SoleXplorer.XGB.OrderedDict
        @test haskey(watchlist, "train")
        @test haskey(watchlist, "eval")
        @test watchlist["train"] isa SoleXplorer.XGB.DMatrix
        @test watchlist["eval"] isa SoleXplorer.XGB.DMatrix
        
        # Test makewatchlist error when no validation data
        tt = SoleXplorer.TT_indexes(1:7, Int64[], 8:10)
        
        # Create dataset
        empty_ds = Dataset(Matrix(X), y, tt, info)

        @test_throws ArgumentError SoleXplorer.makewatchlist(empty_ds)
    end
    
    @testset "XGBoostClassifierModel" begin
        # Test constructor returns correct type
        xgb_model = SoleXplorer.XGBoostClassifierModel()
        @test xgb_model isa SoleXplorer.ModelSetup{SoleXplorer.TypeXGC}
        
        # Test basic configuration
        @test xgb_model.type == SoleXplorer.MLJXGBoostInterface.XGBoostClassifier
        @test xgb_model.config.algo == :classification
        @test xgb_model.config.type == SoleXplorer.DecisionEnsemble
        @test xgb_model.config.treatment == :aggregate
        @test xgb_model.config.rawapply == SoleXplorer.XGBoost.predict

        
        # Test hyperparameters
        @test xgb_model.params.num_round == 10
        @test xgb_model.params.booster == "gbtree"
        @test xgb_model.params.eta == 0.3
        @test xgb_model.params.max_depth == 6
        @test xgb_model.params.objective == "automatic"
        @test xgb_model.params.early_stopping_rounds == 0
        @test xgb_model.params.watchlist === nothing
        @test xgb_model.params.importance_type == "gain"
        
        # Test window parameters
        @test xgb_model.winparams.type == wholewindow
        
        # Test model extraction
        @test xgb_model.rawmodel isa Tuple
        @test length(xgb_model.rawmodel) == 2
        
        # Test learn method
        @test xgb_model.learn_method isa Tuple
        @test length(xgb_model.learn_method) == 2
        
        # Test tuning setup
        @test xgb_model.tuning isa SoleXplorer.TuningParams
        @test xgb_model.tuning.method.type == latinhypercube
        @test xgb_model.tuning.method.params.ntour == 20
        @test length(xgb_model.tuning.ranges) == 2
        
        # Test rules extraction
        @test xgb_model.rulesparams isa SoleXplorer.RulesParams
        @test xgb_model.rulesparams.type == :intrees
        
        # Test features configuration
        @test xgb_model.features == SoleXplorer.DEFAULT_FEATS
    end
    
    # @testset "XGBoostRegressorModel" begin
    #     # Test if the function exists
    #     @test isdefined(SoleXplorer, :XGBoostRegressorModel)
        
    #     # Skip remaining tests if function isn't defined or doesn't exist
    #     if isdefined(SoleXplorer, :XGBoostRegressorModel)
    #         # Test constructor returns correct type
    #         xgb_reg_model = XGBoostRegressorModel()
    #         @test xgb_reg_model isa SoleXplorer.ModelSetup{SoleXplorer.TypeXGR}
            
    #         # Test basic configuration
    #         @test xgb_reg_model.config.algo == :regression
    #         @test xgb_reg_model.config.type == DecisionEnsemble
            
    #         # Test tuning parameters
    #         @test xgb_reg_model.tuning.params == TUNING_PARAMS[:regression]
    #     end
    # end
end