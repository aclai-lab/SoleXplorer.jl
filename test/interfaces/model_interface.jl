using Test
using SoleXplorer
using MLJ, MLJDecisionTreeInterface, Random
using SoleModels
using DataFrames

@testset "Model Interface" begin
    
    @testset "Type Aliases and Basic Types" begin
        # Test type aliases
        @test SoleXplorer.Cat_Value == Union{AbstractString, Symbol, MLJ.CategoricalValue}
        @test SoleXplorer.Reg_Value == Number
        @test SoleXplorer.Y_Value == Union{SoleXplorer.Cat_Value, SoleXplorer.Reg_Value}
        
        # Test model type groupings
        @test SoleXplorer.TypeTreeForestC == Union{SoleXplorer.TypeDTC, SoleXplorer.TypeRFC, SoleXplorer.TypeABC, SoleXplorer.TypeMDT, SoleXplorer.TypeXGC}
        @test SoleXplorer.TypeTreeForestR == Union{SoleXplorer.TypeDTR, SoleXplorer.TypeRFR}
        @test SoleXplorer.TypeModalForest == Union{SoleXplorer.TypeMRF, SoleXplorer.TypeMAB}
    end
    
    @testset "ModelSetup Construction and Accessors" begin
        # Create a basic model setup
        dt_setup = SoleXplorer.ModelSetup{SoleXplorer.TypeDTC}(
            SoleXplorer.DecisionTreeClassifierModel,
            (treatment=:none, algo=:classification),
            (max_depth=5,),
            nothing,  # No feature extraction
            nothing,  # No resampling
            SoleXplorer.WinParams(movingwindow, (window_size = 2048, window_step = 1024)),
            SoleXplorer.DecisionTreeClassifierModel,
            MLJ.fit!,
            false,  # No tuning
            true,   # Extract rules with default params
            (train_ratio=0.8, valid_ratio=0.2, rng=123)
        )
        
        # Test accessors
        @test SoleXplorer.get_config(dt_setup) == (treatment=:none, algo=:classification)
        @test SoleXplorer.get_params(dt_setup) == (max_depth=5,)
        @test SoleXplorer.get_features(dt_setup) === nothing
        @test SoleXplorer.get_winparams(dt_setup) isa SoleXplorer.WinParams
        @test SoleXplorer.get_tuning(dt_setup) === false
        @test SoleXplorer.get_resample(dt_setup) === nothing
        @test SoleXplorer.get_preprocess(dt_setup).train_ratio == 0.8
        @test SoleXplorer.get_treatment(dt_setup) == :none
        @test SoleXplorer.get_algo(dt_setup) == :classification
        
        # Test setters
        SoleXplorer.set_config!(dt_setup, (treatment=:standardize, algo=:classification))
        @test SoleXplorer.get_treatment(dt_setup) == :standardize
        
        SoleXplorer.set_params!(dt_setup, (max_depth=10, min_samples_split=2))
        @test SoleXplorer.get_params(dt_setup).max_depth == 10
        @test SoleXplorer.get_params(dt_setup).min_samples_split == 2
        
        SoleXplorer.set_features!(dt_setup, [maximum, minimum, mean, std])
        @test SoleXplorer.length(SoleXplorer.get_features(dt_setup)) == 4
        @test SoleXplorer.get_features(dt_setup)[1] == maximum
    end
    
    @testset "get_rulesparams" begin
        # Test with RulesParams object
        rule_params = SoleXplorer.RulesParams(:intrees, (prune_rules=true, max_rules=10))
        dt_setup = SoleXplorer.ModelSetup{SoleXplorer.TypeDTC}(
            SoleXplorer.DecisionTreeClassifierModel,
            (treatment=:none, algo=:classification),
            (max_depth=5,),
            nothing, 
            nothing, 
            SoleXplorer.WinParams(movingwindow, (window_size = 2048, window_step = 1024)),
            SoleXplorer.DecisionTreeClassifierModel,
            MLJ.fit!,
            false,  # No tuning
            rule_params, # Use our rule params
            (train_ratio=0.8, valid_ratio=0.2, rng=123)
        )
        
        @test SoleXplorer.get_rulesparams(dt_setup) === rule_params
        
        # Test with boolean
        dt_setup2 = SoleXplorer.ModelSetup{SoleXplorer.TypeDTC}(
            SoleXplorer.DecisionTreeClassifierModel,
            (treatment=:none, algo=:classification),
            (max_depth=5,),
            nothing, 
            nothing, 
            SoleXplorer.WinParams(movingwindow, (window_size = 2048, window_step = 1024)),
            SoleXplorer.DecisionTreeClassifierModel,
            MLJ.fit!,
            false,  # No tuning
            true,   # Use boolean
            (train_ratio=0.8, valid_ratio=0.2, rng=123)
        )
        
        @test SoleXplorer.get_rulesparams(dt_setup2) === true
    end
    
    @testset "get_pfeatures" begin
        # Create a model setup with params containing features
        dt_setup = SoleXplorer.ModelSetup{SoleXplorer.TypeDTC}(
            SoleXplorer.DecisionTreeClassifierModel,
            (treatment=:none, algo=:classification),
            (max_depth=5, features=[:x1, :x2, :x3]),  # Include features in params
            nothing, 
            nothing, 
            SoleXplorer.WinParams(movingwindow, (window_size = 2048, window_step = 1024)),
            SoleXplorer.DecisionTreeClassifierModel,
            MLJ.fit!,
            false,
            true,
            (train_ratio=0.8, valid_ratio=0.2, rng=123)
        )
        
        @test SoleXplorer.get_pfeatures(dt_setup) == [:x1, :x2, :x3]
        
        # Test behavior when features field is missing
        dt_setup2 = SoleXplorer.ModelSetup{SoleXplorer.TypeDTC}(
            SoleXplorer.DecisionTreeClassifierModel,
            (treatment=:none, algo=:classification),
            (max_depth=5,),  # No features field
            nothing, 
            nothing, 
            SoleXplorer.WinParams(movingwindow, (window_size = 2048, window_step = 1024)),
            SoleXplorer.DecisionTreeClassifierModel,
            MLJ.fit!,
            false,
            true,
            (train_ratio=0.8, valid_ratio=0.2, rng=123)
        )
        
        # This should throw a KeyError since there's no features field
        @test_throws ErrorException SoleXplorer.get_pfeatures(dt_setup2)
    end
    
    @testset "When fields are tuples" begin
        # Create model setup with tuple values for rawmodel and learn_method
        dt_setup = SoleXplorer.ModelSetup{SoleXplorer.TypeDTC}(
            SoleXplorer.DecisionTreeClassifierModel,
            (treatment=:none, algo=:classification),
            (max_depth=5,),
            nothing, 
            nothing, 
            SoleXplorer.WinParams(movingwindow, (window_size = 2048, window_step = 1024)),
            (SoleXplorer.DecisionTreeClassifierModel, SoleXplorer.RandomForestClassifierModel),  # Tuple of callables
            (MLJ.fit!, MLJ.predict),  # Tuple of callables
            false,
            true,
            (train_ratio=0.8, valid_ratio=0.2, rng=123)
        )
        
        # Test the rawmodel getters
        @test SoleXplorer.get_rawmodel(dt_setup) === SoleXplorer.DecisionTreeClassifierModel
        @test SoleXplorer.get_resampled_rawmodel(dt_setup) === SoleXplorer.RandomForestClassifierModel
        
        # Test the learn_method getters
        @test SoleXplorer.get_learn_method(dt_setup) === MLJ.fit!
        @test SoleXplorer.get_resampled_learn_method(dt_setup) === MLJ.predict
    end
    
    @testset "When fields are single callables" begin
        # Create model setup with single callable values
        dt_setup = SoleXplorer.ModelSetup{SoleXplorer.TypeDTC}(
            SoleXplorer.DecisionTreeClassifierModel,
            (treatment=:none, algo=:classification),
            (max_depth=5,),
            nothing, 
            nothing, 
            SoleXplorer.WinParams(movingwindow, (window_size = 2048, window_step = 1024)),
            SoleXplorer.DecisionTreeClassifierModel,  # Single callable
            MLJ.fit!,  # Single callable
            false,
            true,
            (train_ratio=0.8, valid_ratio=0.2, rng=123)
        )
        
        # These should throw errors since we're trying to index single callables
        @test_throws MethodError SoleXplorer.get_rawmodel(dt_setup)
        @test_throws MethodError SoleXplorer.get_resampled_rawmodel(dt_setup)
        @test_throws MethodError SoleXplorer.get_learn_method(dt_setup)
        @test_throws MethodError SoleXplorer.get_resampled_learn_method(dt_setup)
    end
    
    @testset "Suggested fix for handling both cases" begin
        # Create alternate getter functions that handle both cases
        safe_get_rawmodel(m::SoleXplorer.ModelSetup) = isa(m.rawmodel, Tuple) ? m.rawmodel[1] : m.rawmodel
        safe_get_resampled_rawmodel(m::SoleXplorer.ModelSetup) = isa(m.rawmodel, Tuple) ? m.rawmodel[2] : nothing
        safe_get_learn_method(m::SoleXplorer.ModelSetup) = isa(m.learn_method, Tuple) ? m.learn_method[1] : m.learn_method
        safe_get_resampled_learn_method(m::SoleXplorer.ModelSetup) = isa(m.learn_method, Tuple) ? m.learn_method[2] : nothing
        
        # Test with tuple values
        dt_setup1 = SoleXplorer.ModelSetup{SoleXplorer.TypeDTC}(
            SoleXplorer.DecisionTreeClassifierModel,
            (treatment=:none, algo=:classification),
            (max_depth=5,),
            nothing, 
            nothing, 
            SoleXplorer.WinParams(movingwindow, (window_size = 2048, window_step = 1024)),
            (SoleXplorer.DecisionTreeClassifierModel, SoleXplorer.RandomForestClassifierModel),
            (MLJ.fit!, MLJ.predict),
            false,
            true,
            (train_ratio=0.8, valid_ratio=0.2, rng=123)
        )
        
        @test safe_get_rawmodel(dt_setup1) === SoleXplorer.DecisionTreeClassifierModel
        @test safe_get_resampled_rawmodel(dt_setup1) === SoleXplorer.RandomForestClassifierModel
        @test safe_get_learn_method(dt_setup1) === MLJ.fit!
        @test safe_get_resampled_learn_method(dt_setup1) === MLJ.predict
        
        # Test with single callable values
        dt_setup2 = SoleXplorer.ModelSetup{SoleXplorer.TypeDTC}(
            SoleXplorer.DecisionTreeClassifierModel,
            (treatment=:none, algo=:classification),
            (max_depth=5,),
            nothing, 
            nothing, 
            SoleXplorer.WinParams(movingwindow, (window_size = 2048, window_step = 1024)),
            SoleXplorer.DecisionTreeClassifierModel,
            MLJ.fit!,
            false,
            true,
            (train_ratio=0.8, valid_ratio=0.2, rng=123)
        )
        
        @test safe_get_rawmodel(dt_setup2) === SoleXplorer.DecisionTreeClassifierModel
        @test safe_get_resampled_rawmodel(dt_setup2) === nothing
        @test safe_get_learn_method(dt_setup2) === MLJ.fit!
        @test safe_get_resampled_learn_method(dt_setup2) === nothing
    end

    @testset "ModelSetup Setter Methods" begin
        # Create a base model setup for testing
        dt_setup = SoleXplorer.ModelSetup{SoleXplorer.TypeDTC}(
            SoleXplorer.DecisionTreeClassifierModel,
            (treatment=:none, algo=:classification),
            (max_depth=5,),
            nothing, 
            nothing, 
            SoleXplorer.WinParams(movingwindow, (window_size = 2048, window_step = 1024)),
            SoleXplorer.DecisionTreeClassifierModel,
            MLJ.fit!,
            false,
            true,
            (train_ratio=0.8, valid_ratio=0.2, rng=123)
        )
        
        @testset "set_winparams!" begin
            new_winparams = SoleXplorer.WinParams(splitwindow, (n_windows = 10,))
            SoleXplorer.set_winparams!(dt_setup, new_winparams)
            @test dt_setup.winparams === new_winparams
            @test SoleXplorer.get_winparams(dt_setup) === new_winparams
        end
        
        @testset "set_tuning!" begin
            # Test with boolean
            SoleXplorer.set_tuning!(dt_setup, true)
            @test dt_setup.tuning === true
            @test SoleXplorer.get_tuning(dt_setup) === true
            
            # Test with TuningParams object
            tuning_params = SoleXplorer.TuningParams(
                SoleXplorer.TuningStrategy(grid, (resolution=10,)),
                (n=20, measure=MLJ.accuracy),
                (SoleXplorer.range(:max_depth, lower=3, upper=10),)
            )
            SoleXplorer.set_tuning!(dt_setup, tuning_params)
            @test dt_setup.tuning === tuning_params
            @test SoleXplorer.get_tuning(dt_setup) === tuning_params
        end
        
        @testset "set_resample!" begin
            # Test with nothing
            SoleXplorer.set_resample!(dt_setup, nothing)
            @test dt_setup.resample === nothing
            @test SoleXplorer.get_resample(dt_setup) === nothing
            
            # Test with Resample object
            cv_params = (nfolds=5, shuffle=true, rng=TaskLocalRNG())
            resample_obj = SoleXplorer.Resample(CV, cv_params)
            SoleXplorer.set_resample!(dt_setup, resample_obj)
            @test dt_setup.resample === resample_obj
            @test SoleXplorer.get_resample(dt_setup) === resample_obj
        end
        
        @testset "set_rulesparams!" begin
            # Test with boolean
            SoleXplorer.set_rulesparams!(dt_setup, false)
            @test dt_setup.rulesparams === false
            @test SoleXplorer.get_rulesparams(dt_setup) === false
            
            # Test with RulesParams object
            rules_params = SoleXplorer.RulesParams(:intrees, (prune_rules=true, max_rules=10))
            SoleXplorer.set_rulesparams!(dt_setup, rules_params)
            @test dt_setup.rulesparams === rules_params
            @test SoleXplorer.get_rulesparams(dt_setup) === rules_params
        end
        
        @testset "set_rawmodel!" begin
            new_model = SoleXplorer.RandomForestClassifierModel
            SoleXplorer.set_rawmodel!(dt_setup, new_model)
            @test dt_setup.rawmodel === new_model
            
            # The getter might fail if it expects a tuple - testing for that
            if isa(dt_setup.rawmodel, Tuple)
                @test SoleXplorer.get_rawmodel(dt_setup) === new_model[1]
            else
                @test_throws MethodError SoleXplorer.get_rawmodel(dt_setup)
                # Alternatively, test a safe getter:
                safe_get_rawmodel(m) = isa(m.rawmodel, Tuple) ? m.rawmodel[1] : m.rawmodel
                @test safe_get_rawmodel(dt_setup) === new_model
            end
        end
        
        @testset "set_learn_method!" begin
            new_method = MLJ.predict
            SoleXplorer.set_learn_method!(dt_setup, new_method)
            @test dt_setup.learn_method === new_method
            
            # The getter might fail if it expects a tuple - testing for that
            if isa(dt_setup.learn_method, Tuple)
                @test SoleXplorer.get_learn_method(dt_setup) === new_method[1]
            else
                @test_throws MethodError SoleXplorer.get_learn_method(dt_setup)
                # Alternatively, test a safe getter:
                safe_get_learn_method(m) = isa(m.learn_method, Tuple) ? m.learn_method[1] : m.learn_method
                @test safe_get_learn_method(dt_setup) === new_method
            end
        end
    end

    @testset "Model Types and Constructor Functions" begin
        # Test that all model types are properly defined
        @test isdefined(SoleXplorer, :TypeDTC)
        @test isdefined(SoleXplorer, :TypeRFC)
        @test isdefined(SoleXplorer, :TypeABC)
        @test isdefined(SoleXplorer, :TypeDTR)
        @test isdefined(SoleXplorer, :TypeRFR)
        @test isdefined(SoleXplorer, :TypeMDT)
        @test isdefined(SoleXplorer, :TypeMRF)
        @test isdefined(SoleXplorer, :TypeMAB)
        @test isdefined(SoleXplorer, :TypeXGC)
        @test isdefined(SoleXplorer, :TypeXGR)
        
        # Test type groupings
        @test SoleXplorer.TypeTreeForestC == Union{SoleXplorer.TypeDTC, SoleXplorer.TypeRFC, SoleXplorer.TypeABC, SoleXplorer.TypeMDT, SoleXplorer.TypeXGC}
        @test SoleXplorer.TypeTreeForestR == Union{SoleXplorer.TypeDTR, SoleXplorer.TypeRFR}
        @test SoleXplorer.TypeModalForest == Union{SoleXplorer.TypeMRF, SoleXplorer.TypeMAB}
        
        # Create a model setup for testing constructor functions
        dt_setup = SoleXplorer.ModelSetup{SoleXplorer.TypeDTC}(
            SoleXplorer.DecisionTreeClassifierModel,
            (treatment=:none, algo=:classification),
            (max_depth=5,),
            nothing, 
            nothing, 
            SoleXplorer.WinParams(movingwindow, (window_size = 2048, window_step = 1024)),
            SoleXplorer.DecisionTreeClassifierModel,
            MLJ.fit!,
            false,
            true,
            (train_ratio=0.8, valid_ratio=0.2, rng=123)
        )
        
        # Test model constructor functions
        @test SoleXplorer.DecisionTreeClassifierModel(dt_setup) === dt_setup
        @test SoleXplorer.RandomForestClassifierModel(dt_setup) === dt_setup
        @test SoleXplorer.AdaBoostClassifierModel(dt_setup) === dt_setup
        @test SoleXplorer.DecisionTreeRegressorModel(dt_setup) === dt_setup
        @test SoleXplorer.RandomForestRegressorModel(dt_setup) === dt_setup
        @test SoleXplorer.ModalDecisionTreeModel(dt_setup) === dt_setup
        @test SoleXplorer.ModalRandomForestModel(dt_setup) === dt_setup
        @test SoleXplorer.ModalAdaBoostModel(dt_setup) === dt_setup
        @test SoleXplorer.XGBoostClassifierModel(dt_setup) === dt_setup
        @test SoleXplorer.XGBoostRegressorModel(dt_setup) === dt_setup
        
        # Test model type constructors directly
        @test SoleXplorer.TypeDTC() isa SoleXplorer.TypeDTC
        @test SoleXplorer.TypeRFC() isa SoleXplorer.TypeRFC
        @test SoleXplorer.TypeDTR() isa SoleXplorer.TypeDTR
        
        # Check the AVAIL_MODELS dictionary
        @test haskey(SoleXplorer.AVAIL_MODELS, :decisiontree_classifier)
        @test haskey(SoleXplorer.AVAIL_MODELS, :randomforest_classifier)
        @test haskey(SoleXplorer.AVAIL_MODELS, :adaboost_classifier)
        @test haskey(SoleXplorer.AVAIL_MODELS, :decisiontree_regressor)
        @test haskey(SoleXplorer.AVAIL_MODELS, :randomforest_regressor)
        @test haskey(SoleXplorer.AVAIL_MODELS, :modaldecisiontree)
        @test haskey(SoleXplorer.AVAIL_MODELS, :modalrandomforest)
        @test haskey(SoleXplorer.AVAIL_MODELS, :modaladaboost)
        @test haskey(SoleXplorer.AVAIL_MODELS, :xgboost_classifier)
        @test haskey(SoleXplorer.AVAIL_MODELS, :xgboost_regressor)
        
        # Test the function references in AVAIL_MODELS
        @test SoleXplorer.AVAIL_MODELS[:decisiontree_classifier] === SoleXplorer.DecisionTreeClassifierModel
        @test SoleXplorer.AVAIL_MODELS[:randomforest_classifier] === SoleXplorer.RandomForestClassifierModel
        @test SoleXplorer.AVAIL_MODELS[:xgboost_regressor] === SoleXplorer.XGBoostRegressorModel
    end
    
    @testset "Default Parameters" begin
        @test SoleXplorer.DEFAULT_MODEL_SETUP == (type=:decisiontree,)
        @test SoleXplorer.DEFAULT_FEATS[1] == maximum
        @test SoleXplorer.DEFAULT_FEATS[3] == MLJ.mean
        @test SoleXplorer.DEFAULT_PREPROC.train_ratio == 0.8
        @test SoleXplorer.DEFAULT_PREPROC.valid_ratio == 1.0
    end
    
    @testset "Available Models" begin
        # Test that all expected models are available
        @test haskey(SoleXplorer.AVAIL_MODELS, :decisiontree_classifier)
        @test haskey(SoleXplorer.AVAIL_MODELS, :randomforest_classifier)
        @test haskey(SoleXplorer.AVAIL_MODELS, :adaboost_classifier)
        @test haskey(SoleXplorer.AVAIL_MODELS, :decisiontree_regressor)
        @test haskey(SoleXplorer.AVAIL_MODELS, :xgboost_classifier)
    end
    
    @testset "Results Structures" begin
        # Test results classes
        class_results = SoleXplorer.ClassResults(0.95)
        @test class_results.accuracy == 0.95
        
        reg_results = SoleXplorer.RegResults(0.85)
        @test reg_results.accuracy == 0.85
        
        # Test results mapping
        @test SoleXplorer.RESULTS[:classification] == SoleXplorer.ClassResults
        @test SoleXplorer.RESULTS[:regression] == SoleXplorer.RegResults
    end
    
    @testset "Modelset Construction" begin
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
            ["f1", "f2", "f3"]
        )
        tt = SoleXplorer.TT_indexes(1:7, 8:9, [10])
        
        # Create dataset
        ds = Dataset(Matrix(X), y, tt, info)
        
        # Create a model setup 
        dt_setup = SoleXplorer.ModelSetup{SoleXplorer.TypeDTC}(
            SoleXplorer.DecisionTreeClassifierModel,
            (treatment=:none, algo=:classification),
            (max_depth=5,),
            [maximum, minimum],
            nothing,  # No resampling
            SoleXplorer.WinParams(movingwindow, (window_size = 2048, window_step = 1024)),
            SoleXplorer.DecisionTreeClassifierModel,
            MLJ.fit!,
            false,  # No tuning
            true,   # Extract rules with default params
            (train_ratio=0.8, valid_ratio=0.2, rng=123)
        )
        
        # Create a modelset
        modelset = Modelset(dt_setup, ds)
        
        # Test fields initially empty
        @test modelset.setup === dt_setup
        @test modelset.ds === ds
        @test modelset.classifier === nothing
        @test modelset.mach === nothing
        @test modelset.model === nothing
        @test modelset.rules === nothing
        @test modelset.results === nothing
        
        # Create a classifier
        classifier = MLJDecisionTreeInterface.DecisionTreeClassifier()
        mach = MLJ.machine(classifier, X, y)
        MLJ.fit!(mach)
        solem = solemodel(MLJ.fitted_params(mach).tree)
        apply!(solem, X, y)

        # Create modelset with filled fields
        filled_modelset = Modelset(
            dt_setup, 
            ds, 
            classifier, 
            mach, 
            solem
        )
        
        @test filled_modelset.setup === dt_setup
        @test filled_modelset.ds === ds
        @test filled_modelset.classifier === classifier
        @test filled_modelset.mach === mach
    end
    
    @testset "ModelSetup show methods" begin
        # Create a ModelSetup object for testing
        dt_setup = SoleXplorer.ModelSetup{SoleXplorer.TypeDTC}(
            SoleXplorer.DecisionTreeClassifierModel,
            (treatment=:none, algo=:classification),
            (max_depth=5,),
            [maximum, minimum, MLJ.mean],  # Some features
            nothing,  # No resampling
            SoleXplorer.WinParams(movingwindow, (window_size = 2048, window_step = 1024)),
            SoleXplorer.DecisionTreeClassifierModel,
            MLJ.fit!,
            false,  # No tuning
            SoleXplorer.RulesParams(:intrees, (prune_rules=true,)),  # Rules extraction
            (train_ratio=0.8, valid_ratio=0.2, rng=123)
        )
        
        # Test the verbose show method (MIME"text/plain")
        output_verbose = sprint(show, MIME("text/plain"), dt_setup)
        
        # Verify expected content in verbose output
        @test occursin("ModelSetup", output_verbose)
        @test occursin("Model type: ", output_verbose)
        @test occursin("Features: 3 features", output_verbose)
        @test occursin("Learning method: ", output_verbose)
        @test occursin("Rules extraction: intrees", output_verbose)
        
        # Test the compact show method
        output_compact = sprint(show, dt_setup)
        
        # Verify expected content in compact output
        @test occursin("ModelSetup(type=", output_compact)
        @test occursin("features=3", output_compact)
        
        # Test with no features
        dt_setup_no_features = SoleXplorer.ModelSetup{SoleXplorer.TypeDTC}(
            SoleXplorer.DecisionTreeClassifierModel,
            (treatment=:none, algo=:classification),
            (max_depth=5,),
            nothing,  # No features
            nothing,
            SoleXplorer.WinParams(movingwindow, (window_size = 2048, window_step = 1024)),
            SoleXplorer.DecisionTreeClassifierModel,
            MLJ.fit!,
            false,
            true,
            (train_ratio=0.8, valid_ratio=0.2, rng=123)
        )
        
        output_no_features = sprint(show, MIME("text/plain"), dt_setup_no_features)
        @test occursin("Features: None", output_no_features)
        
        output_compact_no_features = sprint(show, dt_setup_no_features)
        @test occursin("features=None", output_compact_no_features)
        
        # Test without RulesParams (using boolean)
        dt_setup_no_rules = SoleXplorer.ModelSetup{SoleXplorer.TypeDTC}(
            SoleXplorer.DecisionTreeClassifierModel,
            (treatment=:none, algo=:classification),
            (max_depth=5,),
            nothing,
            nothing,
            SoleXplorer.WinParams(movingwindow, (window_size = 2048, window_step = 1024)),
            SoleXplorer.DecisionTreeClassifierModel,
            MLJ.fit!,
            false,
            false,  # No rules extraction
            (train_ratio=0.8, valid_ratio=0.2, rng=123)
        )
        
        output_no_rules = sprint(show, MIME("text/plain"), dt_setup_no_rules)
        @test !occursin("Rules extraction:", output_no_rules)
    end
    
    @testset "Modelset show method" begin
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
            ["f1", "f2", "f3"]
        )
        tt = SoleXplorer.TT_indexes(1:7, 8:9, [10])
        
        # Create dataset
        ds = Dataset(Matrix(X), y, tt, info)
        
        # Create a model setup
        dt_setup = SoleXplorer.ModelSetup{SoleXplorer.TypeDTC}(
            SoleXplorer.DecisionTreeClassifierModel,
            (treatment=:none, algo=:classification),
            (max_depth=5,),
            nothing,
            nothing,
            SoleXplorer.WinParams(movingwindow, (window_size = 2048, window_step = 1024)),
            SoleXplorer.DecisionTreeClassifierModel,
            MLJ.fit!,
            false,
            true,
            (train_ratio=0.8, valid_ratio=0.2, rng=123)
        )
        
        # Create an empty modelset
        modelset = Modelset(dt_setup, ds)
        
        # Test output of empty modelset
        output = sprint(show, modelset)
        @test occursin("Modelset:", output)
        @test occursin("setup", output)
        @test occursin("classifier = nothing", output) || 
              occursin("classifier =nothing", output)
        @test occursin("rules      = nothing", output) || 
              occursin("rules      =nothing", output)
        
        # Create a modelset with a classifier
        classifier = MLJDecisionTreeInterface.DecisionTreeClassifier()
        mach = MLJ.machine(classifier, X, y)
        MLJ.fit!(mach)
        solem = solemodel(MLJ.fitted_params(mach).tree)
        apply!(solem, X, y)
        
        filled_modelset = Modelset(
            dt_setup, 
            ds, 
            classifier, 
            mach, 
            solem
        )
        
        # Test output of filled modelset
        filled_output = sprint(show, filled_modelset)
        @test occursin("Modelset:", filled_output)
        @test occursin("classifier =", filled_output)
        @test !occursin("classifier = nothing", filled_output)
        
        # Add rules to test rules display
        # if isdefined(Main, :Rule)
        #     rules_params = SoleXplorer.RulesParams(:intrees, (prune_rules=true, max_rules=10))
        #     SoleXplorer.set_rulesparams!(filled_modelset, rules_params)
            
        #     rules_output = sprint(show, filled_modelset)
        #     @test occursin("rules      =", rules_output)
        #     @test !occursin("rules      =nothing", rules_output)
        # end
    end
end