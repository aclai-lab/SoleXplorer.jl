using Test
using SoleXplorer
using SoleXplorer: check_unknown_params, get_function,
                   validate_model, validate_params, validate_features, validate_winparams, 
                   validate_tuning_type, validate_tuning_ranges, validate_tuning,
                   validate_preprocess_params, validate_modelset
                   
using SoleXplorer: movingwindow, wholewindow, splitwindow, adaptivewindow     
using SoleXplorer: DecisionTreeModel              
using Random
using StatsBase
using MLJ, MLJParticleSwarmOptimization, MLJDecisionTreeInterface

model = DecisionTreeModel()

defaults = (;
    max_depth              = -1,
    min_samples_leaf       = 1, 
    min_samples_split      = 2, 
    min_purity_increase    = 0.0, 
    n_subfeatures          = 0,
    post_prune             = false,
    merge_purity_threshold = 1.0,
    display_depth          = 5,
    feature_importance     = :impurity,
    rng                    = Random.TaskLocalRNG()
)

avail_functions = (movingwindow, wholewindow, splitwindow, adaptivewindow)

@testset "validate_modelset.jl" begin
    @testset "check_unknown_params" begin

        @test check_unknown_params(nothing, defaults, "test") === nothing
        @test check_unknown_params((n_subfeatures=1,), defaults, "test") === nothing

        params = (min_samples_leaf = 10, feature_importance = :split)
        @test check_unknown_params(params, defaults, "test") === nothing

        params = (invalid_param=1,)
        @test_throws ArgumentError check_unknown_params(params, defaults, "test")
    end

    @testset "get_function" begin
        @test get_function(nothing, avail_functions) === nothing

        func = (type=wholewindow,)
        @test get_function(func, avail_functions) === wholewindow

        nofunc = (a="something", b=2)
        @test get_function(nofunc, avail_functions) === nothing

        func = (t1=wholewindow, t2=movingwindow)
        @test_throws ArgumentError get_function(func, avail_functions)
    end

    @testset "validate_model" begin
        model = :decisiontree
        @test validate_model(model) isa SoleXplorer.SymbolicModelSet

        model = :invalid
        @test_throws ArgumentError validate_model(model)

    end

    @testset "validate_params" begin
        globals = (n_subfeatures=8, feature_importance=:split)
        users = (min_purity_increase=2.3, n_subfeatures=10, post_prune=true)

        result = validate_params(defaults, globals, users)
        @test result.min_purity_increase == 2.3
        @test result.n_subfeatures == 10
        @test result.post_prune == true
        @test result.feature_importance == :split

        result = validate_params(defaults, globals, nothing)
        @test result.n_subfeatures == 8

        @test validate_params(defaults, nothing, nothing) == defaults
    end

    @testset "validate_features" begin
        deffeats = [maximum, minimum, mean, std]
        globfeats = [cov, mode_5]
        usrfeats = [minimum, mean, cov]
        
        @test validate_features(deffeats, nothing, nothing) == deffeats
        @test validate_features(deffeats, globfeats, nothing) == globfeats
        @test validate_features(deffeats, globfeats, usrfeats) == usrfeats
        @test_throws ArgumentError validate_features(deffeats, ["not_a_function"], nothing)
    end

    @testset "validate_winparams" begin
        defwin = (type=wholewindow,)
        globwin = (type=adaptivewindow, nwindows=20)
        usrwin = (type=movingwindow, window_size=300)

        @test validate_winparams(defwin, nothing, nothing, :aggregate) == defwin
        @test validate_winparams(defwin, globwin, nothing, :reducesize) == (type=adaptivewindow, nwindows=20, relative_overlap=0.5,)
        @test validate_winparams(defwin, globwin, usrwin, :aggregate) == (type=movingwindow, window_size=300, window_step=512,)

        invalid = (type=adaptivewindow, nwindows=2)
        @test_throws ArgumentError validate_winparams(defwin, invalid, nothing, :reducesize)
    end

    @testset "validate_tuning_type" begin
        deftune = (type=latinhypercube, ntour=35)
        globtune = (type=grid,)
        usrtune = (type=particleswarm, n_particles=5, w=1.7, c1=2.5,)
        
        result = validate_tuning_type(deftune, nothing, nothing)
        @test result isa LatinHypercube
        @test result.ntour == 35

        result = validate_tuning_type(deftune, globtune, nothing)
        @test result isa grid

        result = validate_tuning_type(deftune, globtune, usrtune)
        @test result isa ParticleSwarm
        @test result.n_particles == 5
        @test result.w == 1.7
        @test result.c1 == 2.5
    end

    @testset "validate_tuning_ranges" begin
        defrange = [
            model -> MLJ.range(model, :merge_purity_threshold, lower=0, upper=1),
            model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
        ]
        globrange = [model -> MLJ.range(model, :merge_purity_threshold, lower=0, upper=1)]
        usrrange = [model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])]
        
        @test validate_tuning_ranges(defrange, nothing, nothing) == defrange
        @test validate_tuning_ranges(defrange, globrange, nothing) == globrange
        @test validate_tuning_ranges(defrange, globrange, usrrange) == usrrange
        @test_throws ArgumentError validate_tuning_ranges(defrange, ["not_a_range"], nothing)
    end

    @testset "validate_tuning" begin
        deftune = (
            tuning        = false,
            method        = (type = latinhypercube, ntour = 20),
            params        = SoleXplorer.TUNING_PARAMS,
            ranges        = [
                model -> MLJ.range(model, :merge_purity_threshold, lower=0, upper=1),
                model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
            ]
        )
        
        @test validate_tuning(deftune, nothing, nothing) == (tuning=false, method=nothing, params=NamedTuple(), ranges=nothing)

        globtune = true
        result = validate_tuning(deftune, globtune, nothing)
        @test result isa NamedTuple
        @test result.method isa LatinHypercube

        usrtune = true
        result = validate_tuning(deftune, nothing, usrtune)
        @test result isa NamedTuple
        @test result.method isa LatinHypercube

        globtune=(
            method=(type=grid, resolution=20,), 
            params=(repeats=11,), 
            ranges=[SoleXplorer.range(:feature_importance; values=[:impurity, :split])]
        )
        usrtune=(
            method=(resolution=35,), 
            params=(repeats=11,), 
            ranges=[SoleXplorer.range(:feature_importance; values=[:impurity, :split])]
        )
        result = validate_tuning(deftune, globtune, usrtune)
        @test result isa NamedTuple
        @test result.method isa grid
        @test result.method.resolution == 35
        @test result.params.repeats == 11
    end

    @testset "validate_preprocess_params" begin
        defpreprocs = (train_ratio = 0.8, shuffle = false, stratified = false, nfolds = 6, rng = TaskLocalRNG(),)
        usrpreprocs = (train_ratio = 0.5,)

        @test validate_preprocess_params(defpreprocs, nothing) === defpreprocs

        result = validate_preprocess_params(defpreprocs, usrpreprocs)
        @test result.train_ratio == 0.5
    end

    @testset "validate_modelset" begin
        model_spec = [(
            type=:decisiontree,
            params=(max_depth=3, min_samples_leaf=14),
            winparams=(type=movingwindow, window_size=12),
            features=[minimum, mean, cov, mode_5]
        )]
        globals = (
            params=(min_samples_split=17,),
            winparams=(type=adaptivewindow,),
            features=[std]
        )
        preprocess = (
                train_ratio = 0.5,
                shuffle     = false,
        )

        result = validate_modelset(model_spec, globals, nothing)
        @test length(result) == 1
        @test result[1] isa SoleXplorer.SymbolicModelSet
        @test result[1].type === DecisionTreeClassifier
        @test result[1].params.min_samples_split == 17

        result = validate_modelset(model_spec, globals, preprocess)
        @test length(result) == 1
        @test result[1] isa SoleXplorer.SymbolicModelSet
        @test result[1].type === DecisionTreeClassifier
        @test result[1].preprocess.train_ratio == 0.5
        
        @test_throws ArgumentError validate_modelset([(params=(a=1,),)], nothing)
    end
end



