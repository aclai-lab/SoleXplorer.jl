using Test
using SoleXplorer
using DataFrames
using StatsBase: sample
# using CategoricalArrays
# using Random
# using Statistics, StatsBase

@testset "prepare_dataset.jl" begin
    
    @testset "check utility: check_dataframe_type" begin
        df_valid = DataFrame(a = [1.0, 2.0], b = [3, 4])
        df_invalid = DataFrame(a = ["a", "b"], b = [1, 2])
        
        @test SoleXplorer.check_dataset_type(df_valid) == true
        @test SoleXplorer.check_dataset_type(df_invalid) == false
        @test SoleXplorer.check_dataset_type(Matrix(df_valid)) == true
        @test SoleXplorer.check_dataset_type(Matrix(df_invalid)) == false
    end

    @testset "check utility: hasnans" begin
        df = DataFrame(a = [1.0, 2.0], b = [3, 4])
        df_hasnans = DataFrame(a = [1.0, NaN], b = [3, 4])
        
        @test SoleXplorer.hasnans(df) == false
        @test SoleXplorer.hasnans(df_hasnans) == true
        @test SoleXplorer.hasnans(Matrix(df)) == false
        @test SoleXplorer.hasnans(Matrix(df_hasnans)) == true
    end

    X, y = load_arff_dataset("NATOPS")
    num_cols_to_sample, num_rows_to_sample, rng = 10, 50, Xoshiro(11)
    chosen_cols = sample(rng, 1:size(X, 2), num_cols_to_sample; replace=false)
    chosen_rows = sample(rng, 1:size(X, 1), num_rows_to_sample; replace=false)
    X = X[chosen_rows, chosen_cols]
    y = y[chosen_rows]

    @testset "prepare_dataset check output" begin
        # # Test numeric dataframe
        # X = DataFrame(x1 = [1.0, 2.0, 3.0], x2 = [4.0, 5.0, 6.0])
        # y = [1, 0, 1]
        
        ds = prepare_dataset(X, y; model=(type=:decisiontree,), preprocess=(;rng))

        @test ds isa SoleXplorer.Dataset
        @test size(ds.X) == (3, 2)
        @test all(eltype.(eachcol(ds.X)) .<: Float64)
        @test size(ds.y) == (3,)
        @test ds.y isa CategoricalArray
        @test ds.tt.train isa Vector{Int}
        @test ds.tt.test isa Vector{Int}
        @test ds.info isa SoleXplorer.DatasetInfo
        
        # Test vector-valued dataframe
        X_vec = DataFrame(
            x1 = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            x2 = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
        )
        
        ds_vec = prepare_dataset(
            X_vec, 
            y,
            treatment=:reducesize,
            algo=:classification,
            winparams=(type=SoleXplorer.adaptivewindow, nwindows=2)
        )

        @test ds isa SoleXplorer.Dataset
        @test size(ds_vec.X) == (3, 2)
        @test all(eltype.(eachcol(ds_vec.X)) .<: AbstractVector{<:Number})
        @test ds.tt.train isa Vector{Int}
        @test ds.tt.test isa Vector{Int}
        @test ds.info isa SoleXplorer.DatasetInfo
    end

    @testset "prepare_dataset multidispach" begin
        X = DataFrame(x1 = [1.0, 2.0, 3.0], x2 = [4.0, 5.0, 6.0])
        y = [1.5, 2.5, 3.5]
        model = SoleXplorer.DecisionTreeRegressorModel()

        @test_throws ArgumentError ds = prepare_dataset(X, :x2)
        ds = prepare_dataset(X, :x2, algo=:regression)
        @test size(ds.X, 2) == 1
        @test !in(:x2, names(ds.X))

        @test_throws ArgumentError ds = prepare_dataset(X, y)
        @test_throws ArgumentError ds = prepare_dataset(X, y, algo=:classification)
        @test_nowarn ds = prepare_dataset(X, y, algo=:regression)
        @test_nowarn ds = prepare_dataset(X, y, model)
    end

    @testset "prepare_dataset error handling" begin
        # Invalid parameter
        X = DataFrame(x1 = [1.0, 2.0], x2 = [4.0, 5.0])
        y = [1, 0]

        @test_throws MethodError prepare_dataset(X, y, invalid=:invalid)

        # DataFrame must contain only numeric values
        X = DataFrame(a = ["a", "b"], b = [1, 2])
        y = [1, 0]

        @test_throws ArgumentError prepare_dataset(X, y)

        # Number of rows in DataFrame must match length of class labels
        X = DataFrame(x1 = [1.0, 2.0], x2 = [4.0, 5.0])
        y = [1, 0, 1]
        
        @test_throws ArgumentError prepare_dataset(X, y)
        
        # Regression requires a numeric target variable
        X = DataFrame(x1 = [1.0, 2.0], x2 = [4.0, 5.0])
        y = ["a", "b"]

        @test_throws ArgumentError prepare_dataset(X, y, algo=:regression)

        # Algorithms supported, :regression and :classification
        X = DataFrame(x1 = [1.0, 2.0], x2 = [4.0, 5.0])
        y = [1, 0]

        @test_throws ArgumentError prepare_dataset(X, y, algo=:invalid)

        # Column type not yet supported
        X = DataFrame(a = [rand(2,2) for _ in 1:3], b = [rand(2,2) for _ in 1:3])
        y = [1, 0, 1]
        
        @test_throws ArgumentError prepare_dataset(X, y)

        # winparams must contain a type, movingwindow, wholewindow, splitwindow or adaptivewindow
        X_vec = DataFrame(
            x1 = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            x2 = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
        )
        y = ["a", "b", "c"]

        @test_throws ArgumentError prepare_dataset(X_vec, y, winparams=(nwindows=20,))
        @test_throws ArgumentError prepare_dataset(X_vec, y, winparams=(type=:invalid,))

        # Treatment must be one of: $AVAIL_TREATMENTS
        @test_throws ArgumentError prepare_dataset(X_vec, y, treatment=:invalid)
    end

    @testset "prepare_dataset stratified sampling" begin
        X = DataFrame(x1 = collect(1.0:10.0), x2 = collect(11.0:20.0))
        y = repeat([0, 1], 5)
        nfolds = 5
        
        ds = prepare_dataset(
            X, 
            y,
            stratified=true,
            nfolds=nfolds
        )
        @test length(ds.tt) == nfolds

        # Additional tests for train/test splits
        @test all(size.(ds.Xtrain, 2) .== size(X, 2))
        @test all(size.(ds.Xtest, 2) .== size(X, 2))
        @test size(ds.Xtrain, 1) + size(ds.Xtest, 1) == size(X, 1)
        @test length(ds.ytrain) + length(ds.ytest) == length(y)
        @test eltype(ds.ytrain) == eltype(ds.ytest)
        @test length(ds.Xtrain) == nfolds
        @test length(ds.Xtest) == nfolds
        @test length(ds.ytrain) == nfolds
        @test length(ds.ytest) == nfolds
    end

    @testset "prepare_dataset usage examples" begin
        X, y = Sole.load_arff_dataset("NATOPS")
        train_seed = 11
        rng = Random.Xoshiro(train_seed)
        Random.seed!(train_seed)

        # downsize dataset
        num_cols_to_sample = 10
        num_rows_to_sample = 50
        chosen_cols = StatsBase.sample(rng, 1:size(X, 2), num_cols_to_sample; replace=false)
        chosen_rows = StatsBase.sample(rng, 1:size(X, 1), num_rows_to_sample; replace=false)

        X = X[chosen_rows, chosen_cols]
        y = y[chosen_rows]

        ds = prepare_dataset(
            X, y, 
            features=[mean, std], 
            shuffle=false, 
            winparams=(type=splitwindow, nwindows=10)
        )

        # Test parameters are correctly set
        @test ds.info.features == [mean, std]
        @test ds.info.shuffle == false
        @test ds.info.winparams.type == splitwindow
        @test ds.info.winparams.nwindows == 10
        # Test output structure 
        @test ds isa SoleXplorer.Dataset
        @test size(ds.X, 1) == length(y)
        @test length(ds.y) == length(y)

        ds = prepare_dataset(
            X, y,
            # model.config
            algo=:classification,
            treatment=:aggregate,
            features=[mean, std],
            # model.preprocess
            train_ratio=0.8,
            shuffle=true,
            stratified=true,
            nfolds=6,
            rng=rng,
            # model.winparams
            winparams=(type=adaptivewindow, nwindows=10),
            vnames=names(X),
        )

        model = SoleXplorer.DecisionTreeClassifierModel()

        @test_nowarn ds_class = prepare_dataset(X, y, model)

        # Test each parameter was set correctly
        @test ds.info.algo == :classification
        @test ds.info.treatment == :aggregate
        @test ds.info.features == [mean, std]
        @test ds.info.train_ratio == 0.8
        @test ds.info.shuffle == true
        @test ds.info.stratified == true
        @test ds.info.nfolds == 6
        @test ds.info.rng == rng
        @test ds.info.winparams.type == adaptivewindow
        @test ds.info.winparams.nwindows == 10
        @test ds.info.vnames == Symbol.(names(X))
        
        # Test output structure
        @test ds isa SoleXplorer.Dataset
        @test size(ds.X, 1) == length(y)
        @test length(ds.y) == length(y)
        @test length(ds.tt) == 6  # nfolds
    end

    @testset "dataset with train/validation/test" begin
        X, y = Sole.load_arff_dataset("NATOPS")
        train_seed = 11
        rng = Random.Xoshiro(train_seed)
        Random.seed!(train_seed)

        # downsize dataset
        num_cols_to_sample = 10
        num_rows_to_sample = 50
        chosen_cols = StatsBase.sample(rng, 1:size(X, 2), num_cols_to_sample; replace=false)
        chosen_rows = StatsBase.sample(rng, 1:size(X, 1), num_rows_to_sample; replace=false)

        X = X[chosen_rows, chosen_cols]
        y = y[chosen_rows]

        ds = prepare_dataset(
            X, y, 
            valid_ratio=0.8,
            features=[mean, std], 
            shuffle=false, 
            winparams=(type=splitwindow, nwindows=10)
        )

        @test !isempty(ds.Xvalid)
        @test !isempty(ds.yvalid)

        ds = prepare_dataset(
            X, y,
            # model.preprocess
            train_ratio=0.8,
            valid_ratio=0.8,
            shuffle=true,
            stratified=true,
            nfolds=6,
            rng=rng,
        )

        @test !isempty(ds.Xvalid)
        @test !isempty(ds.yvalid)

        ds = prepare_dataset(
            X, y, 
            # valid_ratio=0.8,
            features=[mean, std], 
            shuffle=false, 
            winparams=(type=splitwindow, nwindows=10)
        )

        @test isempty(ds.Xvalid)
        @test isempty(ds.yvalid)

        ds = prepare_dataset(
            X, y,
            # model.preprocess
            train_ratio=0.8,
            # valid_ratio=0.8,
            shuffle=true,
            stratified=true,
            nfolds=6,
            rng=rng,
        )

        @test all(isempty.(ds.Xvalid))
        @test all(isempty.(ds.yvalid))
    end
end

