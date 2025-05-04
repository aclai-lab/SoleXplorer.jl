using SoleXplorer
using Test
using Random

@testset "Dataset Interface" begin

    @testset "DatasetInfo" begin
        # Test valid constructor
        rng = Xoshiro(11)
        info = SoleXplorer.DatasetInfo(
            :classification,
            :standardize,
            nothing,
            0.7,
            0.15,
            rng,
            false,
            ["feature1", "feature2", "feature3"]
        )
        
        # Test validation works
        @test_throws ArgumentError SoleXplorer.DatasetInfo(:regression, :normalize, nothing, -0.1, 0.2, rng, false, nothing)
        @test_throws ArgumentError SoleXplorer.DatasetInfo(:regression, :normalize, nothing, 0.7, 1.2, rng, false, nothing)
        
        # Test getters
        @test SoleXplorer.get_algo(info) == :classification
        @test SoleXplorer.get_treatment(info) == :standardize
        @test SoleXplorer.get_reducefunc(info) === nothing
        @test SoleXplorer.get_train_ratio(info) == 0.7
        @test SoleXplorer.get_valid_ratio(info) == 0.15
        @test SoleXplorer.get_rng(info) === rng
        @test SoleXplorer.get_resample(info) == false
        @test SoleXplorer.get_vnames(info) == ["feature1", "feature2", "feature3"]
    end
    
    @testset "TT_indexes" begin
        # Test constructor
        train_idx = [1, 3, 5, 7, 9]
        valid_idx = [2, 6, 8]
        test_idx = [4, 10]
        tt = SoleXplorer.TT_indexes(train_idx, valid_idx, test_idx)
        
        # Test getters
        @test SoleXplorer.get_train(tt) == train_idx
        @test SoleXplorer.get_valid(tt) == valid_idx
        @test SoleXplorer.get_test(tt) == test_idx
        
        # Test length
        @test length(tt) == 10
    end
    
    @testset "Dataset - Single Split" begin
        # Generate test data
        rng = Xoshiro(11)
        X = rand(rng, 100, 5)
        y = rand(rng, 100)
        
        # Create dataset info
        info = SoleXplorer.DatasetInfo(
            :regression,
            :standardize,
            nothing,
            0.7,
            0.15,
            rng,
            false,
            ["f1", "f2", "f3", "f4", "f5"]
        )
        
        # Create indices
        train_idx = 1:70
        valid_idx = 71:85
        test_idx = 86:100
        tt = SoleXplorer.TT_indexes(collect(train_idx), collect(valid_idx), collect(test_idx))
        
        # Create dataset
        dataset = Dataset(X, y, tt, info)
        
        # Test getters
        @test get_X(dataset) === X
        @test get_y(dataset) === y
        @test get_tt(dataset) === tt
        @test get_info(dataset) === info
        
        # Test data views
        @test size(get_Xtrain(dataset)) == (70, 5)
        @test size(get_Xvalid(dataset)) == (15, 5)
        @test size(get_Xtest(dataset)) == (15, 5)
        @test length(get_ytrain(dataset)) == 70
        @test length(get_yvalid(dataset)) == 15
        @test length(get_ytest(dataset)) == 15
        
        # Test view content
        @test get_Xtrain(dataset) == X[train_idx, :]
        @test get_ytrain(dataset) == y[train_idx]
    end
    
    @testset "Dataset - Multiple Splits (Resampling)" begin
        # Generate test data
        rng = Xoshiro(11)
        X = rand(rng, 100, 5)
        y = rand(rng, 100)
        
        # Create dataset info with resampling
        info = SoleXplorer.DatasetInfo(
            :regression,
            :standardize,
            nothing,
            0.7,
            0.15,
            rng,
            true,  # Enable resampling
            ["f1", "f2", "f3", "f4", "f5"]
        )
        
        # Create multiple TT_indexes (simulating cross-validation folds)
        tt1 = SoleXplorer.TT_indexes(collect(1:70), collect(71:85), collect(86:100))
        tt2 = SoleXplorer.TT_indexes(collect([1:60..., 81:90...]), collect([61:70..., 91:95...]), collect([71:80..., 96:100...]))
        tt = [tt1, tt2]
        
        # Create dataset
        dataset = Dataset(X, y, tt, info)
        
        # Test data views for multiple splits
        @test length(get_Xtrain(dataset)) == 2
        @test length(get_Xvalid(dataset)) == 2
        @test length(get_Xtest(dataset)) == 2
        @test length(get_ytrain(dataset)) == 2
        @test length(get_yvalid(dataset)) == 2
        @test length(get_ytest(dataset)) == 2
        
        # Test first fold
        @test size(get_Xtrain(dataset)[1]) == (70, 5)
        @test get_ytrain(dataset)[1] == y[tt1.train]
        
        # Test second fold
        @test size(get_Xtrain(dataset)[2]) == (70, 5)
        @test get_ytrain(dataset)[2] == y[tt2.train]
    end
    
    @testset "DatasetInfo show method" begin
        info = SoleXplorer.DatasetInfo(
            :classification,
            :standardize,
            nothing,
            0.7,
            0.15,
            Xoshiro(11),
            false,
            ["feature1", "feature2"]
        )
        
        # Capture the output of show method
        io = IOBuffer()
        show(io, info)
        output = String(take!(io))
        
        # Test that the output contains important information
        @test occursin("DatasetInfo:", output)
        @test occursin("algo:", output)
        @test occursin("classification", output)
        @test occursin("treatment:", output)
        @test occursin("standardize", output)
        @test occursin("train_ratio:", output)
        @test occursin("0.7", output)
        @test occursin("valid_ratio:", output)
        @test occursin("0.15", output)
        @test occursin("resample:", output)
        @test occursin("false", output)
        @test occursin("vnames:", output)
        @test occursin("feature1", output)
        @test occursin("feature2", output)
    end
    
    @testset "TT_indexes show method" begin
        tt = SoleXplorer.TT_indexes([1, 2, 3], [4, 5], [6, 7, 8])
        
        # Capture the output of show method
        io = IOBuffer()
        show(io, tt)
        output = String(take!(io))
        
        # Test that the output contains important information
        @test occursin("TT_indexes", output)
        @test occursin("train=", output)
        @test occursin("[1, 2, 3]", output)
        @test occursin("validation=", output)
        @test occursin("[4, 5]", output)
        @test occursin("test=", output)
        @test occursin("[6, 7, 8]", output)
    end
    
    @testset "Dataset show method" begin
        # Create test data
        X = rand(10, 3)
        y = rand(10)
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
        dataset = Dataset(X, y, tt, info)
        
        # Capture the output of show method
        io = IOBuffer()
        show(io, dataset)
        output = String(take!(io))
        
        # Test that the output contains important information
        @test occursin("Dataset:", output)
        @test occursin("X shape:", output)
        @test occursin("(10, 3)", output)
        @test occursin("y length:", output)
        @test occursin("10", output)
        @test occursin("Train indices:", output)
        @test occursin("7", output)
        @test occursin("Valid indices:", output)
        @test occursin("2", output)
        @test occursin("Test indices:", output)
        @test occursin("1", output)
        
        # Test with multiple splits
        tt_multi = [
            SoleXplorer.TT_indexes(1:5, 6:8, 9:10),
            SoleXplorer.TT_indexes(1:6, 7:8, 9:10)
        ]
        info_resample = SoleXplorer.DatasetInfo(
            :classification,
            :standardize,
            nothing,
            0.7,
            0.1,
            Xoshiro(11),
            true,
            ["f1", "f2", "f3"]
        )
        dataset_multi = Dataset(X, y, tt_multi, info_resample)
        
        # Capture the output
        io = IOBuffer()
        show(io, dataset_multi)
        output_multi = String(take!(io))
        
        # Test output for multiple splits
        @test occursin("Dataset:", output_multi)
        @test occursin("Train/Valid/Test:", output_multi)
        @test occursin("2 folds", output_multi)
    end
end