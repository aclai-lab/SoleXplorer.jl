using Test
using SoleXplorer
using DataFrames, CategoricalArrays, Random

@testset "check_dataset_type tests" begin
    df_valid = DataFrame(a = [1.0, 2.0], b = [3, 4])
    df_invalid = DataFrame(a = ["a", "b"], b = [1, 2])
    
    @test SoleXplorer.check_dataset_type(df_valid) == true
    @test SoleXplorer.check_dataset_type(df_invalid) == false
    @test SoleXplorer.check_dataset_type(Matrix(df_valid)) == true
    @test SoleXplorer.check_dataset_type(Matrix(df_invalid)) == false
end

@testset "hasnans tests" begin
    df = DataFrame(a = [1.0, 2.0], b = [3, 4])
    df_hasnans = DataFrame(a = [1.0, NaN], b = [3, 4])
    
    @test SoleXplorer.hasnans(df) == false
    @test SoleXplorer.hasnans(df_hasnans) == true
    @test SoleXplorer.hasnans(Matrix(df)) == false
    @test SoleXplorer.hasnans(Matrix(df_hasnans)) == true
end

@testset "check_row_consistency tests" begin
    @testset "Matrix with consistent row dimensions" begin
        # Case 1: All scalar values
        scalar_matrix = [1 2 3; 4 5 6; 7 8 9]
        @test SoleXplorer.check_row_consistency(scalar_matrix) == true
        
        # Case 2: All arrays with same size in each row
        array_matrix = Matrix{Vector{Float64}}(undef, 3, 3)
        for i in 1:3
            row_size = rand(3:10)  # Random but consistent size for this row
            for j in 1:3
                array_matrix[i, j] = rand(Float64, row_size)
            end
        end
        @test SoleXplorer.check_row_consistency(array_matrix) == true
        
        # Case 3: Mixed scalar and arrays with consistent sizes
        mixed_matrix = Matrix{Any}(undef, 3, 3)
        for i in 1:3
            row_size = rand(3:10)  # Random but consistent size for this row
            for j in 1:3
                if j == 1
                    mixed_matrix[i, j] = rand()  # Scalar
                else
                    mixed_matrix[i, j] = rand(Float64, row_size)  # Array
                end
            end
        end
        @test SoleXplorer.check_row_consistency(mixed_matrix) == true
    end
    
    @testset "Matrix with inconsistent row dimensions" begin
        # Case 1: Arrays with different sizes in same row
        inconsistent_matrix = Matrix{Vector{Float64}}(undef, 2, 3)
        
        # First row has arrays of size 5
        for j in 1:3
            inconsistent_matrix[1, j] = rand(Float64, 5)
        end
        
        # Second row has arrays of different sizes
        inconsistent_matrix[2, 1] = rand(Float64, 3)
        inconsistent_matrix[2, 2] = rand(Float64, 5)  # Same as first row
        inconsistent_matrix[2, 3] = rand(Float64, 7)  # Different size
        
        @test SoleXplorer.check_row_consistency(inconsistent_matrix) == false
        
        # Case 2: Mixed scalar and arrays with inconsistent array sizes
        mixed_inconsistent = Matrix{Any}(undef, 2, 3)
        
        # First row: scalar, array size 4, array size 4
        mixed_inconsistent[1, 1] = 10.5
        mixed_inconsistent[1, 2] = rand(Float64, 4)
        mixed_inconsistent[1, 3] = rand(Float64, 4)
        
        # Second row: array size 3, scalar, array size 5
        mixed_inconsistent[2, 1] = rand(Float64, 3)
        mixed_inconsistent[2, 2] = 20.5
        mixed_inconsistent[2, 3] = rand(Float64, 5)  # Different size from first array
        
        @test SoleXplorer.check_row_consistency(mixed_inconsistent) == false
    end
end

@testset "code_dataset function tests" begin      
    # Test with already categorical data
    df_cat = DataFrame(
        id = 1:3,
        cat = categorical(["low", "medium", "high"])
    )
    result_df_cat = code_dataset(df_cat)
    @test eltype(result_df_cat.cat) <: Integer
    @test result_df_cat.cat == [2, 3, 1]  # "low"=1, "medium"=2, "high"=3
    
    # Test with empty DataFrame
    df_empty = DataFrame()
    result_empty = code_dataset(df_empty)
    @test result_empty === df_empty
    @test isempty(result_empty)
    
    # Test with all-numeric DataFrame
    df_numeric = DataFrame(a = 1:3, b = [1.1, 2.2, 3.3])
    result_numeric = code_dataset(df_numeric)
    @test result_numeric === df_numeric  # Should be unchanged
    @test result_numeric.a == [1, 2, 3]
    @test result_numeric.b == [1.1, 2.2, 3.3]

    # Test with string vector
    vec_str = ["cat", "dog", "cat", "fish", "dog"]
    result_str = code_dataset(vec_str)
    @test result_str == [1, 2, 1, 3, 2]
    
    # Test with symbol vector
    vec_sym = [:apple, :banana, :apple, :cherry]
    result_sym = code_dataset(vec_sym)
    @test result_sym == [1, 2, 1, 3]
    
    # Test with boolean vector
    vec_bool = [true, false, true, true, false]
    result_bool = code_dataset(vec_bool)
    @test result_bool == [1, 0, 1, 1, 0]
    
    # Test with already numeric vector
    vec_num = [1, 2, 3, 4, 5]
    result_num = code_dataset(vec_num)
    @test result_num === vec_num  # Should be unchanged

    # Test with mixed types
    df = DataFrame(
        id = 1:3,
        cat = ["X", "Y", "Z"]
    )
    vec = ["Class1", "Class2", "Class1"]
    
    # Apply conversion
    result_df, result_vec = code_dataset(df, vec)
    
    # Check DataFrame results
    @test result_df === df  # In-place modification
    @test result_df.id == [1, 2, 3]
    @test result_df.cat == [1, 2, 3]
    
    # Check vector results
    @test result_vec == [1, 2, 1]
    
    # Test with already numeric data
    df_num = DataFrame(a = 1:3, b = [1.1, 2.2, 3.3])
    vec_num = [10, 20, 30]
    result_df_num, result_vec_num = code_dataset(df_num, vec_num)
    @test result_df_num === df_num  # Unchanged
    @test result_vec_num === vec_num  # Unchanged
end

@testset "_partition function tests" begin
    # Setup
    rng = Random.MersenneTwister(123)  # Fixed seed for reproducibility
    
    @testset "Basic partitioning (no resample)" begin
        # Create a synthetic dataset
        n = 100
        y = rand(1:3, n)  # Classification target
        
        # Test 1: Standard split with validation set
        train_ratio = 0.8
        valid_ratio = 0.2
        result = SoleXplorer._partition(y, train_ratio, valid_ratio, nothing, rng)
        
        # Type check
        @test result isa SoleXplorer.TT_indexes{Int}
        
        # Check sizes approximately match requested ratios
        n_train = length(result.train)
        n_valid = length(result.valid)
        n_test = length(result.test)
        
        @test n_train + n_valid + n_test == n
        
        # Check for uniqueness - no duplicated indices
        all_indices = vcat(result.train, result.valid, result.test)
        @test length(all_indices) == n
        @test length(unique(all_indices)) == n
        
        # Check index ranges
        @test all(i -> 1 <= i <= n, all_indices)
        
        # Test 2: No validation set (valid_ratio = 1.0)
        result_no_valid = SoleXplorer._partition(y, train_ratio, 1.0, nothing, rng)
        
        @test result_no_valid isa SoleXplorer.TT_indexes{Int}
        @test isempty(result_no_valid.valid)
        @test length(result_no_valid.train) + length(result_no_valid.test) == n
        @test length(result_no_valid.train) ≈ train_ratio * n rtol=0.1
    end
    
    @testset "Cross-validation partitioning (with resample)" begin
        # Create a synthetic dataset
        n = 100
        y = rand(1:3, n)  # Classification target
        
        # Create a CV resample specification
        cv_resample = SoleXplorer.Resample(
            CV,
            (nfolds = 5, shuffle = true, rng = rng)
        )
        
        # Test 1: CV with validation sets
        train_ratio = 0.8
        valid_ratio = 0.2
        result_cv = SoleXplorer._partition(y, train_ratio, valid_ratio, cv_resample, rng)
        
        # Type check
        @test result_cv isa Vector{SoleXplorer.TT_indexes{Int}}
        @test length(result_cv) == 5  # 5 folds
        
        for (i, fold) in enumerate(result_cv)
            # Check that train + valid + test covers all data
            all_indices = vcat(fold.train, fold.valid, fold.test)
            @test length(all_indices) == n
            @test length(unique(all_indices)) == n
            
            # Check validation set comes from training set
            @test issubset(fold.valid, vcat(fold.train, fold.valid))
            
            # Check sizes - validation should be a fraction of the original train set
            n_train_valid = length(fold.train) + length(fold.valid)
            @test length(fold.valid) ≈ valid_ratio * n_train_valid rtol=0.1
            
            # Check index ranges
            @test all(i -> 1 <= i <= n, all_indices)
        end
        
        # Test 2: CV without validation sets
        result_cv_no_valid = SoleXplorer._partition(y, train_ratio, 1.0, cv_resample, rng)
        
        @test result_cv_no_valid isa Vector{SoleXplorer.TT_indexes{Int}}
        @test length(result_cv_no_valid) == 5  # 5 folds
        
        for fold in result_cv_no_valid
            @test isempty(fold.valid)
            all_indices = vcat(fold.train, fold.test)
            @test length(all_indices) == n
            @test length(unique(all_indices)) == n
        end
    end
    
    @testset "Regression targets" begin
        # Create regression target
        n = 100
        y_reg = randn(n)
        
        # Basic test to ensure it works with regression targets
        train_ratio = 0.7
        valid_ratio = 0.3
        result_reg = SoleXplorer._partition(y_reg, train_ratio, valid_ratio, nothing, rng)
        
        @test result_reg isa SoleXplorer.TT_indexes{Int}
        @test length(result_reg.train) + length(result_reg.valid) + length(result_reg.test) == n
    end
    
    @testset "Deterministic with same RNG" begin
        # Test that with the same RNG seed, we get the same partitioning
        y = rand(1:4, 50)
        
        rng1 = Random.MersenneTwister(42)
        rng2 = Random.MersenneTwister(42)
        
        result1 = SoleXplorer._partition(y, 0.8, 0.2, nothing, rng1)
        result2 = SoleXplorer._partition(y, 0.8, 0.2, nothing, rng2)
        
        @test result1.train == result2.train
        @test result1.valid == result2.valid
        @test result1.test == result2.test
    end
end

@testset "prepare_dataset functions" begin    
    # Numeric features for regression
    df_reg = DataFrame(
        feature1 = randn(rng, 100),
        feature2 = randn(rng, 100),
        feature3 = randn(rng, 100)
    )
    y_reg = 2 .* df_reg.feature1 .+ 3 .* df_reg.feature2 .+ randn(rng, 100)
    
    # Mixed features for classification
    df_class = DataFrame(
        feature1 = randn(rng, 100),
        feature2 = rand(1:5, 100),
        feature3 = repeat(["A", "B", "C", "A", "B", "A", "C", "B", "A", "C"], 10)
    )
    y_class = categorical(repeat(["class1", "class2", "class1", "class3"], 25))
    
    # Time series data
    df_class = DataFrame(
        feature1 = randn(rng, 100),
        feature2 = rand(1:5, 100),
        feature3 = rand(1:3, 100)  # Use numeric values instead of strings
    )
    y_ts = categorical(repeat(["class1", "class2", "class3"], 10))
    
    @testset "Basic prepare_dataset functionality" begin
        # Test 1: Classification with default settings
        modelset = prepare_dataset(df_class[1:40, :], y_class[1:40])
        
        @test modelset isa SoleXplorer.Modelset
        @test modelset.ds.info.algo == :classification
        @test modelset.ds.info.treatment == :aggregate
        @test !isnothing(modelset.ds)
        
        # Test data shapes
        @test size(modelset.ds.X, 1) == 40  # Number of samples
        @test length(modelset.ds.y) == 40
        
        # Test partition indices
        train_idx = modelset.ds.ytrain
        valid_idx = modelset.ds.yvalid
        test_idx = modelset.ds.ytest
        
        @test length(train_idx) + length(valid_idx) + length(test_idx) == 40
        @test !isempty(train_idx)
        @test !isempty(test_idx)
        
        # Test 2: Regression
        modelset_reg = prepare_dataset(df_reg, y_reg, 
            model=(type=:decisiontree,))
        
        @test modelset_reg.ds.info.algo == :regression
        @test length(modelset_reg.ds.ytrain) == 80
        @test eltype(modelset_reg.ds.ytrain) <: AbstractFloat
        
        # Test 3: Using column name instead of vector
        df_with_target = DataFrame(df_class)
        df_with_target.target = y_class
        
        modelset_colname = prepare_dataset(df_with_target, :target)
        @test modelset_colname.ds.info.algo == :classification
        @test length(modelset_colname.ds.ytrain) == 80
    end
    
    @testset "Advanced configurations" begin
        # Test 1: Custom model configuration
        model_config = (
            type = :xgboost,
            params = (
                max_depth = 5,
                eta = 0.1,
                objective = "multi:softmax",
            )
        )
        
        modelset = prepare_dataset(df_class, y_class, model=model_config)
        @test modelset.setup.params.max_depth == 5
        @test modelset.setup.params.eta == 0.1
        
        # Test 2: With resampling (CV)
        cv_config = (
            model = model_config,
            resample = (type = CV, params = (nfolds = 3,)),
            preprocess = (train_ratio = 0.7, valid_ratio = 0.2)
        )
        
        modelset_cv = prepare_dataset(df_class, y_class; 
            model=cv_config.model, 
            resample=cv_config.resample,
            preprocess=cv_config.preprocess
        )
    
        @test modelset_cv.ds.tt isa Vector{SoleXplorer.TT_indexes{Int}}
    end
end
    
@testset "error handling" begin
    # Create a DataFrame with non-numeric data
    df_non_numeric = DataFrame(
        numeric = [1, 2, 3, 4],
        strings = ["a", "b", "c", "d"]  # Non-numeric column
    )
    y_valid = [1, 2, 3, 4]
    # Test direct access to prepare_dataset to trigger the specific validation
    @test_throws ArgumentError SoleXplorer.prepare_dataset(df_non_numeric, y_valid)
        
    # Create mismatched data sizes
    df_valid = DataFrame(a = [1.0, 2.0, 3.0, 4.0], b = [5.0, 6.0, 7.0, 8.0])
    y_short = [1, 2, 3]  # One element short
    @test_throws ArgumentError SoleXplorer.prepare_dataset(df_valid, y_short;)
    
    # Create data with inconsistent dimensions in array elements
    df_inconsistent = DataFrame(
        col1 = [[1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0]],  # Inconsistent array lengths
        col2 = [10, 20, 30]
    )
    y_valid = [1, 2, 3]
    @test_throws ArgumentError SoleXplorer.prepare_dataset(df_inconsistent, y_valid)
end
        
@testset "Custom reduce function" begin
    # Test setting a custom reduce function for time series
    df_ts2 = DataFrame(
        ts1 = [randn(rng, 20) for _ in 1:20],
        ts2 = [randn(rng, 20) for _ in 1:20]
    )
    y_ts2 = categorical(repeat(["class1", "class2"], 10))
    
    # Test with median as reduce function
    modelset_median = prepare_dataset(df_ts2, y_ts2,
        reducefunc = median,
        model = (
            type = :xgboost, 
            params = (objective = "multi:softmax",)
        ),
        win = (type = adaptivewindow, params = (nwindows = 4,))
    )
    
    @test modelset_median.ds.info.reducefunc == median
end

