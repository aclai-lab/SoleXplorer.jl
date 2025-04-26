using Test
using Random
using SoleXplorer

X, y = load_arff_dataset("NATOPS")

# downsize dataset
# it is important to downsize the dataset to avoid long running times
# and to avoid memory issues
using StatsBase: sample
num_cols_to_sample, num_rows_to_sample, rng = 10, 50, Xoshiro(11)
chosen_cols = sample(rng, 1:size(X, 2), num_cols_to_sample; replace=false)
chosen_rows = sample(rng, 1:size(X, 1), num_rows_to_sample; replace=false)
X = X[chosen_rows, chosen_cols]
y = y[chosen_rows]

model = SoleXplorer.DEFAULT_MODEL_SETUP
resample = nothing
win = nothing
features = nothing
tuning = false
extract_rules = false
preprocess = nothing
reducefunc = nothing

SoleXplorer.validate_modelset(model, eltype(y); resample, win, features, tuning, extract_rules, preprocess, reducefunc)

@btime SoleXplorer.validate_modelset(model, eltype(y); resample, win, features, tuning, extract_rules, preprocess, reducefunc)
# 3.023 μs (14 allocations: 696 bytes)



@testset "check_params tests" begin
    # Test case 1: params is nothing
    @test begin
        SoleXplorer.check_params(nothing, (:a, :b, :c))
        true  # If we reach here, no exception was thrown
    end

    # Test case 2: All keys in params are in allowed_keys
    @test begin
        SoleXplorer.check_params((a=1, b=2), (:a, :b, :c))
        true  # If we reach here, no exception was thrown
    end
    
    # Test case 3: Empty params NamedTuple
    @test begin
        SoleXplorer.check_params(NamedTuple(), (:a, :b, :c))
        true  # If we reach here, no exception was thrown
    end

    # Test case 4: Some keys in params are not in allowed_keys
    @test_throws ArgumentError SoleXplorer.check_params((a=1, d=4), (:a, :b, :c))
    
    # Test case 5: No keys in params are in allowed_keys
    @test_throws ArgumentError SoleXplorer.check_params((d=4, e=5), (:a, :b, :c))
    
    # Test case 6: Empty allowed_keys
    @test_throws ArgumentError SoleXplorer.check_params((a=1, b=2), ())
    
    # Test case 7: Test error message content
    @test try
        SoleXplorer.check_params((a=1, d=4, e=5), (:a, :b, :c))
        false
    catch e
        isa(e, ArgumentError) && occursin("Unknown fields: [:d, :e]", e.msg)
    end
end

@testset "filter_params tests" begin
    # Test case 1: Input is nothing
    @test SoleXplorer.filter_params(nothing) == NamedTuple()
    
    # Test case 2: Input is a non-empty NamedTuple
    test_tuple = (a=1, b="test", c=3.14)
    @test SoleXplorer.filter_params(test_tuple) === test_tuple
    
    # Test case 3: Input is an empty NamedTuple
    empty_tuple = NamedTuple()
    @test SoleXplorer.filter_params(empty_tuple) === empty_tuple
    
    # Test case 4: Identity property - calling it twice should be the same as calling it once
    @test SoleXplorer.filter_params(SoleXplorer.filter_params(nothing)) == SoleXplorer.filter_params(nothing)
    @test SoleXplorer.filter_params(SoleXplorer.filter_params(test_tuple)) === test_tuple
end

@testset "get_type tests" begin
    # Define available types for testing
    avail_types = (:type1, :type2, :type3)
    
    # Test case 1: params is nothing
    @test SoleXplorer.get_type(nothing, avail_types) === nothing
    
    # Test case 2: params doesn't have a :type field
    @test SoleXplorer.get_type((a=1, b=2), avail_types) === nothing
    
    # Test case 3: params has a :type field that's nothing
    @test SoleXplorer.get_type((type=nothing, a=1), avail_types) === nothing
    
    # Test case 4: params has a :type field that's not in avail_types
    @test_throws ArgumentError SoleXplorer.get_type((type=:unknown_type, a=1), avail_types)
    
    # Test case 5: params has a :type field that's in avail_types
    @test SoleXplorer.get_type((type=:type1, a=1), avail_types) === :type1
    @test SoleXplorer.get_type((;type=:type2), avail_types) === :type2
    
    # Test case 6: Empty avail_types
    @test_throws ArgumentError SoleXplorer.get_type((;type=:type1), ())
    
    # Test case 7: Test error message content
    @test try
        SoleXplorer.get_type((;type=:invalid), avail_types)
        false
    catch e
        @show e.msg
        isa(e, ArgumentError) && occursin("Type :invalid not found in available types", e.msg)
    end
end

@testset "check_user_params tests" begin
    # Setup mock data
    mock_defaults = Dict(
        :type1 => (a=1, b=2, c=3),
        :type2 => (x="test", y=42.0),
        :empty_type => NamedTuple()
    )
    
    # Test case 1: users is nothing
    @test SoleXplorer.check_user_params(nothing, mock_defaults) == NamedTuple()
    
    # Test case 2: users has no :params field
    @test SoleXplorer.check_user_params((type=:type1,), mock_defaults) == NamedTuple()
    
    # Test case 3: users has a :params field with valid parameters
    @test SoleXplorer.check_user_params(
        (type=:type1, params=(a=10, b=20)), 
        mock_defaults
    ) == (a=10, b=20)
    
    # Test case 4: users has a :params field with invalid parameters (should throw ArgumentError)
    @test_throws ArgumentError SoleXplorer.check_user_params(
        (type=:type1, params=(a=10, d=40)), 
        mock_defaults
    )
    
    # Test case 5: users has a :type field that doesn't exist in default_params
    @test_throws KeyError SoleXplorer.check_user_params(
        (type=:nonexistent, params=(a=10,)), 
        mock_defaults
    )
    
    # Test case 6: Empty params
    @test SoleXplorer.check_user_params(
        (type=:type1, params=NamedTuple()), 
        mock_defaults
    ) == NamedTuple()
    
    # Test case 7: Type with empty allowed params
    @test SoleXplorer.check_user_params(
        (type=:empty_type, params=NamedTuple()), 
        mock_defaults
    ) == NamedTuple()
    
    # Test case 8: When default_params is an empty Dict
    @test SoleXplorer.check_user_params(
        nothing, 
        Dict()
    ) == NamedTuple()
    
    # Test case 9: Test correct conversion to NamedTuple
    result = SoleXplorer.check_user_params(
        (type=:type1, params=(a=10, c=30)), 
        mock_defaults
    )
    @test result isa NamedTuple
    @test result == (a=10, c=30)
    @test result.a == 10 && result.c == 30
end

@testset "merge_params tests" begin
    # Setup test data
    defaults_with_rng = (a=1, b="test", rng=nothing)
    defaults_without_rng = (a=1, b="test")
    user_params = (c=3.14, d="user")
    test_rng = Xoshiro(11)
    
    # Test case 1: Both users and rng are nothing
    result1 = SoleXplorer.merge_params(defaults_without_rng, nothing)
    @test result1 == defaults_without_rng
    
    # Test case 2: users is a NamedTuple, rng is nothing
    result2 = SoleXplorer.merge_params(defaults_without_rng, user_params)
    @test result2 == (a=1, b="test", c=3.14, d="user")
    @test result2.a == 1 && result2.c == 3.14
    
    # Test case 3: users is nothing, rng is provided, defaults has :rng key
    result3 = SoleXplorer.merge_params(defaults_with_rng, nothing, test_rng)
    @test result3 == (a=1, b="test", rng=test_rng)
    @test result3.rng === test_rng
    
    # Test case 4: users is nothing, rng is provided, defaults has no :rng key
    result4 = SoleXplorer.merge_params(defaults_without_rng, nothing, test_rng)
    @test result4 == defaults_without_rng
    @test !haskey(result4, :rng)
    
    # Test case 5: Both users and rng are provided, defaults has :rng key
    result5 = SoleXplorer.merge_params(defaults_with_rng, user_params, test_rng)
    @test result5 == (a=1, b="test", rng=test_rng, c=3.14, d="user")
    @test result5.rng === test_rng
    
    # Test case 6: Both users and rng are provided, defaults has no :rng key
    result6 = SoleXplorer.merge_params(defaults_without_rng, user_params, test_rng)
    @test result6 == (a=1, b="test", c=3.14, d="user")
    @test !haskey(result6, :rng)
    
    # Test case 7: Override behavior - user params override defaults
    result7 = SoleXplorer.merge_params(defaults_with_rng, (a=100, e=500), test_rng)
    @test result7 == (a=100, b="test", rng=test_rng, e=500)
    @test result7.a == 100 && result7.e == 500
    
    # Test case 8: Empty NamedTuples
    result8 = SoleXplorer.merge_params(NamedTuple(), NamedTuple())
    @test result8 == NamedTuple()
    @test isempty(keys(result8))
end






@testset "validate_model" begin
    @test_nowarn train_test(X, y; model=(type=:decisiontree_classifier,))

    y_class = y
    auto_algo = train_test(X, y_class; model=(type=:decisiontree,))
    @test auto_algo.setup.config.algo == :classification
    y_reg = rand(length(y))
    auto_algo = train_test(X, y_reg; model=(type=:decisiontree,))
    @test auto_algo.setup.config.algo == :regression

    @test_throws ArgumentError train_test(X, y; model=(missing_type=:decisiontree_classifier,))
    @test_throws ArgumentError train_test(X, y; model=(type=:invalid_classifier,))
end

@testset "validate_params" begin
    @test_nowarn train_test(X, y; 
        model=(type=:decisiontree, params=(max_depth=5, min_samples_leaf=1)))
    m_params = train_test(X, y; model=(type=:decisiontree, params=(max_depth=5, min_samples_leaf=1)))
    @test m_params.setup.params.max_depth == 5
    @test m_params.setup.params.min_samples_leaf == 1

    @test_throws ArgumentError train_test(X, y; model=(type=:decisiontree, invalid=(max_depth=5,)))
    @test_throws ArgumentError train_test(X, y; model=(type=:decisiontree, params=(invalid=5,)))
end

@testset "validate_features" begin
    @test_nowarn train_test(X, y; model=(type=:decisiontree,), features=(minimum, mean, cov, mode_5))
    
    m_feat = train_test(X, y; model=(type=:decisiontree,), features=(minimum,))
    @test m_feat.setup.features == [minimum]

    @test_throws ArgumentError train_test(X, y; model=(type=:decisiontree,), features=(mean, :invalid_feature))
end

@testset "validate_win" begin
    @test_nowarn train_test(X, y; model=(type=:decisiontree,), win=(type=movingwindow, params=(window_size=12,)))

    @test_throws ArgumentError train_test(X, y; model=(type=:decisiontree,), win=(type=movingwindow, invalid=(window_size=12,)))
    @test_throws ArgumentError train_test(X, y; model=(type=:decisiontree,), win=(type=mean, params=(window_size=12,)))
    @test_throws ArgumentError train_test(X, y; model=(type=:decisiontree,), win=(type=movingwindow, params=(invalid=12,)))

    m_win = train_test(X, y; model=(type=:decisiontree,))
    @test m_win.setup.winparams.type == wholewindow
    d_win = train_test(X, y; model=(type=:decisiontree,), win=(type=adaptivewindow, params=(nwindows=5,)))
    @test d_win.setup.winparams.type == adaptivewindow
    @test d_win.setup.winparams.params.nwindows == 5

    @test_throws ArgumentError train_test(X, y; model=(type=:modaldecisiontree,), win=(type=adaptivewindow, params=(nwindows=2,)))

    nwindows = 5
    win = train_test(X, y; model=(type=:decisiontree,), features=(mean,), win=(type=adaptivewindow, params=(;nwindows)))
    @test size(win.ds.X, 2) == size(X, 2) * nwindows
    nwindows = 15
    win = train_test(X, y; model=(type=:decisiontree,), features=(mean,), win=(type=adaptivewindow, params=(;nwindows)))
    @test size(win.ds.X, 2) == size(X, 2) * nwindows
end

@testset "validate_rulesparams" begin
    @test_nowarn train_test(X, y; model=(type=:decisiontree,), extract_rules=(type=:intrees,))

    rule = train_test(X, y; model=(type=:decisiontree,), extract_rules=(type=:intrees,))
    @test rule.setup.rulesparams.type == :intrees
    rule = train_test(X, y; model=(type=:decisiontree,),)
    @test rule.setup.rulesparams == false

    @test_throws ArgumentError train_test(X, y; model=(type=:decisiontree, extract_rules=(invalid=PlainRuleExtractor(),)))
    @test_throws ArgumentError train_test(X, y; model=(type=:decisiontree, extract_rules=(type=mean, params=(compute_metrics=true,))))
    @test_throws ArgumentError train_test(X, y; model=(type=:decisiontree, extract_rules=(type=:lumen, params=(invalid=12,))))
end

@testset "validate_preprocess" begin
    train_seed = 11
    rng = Xoshiro(train_seed)
    @test_nowarn train_test(X, y; model=(type=:decisiontree,), preprocess=(;rng))

    @test_throws ArgumentError train_test(X, y; model=(type=:decisiontree,), preprocess=(;invalid=rng))
end
