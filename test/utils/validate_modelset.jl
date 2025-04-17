using Test
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

# @testset "validate_rulesparams" begin
#     @test_nowarn train_test(X, y; model=(type=:decisiontree,), rulesparams=(type=PlainRuleExtractor(),))

#     @test_throws ArgumentError train_test(X, y; model=(type=:decisiontree, rulesparams=(invalid=PlainRuleExtractor(),)))
#     @test_throws ArgumentError train_test(X, y; model=(type=:decisiontree, rulesparams=(type=mean, params=(compute_metrics=true,))))
#     @test_throws ArgumentError train_test(X, y; model=(type=:decisiontree, rulesparams=(type=PlainRuleExtractor(), params=(invalid=12,))))

#     m_rls = train_test(X, y; model=(type=:decisiontree,))
#     @test m_rls.setup.rulesparams.type == PlainRuleExtractor()
#     d_rls = train_test(X, y; model=(type=:decisiontree, rulesparams=(type=InTreesRuleExtractor(), params=(prune_rules=false,))))
#     @test d_rls.setup.rulesparams.type == InTreesRuleExtractor()
#     @test d_rls.setup.rulesparams.params.prune_rules == false
#     g_rls = train_test(X, y; model=(type=:decisiontree,), globals=(rulesparams=(type=InTreesRuleExtractor(), params=(prune_rules=false,)),))
#     @test g_rls.setup.rulesparams.type == InTreesRuleExtractor()
#     @test g_rls.setup.rulesparams.params.prune_rules == false
# end

@testset "validate_preprocess" begin
    train_seed = 11
    rng = Xoshiro(train_seed)
    @test_nowarn train_test(X, y; model=(type=:decisiontree,), preprocess=(;rng))

    @test_throws ArgumentError train_test(X, y; model=(type=:decisiontree,), preprocess=(;invalid=rng))
end
