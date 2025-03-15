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
    @test_nowarn traintest(X, y; models=(type=:decisiontree_classifier,))

    y_class = y
    auto_algo = traintest(X, y_class; models=(type=:decisiontree,))
    @test auto_algo.setup.config.algo == :classification
    y_reg = rand(length(y))
    auto_algo = traintest(X, y_reg; models=(type=:decisiontree,))
    @test auto_algo.setup.config.algo == :regression

    @test_throws ArgumentError traintest(X, y; models=(missing_type=:decisiontree_classifier,))
    @test_throws ArgumentError traintest(X, y; models=(type=:invalid_classifier,))
end

@testset "validate_params" begin
    @test_nowarn traintest(X, y; 
        models=(type=:decisiontree, params=(max_depth=5, min_samples_leaf=1)))
    m_params = traintest(X, y; models=(type=:decisiontree, params=(max_depth=5, min_samples_leaf=1)))
    @test m_params.setup.params.max_depth == 5
    @test m_params.setup.params.min_samples_leaf == 1

    @test_nowarn traintest(X, y; 
        models=(type=:decisiontree, params=(max_depth=5, min_samples_leaf=1)), globals=(params=(min_samples_split=11,),))
    g_params = traintest(X, y; models=(
        type=:decisiontree, 
        params=(max_depth=5, min_samples_leaf=1)), 
        globals=(params=(min_samples_split=11,),))
    @test g_params.setup.params.min_samples_split == 11

    @test_throws ArgumentError traintest(X, y; models=(type=:decisiontree, invalid=(max_depth=5,)))

    @test_throws ArgumentError traintest(X, y; models=(type=:decisiontree, params=(invalid=5,)))
    @test_throws ArgumentError traintest(X, y; models=(type=:decisiontree,), globals=(params=(invalid=5,),))
end

@testset "validate_features" begin
    @test_nowarn traintest(X, y; models=(type=:decisiontree, features=[minimum, mean, cov, mode_5]))
    
    m_feat = traintest(X, y; models=(type=:decisiontree, features=[minimum]))
    @test m_feat.setup.features == [minimum]
    g_feat = traintest(X, y; models=(type=:decisiontree,), globals=(features=[mode_5],))
    @test g_feat.setup.features == [mode_5]

    @test_throws ArgumentError traintest(X, y; models=(type=:decisiontree, features=[:invalid_feature]))
end

@testset "validate_winparams" begin
    @test_nowarn traintest(X, y; models=(type=:decisiontree, winparams=(type=movingwindow, params=(window_size=12,))))

    @test_throws ArgumentError traintest(X, y; models=(type=:decisiontree, winparams=(type=movingwindow, invalid=(window_size=12,))))
    @test_throws ArgumentError traintest(X, y; models=(type=:decisiontree, winparams=(type=mean, params=(window_size=12,))))
    @test_throws ArgumentError traintest(X, y; models=(type=:decisiontree, winparams=(type=movingwindow, params=(invalid=12,))))

    m_win = traintest(X, y; models=(type=:decisiontree,))
    @test m_win.setup.winparams.type == wholewindow
    d_win = traintest(X, y; models=(type=:decisiontree, winparams=(type=adaptivewindow, params=(nwindows=5,))))
    @test d_win.setup.winparams.type == adaptivewindow
    @test d_win.setup.winparams.params.nwindows == 5
    g_win = traintest(X, y; models=(type=:decisiontree,), globals=(winparams=(type=movingwindow, params=(window_size=12,)),))
    @test g_win.setup.winparams.type == movingwindow
    @test g_win.setup.winparams.params.window_size == 12

    @test_throws ArgumentError traintest(X, y; models=(type=:modaldecisiontree, winparams=(type=adaptivewindow, params=(nwindows=2,))))

    nwindows = 5
    win = traintest(X, y; models=(type=:decisiontree, features=[mean], winparams=(type=adaptivewindow, params=(;nwindows))))
    @test size(win.ds.X, 2) == size(X, 2) * nwindows
    nwindows = 15
    win = traintest(X, y; models=(type=:decisiontree, features=[mean], winparams=(type=adaptivewindow, params=(;nwindows))))
    @test size(win.ds.X, 2) == size(X, 2) * nwindows
end

@testset "validate_rulesparams" begin
    @test_nowarn traintest(X, y; models=(type=:decisiontree, rulesparams=(type=PlainRuleExtractor(),)))

    @test_throws ArgumentError traintest(X, y; models=(type=:decisiontree, rulesparams=(invalid=PlainRuleExtractor(),)))
    @test_throws ArgumentError traintest(X, y; models=(type=:decisiontree, rulesparams=(type=mean, params=(compute_metrics=true,))))
    @test_throws ArgumentError traintest(X, y; models=(type=:decisiontree, rulesparams=(type=PlainRuleExtractor(), params=(invalid=12,))))

    m_rls = traintest(X, y; models=(type=:decisiontree,))
    @test m_rls.setup.rulesparams.type == PlainRuleExtractor()
    d_rls = traintest(X, y; models=(type=:decisiontree, rulesparams=(type=InTreesRuleExtractor(), params=(prune_rules=false,))))
    @test d_rls.setup.rulesparams.type == InTreesRuleExtractor()
    @test d_rls.setup.rulesparams.params.prune_rules == false
    g_rls = traintest(X, y; models=(type=:decisiontree,), globals=(rulesparams=(type=InTreesRuleExtractor(), params=(prune_rules=false,)),))
    @test g_rls.setup.rulesparams.type == InTreesRuleExtractor()
    @test g_rls.setup.rulesparams.params.prune_rules == false
end

@testset "validate_preprocess" begin
    train_seed = 11
    rng = Xoshiro(train_seed)
    @test_nowarn traintest(X, y; models=(type=:decisiontree,), preprocess=(;rng))

    @test_throws ArgumentError traintest(X, y; models=(type=:decisiontree,), preprocess=(;invalid=rng))
end
