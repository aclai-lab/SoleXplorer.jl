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

@testset "check every model" begin
    @test_nowarn model = symbolic_analysis(X, y; models=(type=:decisiontree,), preprocess=(;rng=Xoshiro(11)))
    @test_nowarn model = symbolic_analysis(X, y; models=(type=:decisiontree_classifier,), preprocess=(;rng=Xoshiro(11)))
    @test_nowarn model = symbolic_analysis(X, y; models=(type=:randomforest,), preprocess=(;rng=Xoshiro(11)))
    @test_nowarn model = symbolic_analysis(X, y; models=(type=:randomforest_classifier,), preprocess=(;rng=Xoshiro(11)))
    @test_nowarn model = symbolic_analysis(X, y; models=(type=:adaboost,), preprocess=(;rng=Xoshiro(11)))
    @test_nowarn model = symbolic_analysis(X, y; models=(type=:adaboost_classifier,), preprocess=(;rng=Xoshiro(11)))

    # TODO Marco, modalextractrules is not working with modal models, 
    # I don't know if it's a SoleXplorer fault or PostHoc is still in developement with that
    model = symbolic_analysis(X, y; models=(type=:modaldecisiontree,), preprocess=(;rng=Xoshiro(11)))
    model = symbolic_analysis(X, y; models=(type=:modalrandomforest,), preprocess=(;rng=Xoshiro(11)))
    model = symbolic_analysis(X, y; models=(type=:modaladaboost,), preprocess=(;rng=Xoshiro(11)))

    # TODO Marco, I've made a small hack to let PostHoc working with XgBoost singular behavior:
    # It's usual that XgBoost creates trees with no leafs.
    # Please check PostHoc intrees.jl, line 184, branch devPaso
    @test_nowarn model = symbolic_analysis(X, y; models=(type=:xgboost,), preprocess=(;rng=Xoshiro(11)))
    @test_nowarn model = symbolic_analysis(X, y; models=(type=:xgboost_classifier,), preprocess=(;rng=Xoshiro(11)))
end


