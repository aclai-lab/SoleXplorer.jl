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
    model = symbolic_analysis(X, y; models=(type=:decisiontree,), preprocess=(;rng))
    @test model.classifier == SoleXplorer.DecisionTreeClassifier
    model = symbolic_analysis(X, y; models=(type=:randomforest,), preprocess=(;rng))

end

model = symbolic_analysis(X, y; models=(type=:decisiontree,), preprocess=(;rng=Xoshiro(11)))
model = symbolic_analysis(X, y; models=(type=:randomforest,), preprocess=(;rng=Xoshiro(11)));