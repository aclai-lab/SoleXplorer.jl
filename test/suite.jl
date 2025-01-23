using Test
using Sole
using SoleXplorer
using Random, StatsBase, JLD2, DataFrames
using RDatasets

train_seed = 1

X, y = SoleData.load_arff_dataset("NATOPS")

my_decision_tree_parametrization = TODO # copia quella del SoleXplorer.AVAIL_MODELS[:decision_tree], ma voglio poter specificare dei ranges diversi.

for model_name in [:decision_tree, my_decision_tree_parametrization, :decision_forest, :modal_decision_tree]
  @info "Test 6: Decision Forest based on movingwindow 'adaptivewindow'"
  features = [minimum, mean, StatsBase.cov, mode_5]
  rng = Random.Xoshiro(train_seed)
  @test_nowarn symbolic_analysis(model, X, y, features, treatment, treatment_params; rng)
end

function symbolic_analysis(model::Union{ModelConfig,Symbol}, X, y, features, treatment, treatment_params; rng)
  model = SoleXplorer.get_model(model)

  ds = SoleXplorer.prepare_dataset(X, y, model, features, treatment, treatment_params)

  SoleXplorer.modelfit!(model, ds; features=features, rng=rng)
  SoleXplorer.modeltest!(model, ds)
  SoleXplorer.get_rules(model);
  SoleXplorer.get_predict(model, ds);
end