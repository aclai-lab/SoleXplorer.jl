using Sole
using SoleXplorer
using Random, StatsBase

X, y = SoleData.load_arff_dataset("NATOPS");
rng = Random.Xoshiro(1)
features = [minimum, mean, StatsBase.cov, mode_5]
nwindows = 10
overlap = 2

@info "Test: Decision Tree based on world filtering"
model_name = :modal_decision_tree
features = [minimum, mean, StatsBase.cov, mode_5]

model = SoleXplorer.get_model(model_name; relations=:IA7, features=features, set=X)

valid_X = get_treatment(X, model, features; worlds=realtive_movingwindow)
