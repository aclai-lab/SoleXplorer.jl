using Sole
using SoleXplorer
using Random, StatsBase, JLD2, DataFrames
using RDatasets

# ---------------------------------------------------------------------------- #
X, y = SoleData.load_arff_dataset("NATOPS")
train_seed = 11;

# ---------------------------------------------------------------------------- #
#                          basic modal adaboost tree                           #
# ---------------------------------------------------------------------------- #
@info "Test 1: Basic Modal Adaboost Tree"
model_name = :modal_adaboost
features = [minimum, mean]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name; relations=:IA7, features, set=X)
ds = SoleXplorer.preprocess_dataset(X, y, model; features, treatment_params=(nwindows=5,))

SoleXplorer.modelfit!(model, ds; features, rng=rng)
SoleXplorer.modeltest!(model, ds)

@show SoleXplorer.get_rules(model);
@show SoleXplorer.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
#                       modal adaboost tree with tuning                        #
# ---------------------------------------------------------------------------- #
@info "Test 2: Modal Adaboost Tree with model tuning"
model_name = :modal_adaboost
features = [minimum, mean]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

tuning_method = adaptiveparticleswarm(rng=rng)
ranges = [
    SoleXplorer.range(:merge_purity_threshold, lower=0, upper=1),
    SoleXplorer.range(:feature_importance, values=[:impurity, :split])
]

model = SoleXplorer.get_model(model_name; relations=:IA7, tuning=tuning_method, features=features, set=X, ranges=ranges, n=25)
ds = SoleXplorer.preprocess_dataset(X, y, model; features=features)

SoleXplorer.modelfit!(model, ds; features=features, rng=rng)
SoleXplorer.modeltest!(model, ds)

@show SoleXplorer.get_rules(model);
@show SoleXplorer.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
filename = "respiratory_Pneumonia.jld2"
filepath = joinpath(@__DIR__, filename)
df = jldopen(filepath)
X, y = df["X"], df["y"]
rng = Random.Xoshiro(1)
train_seed = 11;

# ---------------------------------------------------------------------------- #
#                             modal adaboost tree                              #
# ---------------------------------------------------------------------------- #
@info "Test 3: Modal Adaboost Tree on dataset with different lengths"
model_name = :modal_adaboost
# features = [minimum, mean]
features = catch9
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name; relations=:IA7, features=features, set=X)
ds = SoleXplorer.preprocess_dataset(X, y, model, features=features, treatment=SoleXplorer.adaptivewindow, treatment_params=(nwindows=3,))

SoleXplorer.modelfit!(model, ds; features=features, rng=rng)
SoleXplorer.modeltest!(model, ds)

@show SoleXplorer.get_rules(model);
@show SoleXplorer.get_predict(model, ds);
