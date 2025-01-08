using Sole
using SoleXplorer
import SoleXplorer as SX
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
features   = [mean]
rng        = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SX.get_model(model_name; relations=:IA7, features, set=X)
ds    = SX.preprocess_dataset(X, y, model; features, treatment_params=(nwindows=10,))

SX.modelfit!(model, ds; features, rng)
SX.modeltest!(model, ds)

@show SX.get_rules(model);
@show SX.get_predict(model, ds);

###### DEBUG
_model_name = :adaboost
_model = SX.get_model(_model_name)
_ds = SX.preprocess_dataset(X, y, _model; features)
SX.modelfit!(model, _ds; features, rng)
SX.modeltest!(model, _ds)

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
    SX.range(:merge_purity_threshold, lower=0, upper=1),
    SX.range(:feature_importance, values=[:impurity, :split])
]

model = SX.get_model(model_name; relations=:IA7, tuning=tuning_method, features=features, set=X, ranges=ranges, n=25)
ds = SX.preprocess_dataset(X, y, model; features=features)

SX.modelfit!(model, ds; features=features, rng=rng)
SX.modeltest!(model, ds)

@show SX.get_rules(model);
@show SX.get_predict(model, ds);

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

model = SX.get_model(model_name; relations=:IA7, features=features, set=X)
ds = SX.preprocess_dataset(X, y, model, features=features, treatment=SX.adaptivewindow, treatment_params=(nwindows=3,))

SX.modelfit!(model, ds; features=features, rng=rng)
SX.modeltest!(model, ds)

@show SX.get_rules(model);
@show SX.get_predict(model, ds);
