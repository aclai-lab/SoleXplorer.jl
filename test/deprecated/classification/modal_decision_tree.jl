using Test
using Sole
import SoleXplorer as SX
using SoleXplorer
using Random, StatsBase, JLD2, DataFrames

# ---------------------------------------------------------------------------- #
X, y = SoleData.load_arff_dataset("NATOPS")
train_seed = 11;

# ---------------------------------------------------------------------------- #
#                          basic modal decision tree                           #
# ---------------------------------------------------------------------------- #
@info "Test 1: Basic Modal Decision Tree"
model_name = :modal_decision_tree
features = [minimum, mean]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SX.get_model(model_name; relations=:IA7, features)
ds = SX.prepare_dataset(X, y, model; features, treatment_params=(nwindows=20, relative_overlap=0.3))

SX.modelfit!(model, ds);
SX.modeltest!(model, ds);

@test_nowarn SX.get_rules(model, ds);
@test_nowarn SX.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
#                       modal decision tree with tuning                        #
# ---------------------------------------------------------------------------- #
@info "Test 2: Modal Decision Tree with model tuning"
model_name = :modal_decision_tree
features = [std, SX.mode_10]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

tuning_method = adaptiveparticleswarm(; rng)
ranges = [
    SX.range(:merge_purity_threshold, lower=0, upper=1),
    SX.range(:feature_importance, values=[:impurity, :split])
]

model = SX.get_model(model_name; relations=:IA7, tuning=tuning_method, features, ranges, n=25)
ds = SX.prepare_dataset(X, y, model; features)

SX.modelfit!(model, ds);
SX.modeltest!(model, ds);

@test_nowarn SX.get_rules(model, ds);
@test_nowarn SX.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
filename = "respiratory_Pneumonia.jld2"
filepath = joinpath(@__DIR__, filename)
df = jldopen(filepath)
X, y = df["X"], df["y"]
rng = Random.Xoshiro(1)
train_seed = 11;

# ---------------------------------------------------------------------------- #
#                     modal decision tree adaptive window                      #
# ---------------------------------------------------------------------------- #
@info "Test 3: Decision Forest based on movingwindow 'adaptivewindow'"
model_name = :modal_decision_tree
features = [mean]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SX.get_model(model_name; relations=:IA7, features)
ds = SX.prepare_dataset(X, y, model; features, treatment=adaptivewindow, treatment_params=(nwindows=15, ))

SX.modelfit!(model, ds);
SX.modeltest!(model, ds);

@test_nowarn SX.get_rules(model, ds);
@test_nowarn SX.get_predict(model, ds);