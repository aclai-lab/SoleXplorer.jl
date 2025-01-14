using Test
using Sole
import SoleXplorer as SX
using SoleXplorer
using Random, StatsBase, JLD2, DataFrames
using RDatasets

# ---------------------------------------------------------------------------- #
X, y = SoleData.load_arff_dataset("NATOPS")
train_seed = 11;

# ---------------------------------------------------------------------------- #
#                             basic decision tree                              #
# ---------------------------------------------------------------------------- #
@info "Test 1: Decision Tree"
model_name = :decision_tree
features = catch9
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SX.get_model(model_name)
ds = SX.preprocess_dataset(X, y, model, features=features)

SX.modelfit!(model, ds; features=features, rng=rng)
SX.modeltest!(model, ds);

@test_nowarn SX.get_rules(model);
@test_nowarn SX.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
#                  decision tree with stratified sampling                      #
# ---------------------------------------------------------------------------- #
@info "Test 2: Decision Tree with stratified sampling"
model_name = :decision_tree
features = catch9
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SX.get_model(model_name)
ds = SX.preprocess_dataset(X, y, model; features=features, stratified_sampling=true, nfolds=3, rng=rng)

SX.modelfit!(model, ds; features=features, rng=rng)
SX.modeltest!(model, ds);

@test_nowarn SX.get_rules(model);
@test_nowarn SX.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
#                       decision tree with model tuning                        #
# ---------------------------------------------------------------------------- #
@info "Test 3: Decision Tree with model tuning"
model_name = :decision_tree
features = catch9
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

tuning_method = latinhypercube(gens=2, popsize=120)
ranges = [
    SX.range(:merge_purity_threshold; lower=0, upper=1),
    SX.range(:feature_importance; values=[:impurity, :split])
]

model = SX.get_model(model_name; tuning=tuning_method, ranges=ranges, n=25)
ds = SX.preprocess_dataset(X, y, model, features=features)

SX.modelfit!(model, ds; features=features, rng=rng)
SX.modeltest!(model, ds);

@test_nowarn SX.get_rules(model);
@test_nowarn SX.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
# X, y = SoleData.load_arff_dataset("NATOPS");
# rng = Random.Xoshiro(1)

# ---------------------------------------------------------------------------- #
#                            get worlds: one window                            #
# ---------------------------------------------------------------------------- #
@info "Test 4: Decision Tree based on wholewindow"
model_name = :decision_tree
features = [minimum, mean, StatsBase.cov, mode_5]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SX.get_model(model_name)

ds = SX.preprocess_dataset(X, y, model, features=features; treatment=wholewindow)

SX.modelfit!(model, ds; features=features, rng=rng)
SX.modeltest!(model, ds)

@test_nowarn SX.get_rules(model);
@test_nowarn SX.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
#                           get worlds: moving window                          #
# ---------------------------------------------------------------------------- #
@info "Test 5: Decision Tree based on movingwindow 'movingwindow'"
model_name = :decision_tree
features = [minimum, mean, StatsBase.cov, mode_5]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SX.get_model(model_name)

ds = SX.preprocess_dataset(X, y, model, features=features; treatment=movingwindow, treatment_params=(nwindows=10, relative_overlap=0.2))

SX.modelfit!(model, ds; features=features, rng=rng)
SX.modeltest!(model, ds)

@test_nowarn SX.get_rules(model);
@test_nowarn SX.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
#                     get worlds: fixed number windows                       #
# ---------------------------------------------------------------------------- #
@info "Test 6: Decision Tree based on movingwindow 'adaptivewindow'"
model_name = :decision_tree
features = [minimum, mean, StatsBase.cov, mode_5]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SX.get_model(model_name)

ds = SX.preprocess_dataset(X, y, model, features=features, treatment=adaptivewindow, treatment_params=(nwindows=15, relative_overlap=0.1))

SX.modelfit!(model, ds; features=features, rng=rng)
SX.modeltest!(model, ds)

@test_nowarn SX.get_rules(model);
@test_nowarn SX.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
filename = "respiratory_Pneumonia.jld2"
filepath = joinpath(@__DIR__, filename)
df = jldopen(filepath)
X, y = df["X"], df["y"]
rng = Random.Xoshiro(1)
train_seed = 11;

# ---------------------------------------------------------------------------- #
#                                 decision tree                                #
# ---------------------------------------------------------------------------- #
@info "Test 7: Decision Tree"
model_name = :decision_tree
features = [minimum, mean, StatsBase.cov, mode_5]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SX.get_model(model_name)

ds = SX.preprocess_dataset(X, y, model, features=features)

SX.modelfit!(model, ds; features=features, rng=rng)
SX.modeltest!(model, ds)

@test_nowarn SX.get_rules(model);
@test_nowarn SX.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
#                    decision tree based on movingwindow                    #
# ---------------------------------------------------------------------------- #
@info "Test 8: Decision Tree based on movingwindow 'adaptive_moving_windows'"
model_name = :decision_tree
features = [minimum, mean, StatsBase.cov, mode_5]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SX.get_model(model_name)

ds = SX.preprocess_dataset(X, y, model, features=features, treatment=SX.adaptivewindow, treatment_params=(nwindows=3,))

SX.modelfit!(model,ds; features=features, rng=rng)
SX.modeltest!(model, ds)

@test_nowarn SX.get_rules(model);
@test_nowarn SX.get_predict(model, ds);
