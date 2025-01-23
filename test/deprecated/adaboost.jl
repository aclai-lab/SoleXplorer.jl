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
#                           basic Ada boost forest                             #
# ---------------------------------------------------------------------------- #
@info "Test 1: Basic AdaBoost"
model_name = :adaboost
features = catch9
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SX.get_model(model_name)
ds = SX.prepare_dataset(X, y, model, features=features)

SX.modelfit!(model, ds; features=features, rng=rng)
SX.modeltest!(model, ds);

@show SX.get_rules(model);
@show SX.get_predict(model, ds);


#################################
# ---------------------------------------------------------------------------- #
#                             basic decision tree                              #
# ---------------------------------------------------------------------------- #
@info "Test 1: Decision Tree"
model_name = :decision_tree
features = catch9
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

_model = SX.get_model(model_name)
# ds = SX.prepare_dataset(X, y, model, features=features)

SX.modelfit!(_model, ds; features=features, rng=rng)
SX.modeltest!(_model, ds);

@show SX.get_rules(_model);
@show SX.get_predict(_model, ds);

##############################################à

import DecisionTree as DT
stumps, coeffs = DT.build_adaboost_stumps(string.(ds.y[ds.tt.train]), Matrix(ds.X[ds.tt.train, :]), 10);
result = DT.apply_adaboost_stumps(stumps, coeffs, Matrix(ds.X[ds.tt.test, :]));
# accuracy = nfoldCV_stumps(labels, features, n_folds, n_iterations; verbose = true)



# ---------------------------------------------------------------------------- #
#                 Ada boost forest with sratified sampling                     #
# ---------------------------------------------------------------------------- #
@info "Test 2: Ada boost Forest with stratified sampling"
model_name = :adaboost
features = catch9
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SX.get_model(model_name)
ds = SX.prepare_dataset(X, y, model; features=features, stratified_sampling=true, nfolds=3, rng=rng)

SX.modelfit!(model, ds; features=features, rng=rng)
SX.modeltest!(model, ds);

# @show SX.get_rules(model);
@show SX.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
#                     Ada boost forest with mdel tuning                        #
# ---------------------------------------------------------------------------- #
@info "Test 3: Ada boost Forest with model tuning"
model_name = :adaboost
features = catch9
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

tuning_method = latinhypercube(gens=2, popsize=120)
ranges = [
    SX.range(:n_iter; lower=5, upper=20),
    SX.range(:feature_importance; values=[:impurity, :split])
]

model = SX.get_model(model_name; tuning=tuning_method, ranges=ranges, n=25)
ds = SX.prepare_dataset(X, y, model, features=features)

SX.modelfit!(model, ds; features=features, rng=rng)
SX.modeltest!(model, ds);

# @show SX.get_rules(model);
@show SX.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
# X, y = SoleData.load_arff_dataset("NATOPS");
# rng = Random.Xoshiro(1)

# ---------------------------------------------------------------------------- #
#                            get worlds: one window                            #
# ---------------------------------------------------------------------------- #
@info "Test 4: Ada boost Forest based on wholewindow"
model_name = :adaboost
features = [minimum, mean, StatsBase.cov, mode_5]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SX.get_model(model_name)

ds = SX.prepare_dataset(X, y, model, features=features; treatment=wholewindow)

SX.modelfit!(model, ds; features=features, rng=rng)
SX.modeltest!(model, ds)

# @show SX.get_rules(model);
@show SX.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
#                           get worlds: moving window                          #
# ---------------------------------------------------------------------------- #
@info "Test 5: Ada boost Forest based on movingwindow 'movingwindow'"
model_name = :adaboost
features = [minimum, mean, StatsBase.cov, mode_5]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SX.get_model(model_name)

ds = SX.prepare_dataset(X, y, model, features=features; treatment=movingwindow, treatment_params=(nwindows=10, relative_overlap=0.2))

SX.modelfit!(model, ds; features=features, rng=rng)
SX.modeltest!(model, ds)

# @show SX.get_rules(model);
@show SX.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
#                      get worlds: fixed number windows                        #
# ---------------------------------------------------------------------------- #
@info "Test 6: Ada boost Forest based on movingwindow 'adaptivewindow'"
model_name = :adaboost
features = [minimum, mean, StatsBase.cov, mode_5]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SX.get_model(model_name)

ds = SX.prepare_dataset(X, y, model, features=features, treatment=adaptivewindow, treatment_params=(nwindows=15, relative_overlap=0.1))

SX.modelfit!(model, ds; features=features, rng=rng)
SX.modeltest!(model, ds)

# @show SX.get_rules(model);
@show SX.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
filename = "respiratory_Pneumonia.jld2"
filepath = joinpath(@__DIR__, filename)
df = jldopen(filepath)
X, y = df["X"], df["y"]
rng = Random.Xoshiro(1)
train_seed = 11;

# ---------------------------------------------------------------------------- #
#                                Ada boost forest                              #
# ---------------------------------------------------------------------------- #
@info "Test 7: Ada boost Forest"
model_name = :adaboost
features = [minimum, mean, StatsBase.cov, mode_5]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SX.get_model(model_name)

ds = SX.prepare_dataset(X, y, model, features=features)

SX.modelfit!(model, ds; features=features, rng=rng)
SX.modeltest!(model, ds)

# @show SX.get_rules(model);
@show SX.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
#                    Ada boost forest based n movingwindow                     #
# ---------------------------------------------------------------------------- #
@info "Test 8: Ada boost Forest based on movingwindow 'adaptive_moving_windows'"
model_name = :adaboost
features = [minimum, mean, StatsBase.cov, mode_5]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SX.get_model(model_name)

ds = SX.prepare_dataset(X, y, model, features=features, treatment=SX.adaptivewindow, treatment_params=(nwindows=3,))

SX.modelfit!(model,ds; features=features, rng=rng)
SX.modeltest!(model, ds)

# @show SX.get_rules(model);
@show SX.get_predict(model, ds);

