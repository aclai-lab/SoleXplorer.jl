using Sole
using SoleXplorer
using Random, StatsBase, JLD2, DataFrames
using RDatasets

# ---------------------------------------------------------------------------- #
X, y = SoleData.load_arff_dataset("NATOPS")
train_seed = 11;

# ---------------------------------------------------------------------------- #
#                           basic Ada boost forest                             #
# ---------------------------------------------------------------------------- #
@info "Test 1: Ada boost Forest"
model_name = :adaboost
features = [mean, maximum]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name)
ds = SoleXplorer.preprocess_dataset(X, y, model, features=features)

SoleXplorer.modelfit!(model, ds; features=features, rng=rng)
SoleXplorer.modeltest!(model, ds);

# @show SoleXplorer.get_rules(model);
@show SoleXplorer.get_predict(model, ds);

#####################################################################
# import DecisionTree as DT

# stumps, coeffs = DT.build_adaboost_stumps(string.(_ds.y[_ds.tt.train]), Matrix(_ds.X[_ds.tt.train, :]), 10);
# result = DT.apply_adaboost_stumps(stumps, coeffs, Matrix(_ds.X[_ds.tt.test, :]));

######################################################################

# ---------------------------------------------------------------------------- #
#                 Ada boost forest with sratified sampling                     #
# ---------------------------------------------------------------------------- #
@info "Test 2: Ada boost Forest with stratified sampling"
model_name = :adaboost
features = catch9
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name)
ds = SoleXplorer.preprocess_dataset(X, y, model; features=features, stratified_sampling=true, nfolds=3, rng=rng)

SoleXplorer.modelfit!(model, ds; features=features, rng=rng)
SoleXplorer.modeltest!(model, ds);

# @show SoleXplorer.get_rules(model);
@show SoleXplorer.get_predict(model, ds);

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
    SoleXplorer.range(:n_iter; lower=5, upper=20),
    SoleXplorer.range(:feature_importance; values=[:impurity, :split])
]

model = SoleXplorer.get_model(model_name; tuning=tuning_method, ranges=ranges, n=25)
ds = SoleXplorer.preprocess_dataset(X, y, model, features=features)

SoleXplorer.modelfit!(model, ds; features=features, rng=rng)
SoleXplorer.modeltest!(model, ds);

# @show SoleXplorer.get_rules(model);
@show SoleXplorer.get_predict(model, ds);

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

model = SoleXplorer.get_model(model_name)

ds = SoleXplorer.preprocess_dataset(X, y, model, features=features; treatment=wholewindow)

SoleXplorer.modelfit!(model, ds; features=features, rng=rng)
SoleXplorer.modeltest!(model, ds)

# @show SoleXplorer.get_rules(model);
@show SoleXplorer.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
#                           get worlds: moving window                          #
# ---------------------------------------------------------------------------- #
@info "Test 5: Ada boost Forest based on movingwindow 'movingwindow'"
model_name = :adaboost
features = [minimum, mean, StatsBase.cov, mode_5]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name)

ds = SoleXplorer.preprocess_dataset(X, y, model, features=features; treatment=movingwindow, treatment_params=(nwindows=10, relative_overlap=0.2))

SoleXplorer.modelfit!(model, ds; features=features, rng=rng)
SoleXplorer.modeltest!(model, ds)

# @show SoleXplorer.get_rules(model);
@show SoleXplorer.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
#                      get worlds: fixed number windows                        #
# ---------------------------------------------------------------------------- #
@info "Test 6: Ada boost Forest based on movingwindow 'adaptivewindow'"
model_name = :adaboost
features = [minimum, mean, StatsBase.cov, mode_5]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name)

ds = SoleXplorer.preprocess_dataset(X, y, model, features=features, treatment=adaptivewindow, treatment_params=(nwindows=15, relative_overlap=0.1))

SoleXplorer.modelfit!(model, ds; features=features, rng=rng)
SoleXplorer.modeltest!(model, ds)

# @show SoleXplorer.get_rules(model);
@show SoleXplorer.get_predict(model, ds);

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

model = SoleXplorer.get_model(model_name)

ds = SoleXplorer.preprocess_dataset(X, y, model, features=features)

SoleXplorer.modelfit!(model, ds; features=features, rng=rng)
SoleXplorer.modeltest!(model, ds)

# @show SoleXplorer.get_rules(model);
@show SoleXplorer.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
#                    Ada boost forest based n movingwindow                     #
# ---------------------------------------------------------------------------- #
@info "Test 8: Ada boost Forest based on movingwindow 'adaptive_moving_windows'"
model_name = :adaboost
features = [minimum, mean, StatsBase.cov, mode_5]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name)

ds = SoleXplorer.preprocess_dataset(X, y, model, features=features, treatment=SoleXplorer.adaptivewindow, treatment_params=(nwindows=3,))

SoleXplorer.modelfit!(model,ds; features=features, rng=rng)
SoleXplorer.modeltest!(model, ds)

# @show SoleXplorer.get_rules(model);
@show SoleXplorer.get_predict(model, ds);

