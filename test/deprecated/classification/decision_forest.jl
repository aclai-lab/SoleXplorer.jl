using Test
using Sole
import SoleXplorer as SX
using SoleXplorer
using Random, StatsBase, JLD2, DataFrames

# ---------------------------------------------------------------------------- #
X, y = SoleData.load_arff_dataset("NATOPS")
train_seed = 11;

# ---------------------------------------------------------------------------- #
#                            basic decision forest                             #
# ---------------------------------------------------------------------------- #
@info "Test 1: Decision Forest"
model_name = :decision_forest
features = [mean, maximum]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SX.get_model(model_name)
ds = SX.prepare_dataset(X, y, model; features)

SX.modelfit!(model, ds);
SX.modeltest!(model, ds);

@test_broken SX.get_rules(model, ds);
@test_nowarn SX.get_predict(model, ds);

@test_broken SX.get_rules(model, ds);
@test_nowarn SX.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
#                 decision forest with sratified sampling                      #
# ---------------------------------------------------------------------------- #
@info "Test 2: Decision Forest with stratified sampling"
model_name = :decision_forest
features = catch9
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SX.get_model(model_name)
ds = SX.prepare_dataset(X, y, model; features, stratified_sampling=true, nfolds=3, rng)

SX.modelfit!(model, ds);
SX.modeltest!(model, ds);

@test_broken SX.get_rules(model, ds);
@test_nowarn SX.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
#                      decision forest with mdel tuning                        #
# ---------------------------------------------------------------------------- #
@info "Test 3: Decision Forest with model tuning"
model_name = :decision_forest
features = catch9
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

tuning_method = latinhypercube(gens=2, popsize=120)
ranges = [
    SX.range(:sampling_fraction; lower=0.5, upper=0.8),
    SX.range(:feature_importance; values=[:impurity, :split])
]

model = SX.get_model(model_name; tuning=tuning_method, ranges, n=25)
ds = SX.prepare_dataset(X, y, model; features)

SX.modelfit!(model, ds);
SX.modeltest!(model, ds);

@test_broken SX.get_rules(model, ds);
@test_nowarn SX.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
# X, y = SoleData.load_arff_dataset("NATOPS");
# rng = Random.Xoshiro(1)

# ---------------------------------------------------------------------------- #
#                    Decision Forest based on wholewindow                      #
# ---------------------------------------------------------------------------- #
@info "Test 4: Decision Forest based on wholewindow"
model_name = :decision_forest
features = [minimum, mean, StatsBase.cov, mode_5]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SX.get_model(model_name)

ds = SX.prepare_dataset(X, y, model; features, treatment=wholewindow)

SX.modelfit!(model, ds);
SX.modeltest!(model, ds);

@test_broken SX.get_rules(model, ds);
@test_nowarn SX.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
#            Decision Forest based on movingwindow 'movingwindow'              #
# ---------------------------------------------------------------------------- #
@info "Test 5: Decision Forest based on movingwindow 'movingwindow'"
model_name = :decision_forest
features = [minimum, mean, StatsBase.cov, mode_5]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SX.get_model(model_name)
ds = SX.prepare_dataset(X, y, model; features, treatment=movingwindow, treatment_params=(nwindows=10, relative_overlap=0.2))

SX.modelfit!(model, ds);
SX.modeltest!(model, ds);

@test_broken SX.get_rules(model, ds);
@test_nowarn SX.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
#           Decision Forest based on movingwindow 'adaptivewindow'             #
# ---------------------------------------------------------------------------- #
@info "Test 6: Decision Forest based on movingwindow 'adaptivewindow'"
model_name = :decision_forest
features = [minimum, mean, StatsBase.cov, mode_5]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SX.get_model(model_name)
ds = SX.prepare_dataset(X, y, model; features, treatment=adaptivewindow, treatment_params=(nwindows=15, relative_overlap=0.1))

SX.modelfit!(model, ds);
SX.modeltest!(model, ds);

@test_broken SX.get_rules(model, ds);
@test_nowarn SX.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
filename = "respiratory_Pneumonia.jld2"
filepath = joinpath(@__DIR__, filename)
df = jldopen(filepath)
X, y = df["X"], df["y"]
rng = Random.Xoshiro(1)
train_seed = 11;

# ---------------------------------------------------------------------------- #
#                                decision forest                               #
# ---------------------------------------------------------------------------- #
@info "Test 7: Decision Forest"
model_name = :decision_forest
features = [minimum, mean, StatsBase.cov, mode_5]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SX.get_model(model_name)
ds = SX.prepare_dataset(X, y, model; features)

SX.modelfit!(model, ds);
SX.modeltest!(model, ds);

@test_broken SX.get_rules(model, ds);
@test_nowarn SX.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
#       Decision Forest based on movingwindow 'adaptive_moving_windows'        #
# ---------------------------------------------------------------------------- #
@info "Test 8: Decision Forest based on movingwindow 'adaptive_moving_windows'"
model_name = :decision_forest
features = [minimum, mean, StatsBase.cov, mode_5]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SX.get_model(model_name)
ds = SX.prepare_dataset(X, y, model; features, treatment=SX.adaptivewindow, treatment_params=(nwindows=3,))

SX.modelfit!(model,ds);
SX.modeltest!(model, ds);

@test_broken SX.get_rules(model, ds);
@test_nowarn SX.get_predict(model, ds);
