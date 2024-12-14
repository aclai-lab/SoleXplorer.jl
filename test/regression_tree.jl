using Sole
using SoleXplorer
using Random, StatsBase, JLD2, DataFrames
using MLDatasets

# ---------------------------------------------------------------------------- #
X = MLDatasets.BostonHousing().dataframe
features = setdiff(names(X), ["MEDV"])

train_ratio = 0.8
train_indices = randperm(nrow(df))[1:Int(round(train_ratio * nrow(df)))]

dtrain = df[train_indices, :]
deval = df[setdiff(1:nrow(df), train_indices), :]

_mean, _std = mean(X.MEDV), std(X.MEDV)
transform!(dtrain, :MEDV => (x -> (x .- _mean) ./ _std) => "target")
transform!(deval, :MEDV => (x -> (x .- _mean) ./ _std) => "target")

target_name = "target"
train_seed = 11;

# ---------------------------------------------------------------------------- #
#                            basic regression tree                             #
# ---------------------------------------------------------------------------- #
@info "Test 1: Regression Tree"
model_name = :regression_tree
features = catch9
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name)
ds = SoleXplorer.preprocess_dataset(X, y, model, features=features)

SoleXplorer.modelfit!(model, ds; features=features, rng=rng)
SoleXplorer.modeltest!(model, ds);

@show SoleXplorer.get_rules(model);
@show SoleXplorer.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
#                 regression tree with stratified sampling                     #
# ---------------------------------------------------------------------------- #
@info "Test 2: Regression Tree with stratified sampling"
model_name = :regression_tree
features = catch9
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name)
ds = SoleXplorer.preprocess_dataset(X, y, model; features=features, stratified_sampling=true, nfolds=3, rng=rng)

SoleXplorer.modelfit!(model, ds; features=features, rng=rng)
SoleXplorer.modeltest!(model, ds);

@show SoleXplorer.get_rules(model);
@show SoleXplorer.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
#                      regression tree with model tuning                       #
# ---------------------------------------------------------------------------- #
@info "Test 3: Regression Tree with model tuning"
model_name = :regression_tree
features = catch9
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

tuning_method = latinhypercube(gens=2, popsize=120)
ranges = [
    SoleXplorer.range(:merge_purity_threshold; lower=0, upper=1),
    SoleXplorer.range(:feature_importance; values=[:impurity, :split])
]

model = SoleXplorer.get_model(model_name; tuning=tuning_method, ranges=ranges, n=25)
ds = SoleXplorer.preprocess_dataset(X, y, model, features=features)

SoleXplorer.modelfit!(model, ds; features=features, rng=rng)
SoleXplorer.modeltest!(model, ds);

@show SoleXplorer.get_rules(model);
@show SoleXplorer.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
X, y = SoleData.load_arff_dataset("NATOPS");
rng = Random.Xoshiro(1)

# ---------------------------------------------------------------------------- #
#                            get worlds: one window                            #
# ---------------------------------------------------------------------------- #
@info "Test 4: Regression Tree based on wholewindow"
model_name = :regression_tree
features = [minimum, mean, StatsBase.cov, mode_5]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name)

ds = SoleXplorer.preprocess_dataset(X, y, model, features=features; treatment=wholewindow)

SoleXplorer.modelfit!(model, ds; features=features, rng=rng)
SoleXplorer.modeltest!(model, ds)

@show SoleXplorer.get_rules(model);
@show SoleXplorer.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
#                           get worlds: moving window                          #
# ---------------------------------------------------------------------------- #
@info "Test 5: Regression Tree based on movingwindow 'movingwindow'"
model_name = :regression_tree
features = [minimum, mean, StatsBase.cov, mode_5]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name)

ds = SoleXplorer.preprocess_dataset(X, y, model, features=features; treatment=movingwindow, treatment_params=(nwindows=10, relative_overlap=0.2))

SoleXplorer.modelfit!(model, ds; features=features, rng=rng)
SoleXplorer.modeltest!(model, ds)

@show SoleXplorer.get_rules(model);
@show SoleXplorer.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
#                      get worlds: fixed number windows                        #
# ---------------------------------------------------------------------------- #
@info "Test 6: Regression Tree based on movingwindow 'adaptivewindow'"
model_name = :regression_tree
features = [minimum, mean, StatsBase.cov, mode_5]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name)

ds = SoleXplorer.preprocess_dataset(X, y, model, features=features, treatment=adaptivewindow, treatment_params=(nwindows=15, relative_overlap=0.1))

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
#                                 regression tree                                #
# ---------------------------------------------------------------------------- #
@info "Test 7: Regression Tree"
model_name = :regression_tree
features = [minimum, mean, StatsBase.cov, mode_5]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name)

ds = SoleXplorer.preprocess_dataset(X, y, model, features=features)

SoleXplorer.modelfit!(model, ds; features=features, rng=rng)
SoleXplorer.modeltest!(model, ds)

@show SoleXplorer.get_rules(model);
@show SoleXplorer.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
#                    regression tree based on movingwindow                    #
# ---------------------------------------------------------------------------- #
@info "Test 8: Regression Tree based on movingwindow 'adaptive_moving_windows'"
model_name = :regression_tree
features = [minimum, mean, StatsBase.cov, mode_5]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name)

ds = SoleXplorer.preprocess_dataset(X, y, model, features=features, treatment=SoleXplorer.adaptivewindow, treatment_params=(nwindows=3,))

SoleXplorer.modelfit!(model,ds; features=features, rng=rng)
SoleXplorer.modeltest!(model, ds)

@show SoleXplorer.get_rules(model);
@show SoleXplorer.get_predict(model, ds);


#####################################################################à
n,m = 10^3, 5;
X = rand(n,m);
features = [:x1, :x2, :x3, :x4, :x5]
weights = rand(-1:1,m);
y = X * weights;

R1Tree = DecisionTreeRegressor(
    min_samples_leaf=5,
    merge_purity_threshold=0.1,
    rng=stable_rng(),
)
R2Tree = DecisionTreeRegressor(min_samples_split=5, rng=stable_rng())
model1 = MLJ.machine(model.classifier, X, y, features)
