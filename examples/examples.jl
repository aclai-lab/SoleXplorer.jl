using Sole
using SoleXplorer
using Random, StatsBase, JLD2, DataFrames
using RDatasets

# ---------------------------------------------------------------------------- #
#                                                                              #
#                              simple examples                                 #
#                                                                              #
# ---------------------------------------------------------------------------- #
X, y = SoleData.load_arff_dataset("NATOPS")
train_seed = 11;

# Note about Sole
#
# SoleBase: on branch dev (pulled dec, 7 2024)
# SoleData: on branch dev-autologiset (pulled dec, 7 2024)
# SoleDecisionTree: on branch dev (pulled dec, 2 2024)
# Sole: on branch dev (pulled dec, 7 2024)
# SoleLogics: on branch dev (pulled dec, 7 2024)
# SoleModels: on branch dev (pulled dec, 7 2024)
# ModalDecisionLists: on branch dev (pulled dec, 7 2024)
# ModalDecisionTees: on branch dev (pulled dec, 7 2024)

# References
# https://github.com/ablaom/MLJTutorial.jl/blob/dev/notebooks/04_tuning/notebook.ipynb
# https://juliaai.github.io/DataScienceTutorials.jl/getting-started/model-tuning/

# ---------------------------------------------------------------------------- #
#                             basic decision tree                              #
# ---------------------------------------------------------------------------- #
@info "Test 1: Decision Tree"
model_name = :decision_tree
features = catch9
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name)

ds = SoleXplorer.preprocess_dataset(X, y, model, features=features)

SoleXplorer.modelfit!(model, ds; features=features, rng=rng)
dtree = SoleXplorer.modeltest(model, ds);

@show SoleXplorer.get_rules(dtree);
@show SoleXplorer.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
#                  decision tree with stratified sampling                      #
# ---------------------------------------------------------------------------- #
@info "Test 2: Decision Tree with stratified sampling"
model_name = :decision_tree
features = catch9
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name)
ds = SoleXplorer.preprocess_dataset(X, y, model; features=features, stratified_sampling=true, nfolds=3, rng=rng)

SoleXplorer.modelfit!(model, ds; features=features, rng=rng)
dtree = SoleXplorer.modeltest(model, ds);

@show SoleXplorer.get_rules(dtree);
@show SoleXplorer.get_predict(model, ds);

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
    SoleXplorer.range(:merge_purity_threshold; lower=0, upper=1),
    SoleXplorer.range(:feature_importance; values=[:impurity, :split])
]

model = SoleXplorer.get_model(model_name; tuning=tuning_method, ranges=ranges, n=25)
ds = SoleXplorer.preprocess_dataset(X, y, model, features=features)

SoleXplorer.modelfit!(model, ds; features=features, rng=rng)
dtree = SoleXplorer.modeltest(model, ds);

@show SoleXplorer.get_rules(dtree);
@show SoleXplorer.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
#                          basic modal decision tree                           #
# ---------------------------------------------------------------------------- #
@info "Test 4: Basic Modal Decision Tree"
model_name = :modal_decision_tree
features = [minimum, mean]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name; relations=:IA7, features=features, set=X)
ds = SoleXplorer.preprocess_dataset(X, y, model; features=features, nwindows=20)

SoleXplorer.modelfit!(model, ds; features=features, rng=rng)
dtree = SoleXplorer.modeltest(model, ds)

@show SoleXplorer.get_rules(dtree);
@show SoleXplorer.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
#                       modal decision tree with tuning                        #
# ---------------------------------------------------------------------------- #
@info "Test 5: Modal Decision Tree with model tuning"
model_name = :modal_decision_tree
features = catch9
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

tuning_method = adaptiveparticleswarm(rng=rng)
ranges = [
    SoleXplorer.range(:merge_purity_threshold, lower=0, upper=1),
    SoleXplorer.range(:feature_importance, values=[:impurity, :split])
]

model = SoleXplorer.get_model(model_name; relations=:IA7, features=features, set=X, ranges=ranges, n=25)
ds = SoleXplorer.preprocess_dataset(X, y, model; features=features)

SoleXplorer.modelfit!(model, ds; features=features, rng=rng)
dtree = SoleXplorer.modeltest(model, ds)

@show SoleXplorer.get_rules(dtree);
@show SoleXplorer.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
#                                                                              #
#                      examples based on movingwindow                       #
#                                                                              #
# ---------------------------------------------------------------------------- #
X, y = SoleData.load_arff_dataset("NATOPS");
rng = Random.Xoshiro(1)

# ---------------------------------------------------------------------------- #
#                            get worlds: one window                            #
# ---------------------------------------------------------------------------- #
@info "Test 7: Decision Tree based on wholewindow"
model_name = :decision_tree
features = [minimum, mean, StatsBase.cov, mode_5]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name)

ds = SoleXplorer.preprocess_dataset(X, y, model, features=features; treatment=wholewindow)

SoleXplorer.modelfit!(model, ds; features=features, rng=rng)
dtree = SoleXplorer.modeltest(model, ds)

@show SoleXplorer.get_rules(dtree);
@show SoleXplorer.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
#                           get worlds: moving window                          #
# ---------------------------------------------------------------------------- #
@info "Test 8: Decision Tree based on movingwindow 'movingwindow'"
model_name = :decision_tree
features = [minimum, mean, StatsBase.cov, mode_5]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name)

ds = SoleXplorer.preprocess_dataset(X, y, model, features=features; treatment=movingwindow, treatment_params=(nwindows=10, relative_overlap=0.2))

SoleXplorer.modelfit!(model, ds; features=features, rng=rng)
dtree = SoleXplorer.modeltest(model, ds)

@show SoleXplorer.get_rules(dtree);
@show SoleXplorer.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
#                     get worlds: fixed number windows                       #
# ---------------------------------------------------------------------------- #
@info "Test 12: Decision Tree based on movingwindow 'adaptivewindow'"
model_name = :decision_tree
features = [minimum, mean, StatsBase.cov, mode_5]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name)

ds = SoleXplorer.preprocess_dataset(X, y, model, features=features, treatment=adaptivewindow, treatment_params=(nwindows=15, relative_overlap=0.1))

SoleXplorer.modelfit!(model, ds; features=features, rng=rng)
dtree = SoleXplorer.modeltest(model, ds)

@show SoleXplorer.get_rules(dtree);
@show SoleXplorer.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
#                                                                              #
#               examples based on dataset with different lengths               #
#                                                                              #
# ---------------------------------------------------------------------------- #
filename = "examples/respiratory_Pneumonia.jld2"
df = jldopen(filename)
X, y = df["X"], df["y"]
rng = Random.Xoshiro(1)
train_seed = 11;

# ---------------------------------------------------------------------------- #
#                                 decision tree                                #
# ---------------------------------------------------------------------------- #
@info "Test 13: Decision Tree"
model_name = :decision_tree
features = [minimum, mean, StatsBase.cov, mode_5]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name)

ds = SoleXplorer.preprocess_dataset(X, y, model, features=features)

SoleXplorer.modelfit!(model, ds; features=features, rng=rng)
dtree = SoleXplorer.modeltest(model, ds)

@show SoleXplorer.get_rules(dtree);
@show SoleXplorer.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
#                    decision tree based on movingwindow                    #
# ---------------------------------------------------------------------------- #
@info "Test 14: Decision Tree based on movingwindow 'adaptive_moving_windows'"
model_name = :decision_tree
features = [minimum, mean, StatsBase.cov, mode_5]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name)

ds = SoleXplorer.preprocess_dataset(X, y, model, features=features, treatment=SoleXplorer.adaptivewindow, treatment_params=(nwindows=3,))

SoleXplorer.modelfit!(model,ds; features=features, rng=rng)
dtree = SoleXplorer.modeltest(model, ds)

@show SoleXplorer.get_rules(dtree);
@show SoleXplorer.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
#                             modal decision tree                              #
# ---------------------------------------------------------------------------- #
@info "Test 15: Modal Decision Tree"
model_name = :modal_decision_tree
# features = [minimum, mean]
features = catch9
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name; relations=:IA7, features=features, set=X)
ds = SoleXplorer.preprocess_dataset(X, y, model, features=features, treatment=SoleXplorer.adaptivewindow, treatment_params=(nwindows=3,))

SoleXplorer.modelfit!(model, ds; features=features, rng=rng)
dtree = SoleXplorer.modeltest(model, ds)

@show SoleXplorer.get_rules(dtree);
@show SoleXplorer.get_predict(model, ds);

# ---------------------------------------------------------------------------- #
#                                                                              #
#                            modal decision list                               #
#                                                                              #
# ---------------------------------------------------------------------------- #
table = RDatasets.dataset("datasets", "iris")
y = table[:, :Species]
X = select(table, Not([:Species]));

# X, y = preprocess_inputdata(X,y)
train_seed = 11;

# # ---------------------------------------------------------------------------- #
# #                         basic modal decision list                            #
# # ---------------------------------------------------------------------------- #
# @info "Test 16: Modal Decision List"
# model_name = :modal_decision_list
# features = [mean]
# rng = Random.Xoshiro(train_seed)
# Random.seed!(train_seed)

# model = SoleXplorer.get_model(model_name)
# ds = SoleXplorer.preprocess_dataset(X, y, model)

# SoleXplorer.modelfit!(model, ds; features=features, rng=rng)

# dlist = SoleXplorer.modeltest(model, ds)

# @show SoleXplorer.get_rules(dlist);
# @show SoleXplorer.get_predict(model, dss);

# # ---------------------------------------------------------------------------- #
# #                                                                              #
# #                            examples based on vectors                         #
# #                                                                              #
# # ---------------------------------------------------------------------------- #
# X, y = SoleData.load_arff_dataset("NATOPS")
# train_seed = 11;

# # ---------------------------------------------------------------------------- #
# #                         basic modal decision list                            #
# # ---------------------------------------------------------------------------- #
# @info "Test 17: Modal Decision List on time series"
# model_name = :modal_decision_list
# features = [mean]
# rng = Random.Xoshiro(train_seed)
# Random.seed!(train_seed)

# model = SoleXplorer.get_model(model_name)
# ds = SoleXplorer.preprocess_dataset(X, y, model; treatment=SoleXplorer.wholewindow)

# SoleXplorer.modelfit!(model, ds; features=features, rng=rng)


