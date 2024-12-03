using Sole
using SoleXplorer
using Random, StatsBase, JLD2, DataFrames

# ---------------------------------------------------------------------------- #
#                                                                              #
#                              simple examples                                 #
#                                                                              #
# ---------------------------------------------------------------------------- #
X, y = SoleData.load_arff_dataset("NATOPS")
train_seed = 11;

# Note about Sole
#
# SoleBase: on branch dev (pulled dec, 2 2024)
# SoleData: on branch dev-autologiset (pulled dec, 2 2024)
# SoleDecisionTree: on branch dev (pulled dec, 2 2024)
# Sole: on branch dev (pulled dec, 2 2024)
# SoleLogics: on branch dev (pulled dec, 2 2024)
# SoleModels: on branch dev (pulled dec, 2 2024)

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

valid_X = get_treatment(X, model, features)
tt_pairs = get_partition(y)

fitmodel = SoleXplorer.get_fit(model, valid_X, y, tt_pairs; features=features, rng=rng)
dtree = SoleXplorer.get_test(model, valid_X, y, tt_pairs, fitmodel);

@show SoleXplorer.get_rules(dtree);
@show SoleXplorer.get_predict(fitmodel, valid_X, y, tt_pairs);

# ---------------------------------------------------------------------------- #
#                  decision tree with stratified sampling                      #
# ---------------------------------------------------------------------------- #
@info "Test 2: Decision Tree with stratified sampling"
model_name = :decision_tree
features = catch9
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name)
valid_X = get_treatment(X, model, features)
tt_pairs = get_partition(y; stratified_sampling=true, nfolds=3, rng=rng)

fitmodel = SoleXplorer.get_fit(model, valid_X, y, tt_pairs; features=features, rng=rng)
dtree = SoleXplorer.get_test(model, valid_X, y, tt_pairs, fitmodel);

@show SoleXplorer.get_rules(dtree);
@show SoleXplorer.get_predict(fitmodel, valid_X, y, tt_pairs);

# ---------------------------------------------------------------------------- #
#                       decision tree with model tuning                        #
# ---------------------------------------------------------------------------- #
@info "Test 3: Decision Tree with model tuning"
model_name = :decision_tree
features = catch9
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name)
valid_X = get_treatment(X, model, features)
tt_pairs = get_partition(y)

tuning_method = latinhypercube(gens=2, popsize=120)
ranges = [
    SoleXplorer.range(model.classifier, :merge_purity_threshold, lower=0, upper=1),
    SoleXplorer.range(model.classifier, :feature_importance, values=[:impurity, :split])
]
tunedmodel = SoleXplorer.get_tuning(model, tuning_method; ranges=ranges, n=25)

fitmodel = SoleXplorer.get_fit(tunedmodel, valid_X, y, tt_pairs; features=features, rng=rng)
dtree = SoleXplorer.get_test(tunedmodel, valid_X, y, tt_pairs, fitmodel);

@show SoleXplorer.get_rules(dtree);
@show SoleXplorer.get_predict(fitmodel, valid_X, y, tt_pairs);

# ---------------------------------------------------------------------------- #
#                          basic modal decision tree                           #
# ---------------------------------------------------------------------------- #
@info "Test 4: Basic Modal Decision Tree"
model_name = :modal_decision_tree
features = [minimum, mean]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name; relations=:IA7, features=features, set=X)
valid_X = get_treatment(X, model, features, nwindows=20)
tt_pairs = get_partition(y)

fitmodel = SoleXplorer.get_fit(model, valid_X, y, tt_pairs; features=features, rng=rng)
dtree = SoleXplorer.get_test(model, valid_X, y, tt_pairs, fitmodel)

@show SoleXplorer.get_rules(dtree);
@show SoleXplorer.get_predict(fitmodel, valid_X, y, tt_pairs);

# ---------------------------------------------------------------------------- #
#                       modal decision tree with tuning                        #
# ---------------------------------------------------------------------------- #
@info "Test 5: Modal Decision Tree with model tuning"
model_name = :modal_decision_tree
features = catch9
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name; relations=:IA7, features=features, set=X)
valid_X = get_treatment(X, model, features)
tt_pairs = get_partition(y)

tuning_method = adaptiveparticleswarm(rng=rng)
ranges = [
    SoleXplorer.range(model.classifier, :merge_purity_threshold, lower=0, upper=1),
    SoleXplorer.range(model.classifier, :feature_importance, values=[:impurity, :split])
]
tunedmodel = SoleXplorer.get_tuning(model, tuning_method; ranges=ranges, n=25)

fitmodel = SoleXplorer.get_fit(tunedmodel, valid_X, y, tt_pairs; features=features, rng=rng)
dtree = SoleXplorer.get_test(tunedmodel, valid_X, y, tt_pairs, fitmodel)

@show SoleXplorer.get_rules(dtree);
@show SoleXplorer.get_predict(fitmodel, valid_X, y, tt_pairs);

# ---------------------------------------------------------------------------- #
#                                                                              #
#                      examples based on world filtering                       #
#                                                                              #
# ---------------------------------------------------------------------------- #
X, y = SoleData.load_arff_dataset("NATOPS");
rng = Random.Xoshiro(1)

# ---------------------------------------------------------------------------- #
#                     get worlds: fixed length windows                         # 
# ---------------------------------------------------------------------------- #
@info "Test 6: Decision Tree based on world filtering 'fixedlength_windows'"
model_name = :decision_tree
features = [minimum, mean, StatsBase.cov, mode_5]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name)

valid_X = get_treatment(X, model, features; treatment=fixedlength_windows, winsize=30)
tt_pairs = get_partition(y)

fitmodel = SoleXplorer.get_fit(model, valid_X, y, tt_pairs; features=features, rng=rng)
dtree = SoleXplorer.get_test(model, valid_X, y, tt_pairs, fitmodel)

@show SoleXplorer.get_rules(dtree);
@show SoleXplorer.get_predict(fitmodel, valid_X, y, tt_pairs);

# ---------------------------------------------------------------------------- #
#                            get worlds: one window                            #
# ---------------------------------------------------------------------------- #
@info "Test 7: Decision Tree based on world filtering 'whole'"
model_name = :decision_tree
features = [minimum, mean, StatsBase.cov, mode_5]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name)

valid_X = get_treatment(X, model, features; treatment=whole)
tt_pairs = get_partition(y)

fitmodel = SoleXplorer.get_fit(model, valid_X, y, tt_pairs; features=features, rng=rng)
dtree = SoleXplorer.get_test(model, valid_X, y, tt_pairs, fitmodel)

@show SoleXplorer.get_rules(dtree);
@show SoleXplorer.get_predict(fitmodel, valid_X, y, tt_pairs);

# ---------------------------------------------------------------------------- #
#                    get worlds: absolute moving window                        #
# ---------------------------------------------------------------------------- #
@info "Test 8: Decision Tree based on world filtering 'absolute_movingwindow'"
model_name = :decision_tree
features = [minimum, mean, StatsBase.cov, mode_5]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name)

@btime valid_X = get_treatment(X, model, features; treatment=absolute_movingwindow, winsize=10, overlap=2)
tt_pairs = get_partition(y)

fitmodel = SoleXplorer.get_fit(model, valid_X, y, tt_pairs; features=features, rng=rng)
dtree = SoleXplorer.get_test(model, valid_X, y, tt_pairs, fitmodel)

@show SoleXplorer.get_rules(dtree);
@show SoleXplorer.get_predict(fitmodel, valid_X, y, tt_pairs);

# ---------------------------------------------------------------------------- #
#                     get worlds: absolute split window                        #
# ---------------------------------------------------------------------------- #
@info "Test 9: Decision Tree based on world filtering 'absolute_splitwindow'"
model_name = :decision_tree
features = [minimum, mean, StatsBase.cov, mode_5]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name)

valid_X = get_treatment(X, model, features; treatment=absolute_splitwindow, winsize=10)
tt_pairs = get_partition(y)

fitmodel = SoleXplorer.get_fit(model, valid_X, y, tt_pairs; features=features, rng=rng)
dtree = SoleXplorer.get_test(model, valid_X, y, tt_pairs, fitmodel)

@show SoleXplorer.get_rules(dtree);
@show SoleXplorer.get_predict(fitmodel, valid_X, y, tt_pairs);

# ---------------------------------------------------------------------------- #
#                    get worlds: relative moving window                        #
# ---------------------------------------------------------------------------- #
@info "Test 10: Decision Tree based on world filtering 'relative_movingwindow'"
model_name = :decision_tree
features = [minimum, mean, StatsBase.cov, mode_5]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name)

valid_X = get_treatment(X, model, features; treatment=relative_movingwindow, winsize=0.25, overlap=0.25)
tt_pairs = get_partition(y)

fitmodel = SoleXplorer.get_fit(model, valid_X, y, tt_pairs; features=features, rng=rng)
dtree = SoleXplorer.get_test(model, valid_X, y, tt_pairs, fitmodel)

@show SoleXplorer.get_rules(dtree);
@show SoleXplorer.get_predict(fitmodel, valid_X, y, tt_pairs);

# ---------------------------------------------------------------------------- #
#                     get worlds: relative split windows                       #
# ---------------------------------------------------------------------------- #
@info "Test 11: Decision Tree based on world filtering 'relative_splitwindows'"
model_name = :decision_tree
features = [minimum, mean, StatsBase.cov, mode_5]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name)

valid_X = get_treatment(X, model, features; treatment=relative_splitwindow, winsize=0.25)
tt_pairs = get_partition(y)

fitmodel = SoleXplorer.get_fit(model, valid_X, y, tt_pairs; features=features, rng=rng)
dtree = SoleXplorer.get_test(model, valid_X, y, tt_pairs, fitmodel)

@show SoleXplorer.get_rules(dtree);
@show SoleXplorer.get_predict(fitmodel, valid_X, y, tt_pairs);

# ---------------------------------------------------------------------------- #
#                     get worlds: fixed number windows                       #
# ---------------------------------------------------------------------------- #
@info "Test 12: Decision Tree based on world filtering 'adaptive_moving_windows'"
model_name = :decision_tree
features = [minimum, mean, StatsBase.cov, mode_5]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name)

valid_X = get_treatment(X, model, features; treatment=adaptive_moving_windows, nwindows=15, overlap=0.1)
tt_pairs = get_partition(y)

fitmodel = SoleXplorer.get_fit(model, valid_X, y, tt_pairs; features=features, rng=rng)
dtree = SoleXplorer.get_test(model, valid_X, y, tt_pairs, fitmodel)

@show SoleXplorer.get_rules(dtree);
@show SoleXplorer.get_predict(fitmodel, valid_X, y, tt_pairs);

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

valid_X = get_treatment(X, model, features)
tt_pairs = get_partition(y)

fitmodel = SoleXplorer.get_fit(model, valid_X, y, tt_pairs; features=features, rng=rng)
dtree = SoleXplorer.get_test(model, valid_X, y, tt_pairs, fitmodel)

@show SoleXplorer.get_rules(dtree);
@show SoleXplorer.get_predict(fitmodel, valid_X, y, tt_pairs);

# ---------------------------------------------------------------------------- #
#                    decision tree based on world filtering                    #
# ---------------------------------------------------------------------------- #
@info "Test 14: Decision Tree based on world filtering 'adaptive_moving_windows'"
model_name = :decision_tree
features = [minimum, mean, StatsBase.cov, mode_5]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name)

valid_X = get_treatment(X, model, features; treatment=SoleXplorer.adaptive_moving_windows, nwindows=3)
tt_pairs = get_partition(y)

fitmodel = SoleXplorer.get_fit(model,valid_X, y, tt_pairs; features=features, rng=rng)
dtree = SoleXplorer.get_test(model, valid_X, y, tt_pairs, fitmodel)

@show SoleXplorer.get_rules(dtree);
@show SoleXplorer.get_predict(fitmodel, valid_X, y, tt_pairs);

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
valid_X = get_treatment(X, model, features; treatment=SoleXplorer.adaptive_moving_windows, nwindows=3)
tt_pairs = get_partition(y)

fitmodel = SoleXplorer.get_fit(model, valid_X, y, tt_pairs; features=features, rng=rng)
dtree = SoleXplorer.get_test(model, valid_X, y, tt_pairs, fitmodel)

@show SoleXplorer.get_rules(dtree);
@show SoleXplorer.get_predict(fitmodel, valid_X, y, tt_pairs);
