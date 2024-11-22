using Sole
using SoleXplorer
using Random, StatsBase

# References
# https://github.com/ablaom/MLJTutorial.jl/blob/dev/notebooks/04_tuning/notebook.ipynb
# https://juliaai.github.io/DataScienceTutorials.jl/getting-started/model-tuning/

X, y = SoleData.load_arff_dataset("NATOPS");
rng = Random.Xoshiro(1)
features = [minimum, mean, StatsBase.cov, mode_5]

# ---------------------------------------------------------------------------- #
#                                 decision tree                                #
# ---------------------------------------------------------------------------- #
@info "Test 1: Decision Tree"
model_name = :decision_tree

model = SoleXplorer.get_model(model_name)

valid_X = get_treatment(X, model, features)
tt_pairs = get_partition(y)

fit_model = SoleXplorer.get_fit(valid_X, y, tt_pairs, model; features=features, rng=rng)
dtree = SoleXplorer.get_test(valid_X, y, tt_pairs, model, fit_model)

rules = SoleXplorer.get_rules(dtree)

# ---------------------------------------------------------------------------- #
#           decision tree with model tuning and stratified sampling            #
# ---------------------------------------------------------------------------- #
@info "Test 2: Decision Tree with model tuning and stratified sampling"
model_name = :decision_tree
tuning_method = :latinhypercube

model = SoleXplorer.get_model(model_name)
tuning = SoleXplorer.get_tuning(model, tuning_method)

valid_X = get_treatment(X, model, features)
tt_pairs = get_partition(y; stratified_sampling=true, nfolds=3, rng=rng)

fit_model = SoleXplorer.get_fit(valid_X, y, tt_pairs, model; tuning=tuning, features=features, rng=rng)
dtree = SoleXplorer.get_test(valid_X, y, tt_pairs, model, fit_model)

rules = SoleXplorer.get_rules(dtree)

# # ---------------------------------------------------------------------------- #
# #                  hybrid propositional/modal decision tree                    #
# # ---------------------------------------------------------------------------- #
# @info "Test 3: Windowed Decision Tree"
# model_name = :win_decision_tree

# model = SoleXplorer.get_model(model_name)

# valid_X = get_treatment(X, model, features)
# tt_pairs = get_partition(y)

# fit_model = SoleXplorer.get_fit(valid_X, y, tt_pairs, model; features=features, rng=rng)
# dtree = SoleXplorer.get_test(valid_X, y, tt_pairs, model, fit_model)

# rules = SoleXplorer.get_rules(dtree)

# # ---------------------------------------------------------------------------- #
# #                   hybrid decision tree with model tuning                     #
# # ---------------------------------------------------------------------------- #
# @info "Test 4: Windowed Decision Tree with model tuning and stratified sampling"
# model_name = :win_decision_tree
# tuning_method = :particleswarm

# model = SoleXplorer.get_model(model_name)
# tuning = SoleXplorer.get_tuning(model, tuning_method)

# valid_X = get_treatment(X, model, features)
# tt_pairs = get_partition(y; stratified_sampling=true, nfolds=3, rng=rng)

# fit_model = SoleXplorer.get_fit(valid_X, y, tt_pairs, model; tuning=tuning, features=features, rng=rng)
# dtree = SoleXplorer.get_test(valid_X, y, tt_pairs, model, fit_model)

# rules = SoleXplorer.get_rules(dtree)

# ---------------------------------------------------------------------------- #
#                            modal decision tree                               #
# ---------------------------------------------------------------------------- #
@info "Test 3: Modal Decision Tree"
model_name = :modal_decision_tree
features = [minimum, mean]

model = SoleXplorer.get_model(model_name; relations=:IA7, features=features, set=X)

valid_X = get_treatment(X, model, features; nwindows = 20, relative_overlap=0.3)
tt_pairs = get_partition(y)

fit_model = SoleXplorer.get_fit(valid_X, y, tt_pairs, model; features=features, rng=rng)
dtree = SoleXplorer.get_test(valid_X, y, tt_pairs, model, fit_model)

rules = SoleXplorer.get_rules(dtree)

# ---------------------------------------------------------------------------- #
#                       modal decision tree with tuning                        #
# ---------------------------------------------------------------------------- #
@info "Test 4: Modal Decision Tree with catch9 and tuning: be patient."
features = catch9
model_name = :modal_decision_tree
tuning_method = :latinhypercube

model = SoleXplorer.get_model(model_name; relations=:IA7, features=features, set=X)
tuning = SoleXplorer.get_tuning(model, tuning_method)

valid_X = get_treatment(X, model, features; nwindows = 20, relative_overlap=0.3)
tt_pairs = get_partition(y; stratified_sampling=true, nfolds=3, rng=rng)

fit_model = SoleXplorer.get_fit(valid_X, y, tt_pairs, model; tuning=tuning, features=features, rng=rng)
dtree = SoleXplorer.get_test(valid_X, y, tt_pairs, model, fit_model)

rules = SoleXplorer.get_rules(dtree)


