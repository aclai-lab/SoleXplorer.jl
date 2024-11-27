using Sole
using SoleXplorer
using Random, StatsBase

# References
# https://github.com/ablaom/MLJTutorial.jl/blob/dev/notebooks/04_tuning/notebook.ipynb
# https://juliaai.github.io/DataScienceTutorials.jl/getting-started/model-tuning/

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
ranges = SoleXplorer.range(model.classifier, :feature_importance, values=[:impurity, :split])
tunedmodel = SoleXplorer.get_tuning(model, tuning_method; ranges=ranges, n=25)

fitmodel = SoleXplorer.get_fit(tunedmodel, valid_X, y, tt_pairs; features=features, rng=rng)
dtree = SoleXplorer.get_test(tunedmodel, valid_X, y, tt_pairs, fitmodel);

@show SoleXplorer.get_rules(dtree);
@show SoleXplorer.get_predict(fitmodel, valid_X, y, tt_pairs);

# ---------------------------------------------------------------------------- #
#                          basic modal decision tree                           #
# ---------------------------------------------------------------------------- #
@info "Test 3: Modal Decision Tree"
model_name = :modal_decision_tree
features = [minimum, mean]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name; relations=:IA7, features=features, set=X)

valid_X = get_treatment(X, model, features)
tt_pairs = get_partition(y)

fit_model = SoleXplorer.get_fit(valid_X, y, tt_pairs, model; features=features, rng=rng)
dtree = SoleXplorer.get_test(valid_X, y, tt_pairs, model, fit_model)

@show SoleXplorer.get_rules(dtree);
@show SoleXplorer.get_predict(fit_model, valid_X, y, tt_pairs);

# ---------------------------------------------------------------------------- #
#                       modal decision tree with tuning                        #
# ---------------------------------------------------------------------------- #
@info "Test 4: Modal Decision Tree with catch9 and tuning: be patient."
features = catch9
model_name = :modal_decision_tree
tuning_method = :latinhypercube

model = SoleXplorer.get_model(model_name; relations=:IA7, features=features, set=X)
tuning = SoleXplorer.get_tuning(model, tuning_method)

valid_X = get_treatment(X, model, features; nwindows = 10, relative_overlap=0.3)
tt_pairs = get_partition(y; stratified_sampling=true, nfolds=3, rng=rng)

fit_model = SoleXplorer.get_fit(valid_X, y, tt_pairs, model; tuning=tuning, features=features, rng=rng)
dtree = SoleXplorer.get_test(valid_X, y, tt_pairs, model, fit_model)

@show SoleXplorer.get_rules(dtree);
@show SoleXplorer.get_predict(fit_model, valid_X, y, tt_pairs);

# ---------------------------------------------------------------------------- #
#                                 show results                                 #
# ---------------------------------------------------------------------------- #
@info "Results"
@show dt_rules
@show dtmt_rules
@show mdt_rules
@show mdtmt_rules


latin = LatinHypercube(gens=2, popsize=120)

r = MLJ.range(model.classifier, :feature_importance, values=[:impurity, :split])
self_tuning_forest = TunedModel(
    model=model.classifier,
    tuning=latin,
    resampling=CV(nfolds=6),
    range=r,
    # measure=rms,
    n=25
);
mach = machine(self_tuning_forest, valid_X, y);
fit!(mach, verbosity=0)