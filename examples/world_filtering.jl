using Sole
using SoleXplorer
using Random, StatsBase

X, y = SoleData.load_arff_dataset("NATOPS");
rng = Random.Xoshiro(1)

# ---------------------------------------------------------------------------- #
#                     get worlds: fixed length windows                         # 
# ---------------------------------------------------------------------------- #
@info "Test 1: Decision Tree based on world filtering 'fixed_windows'"
model_name = :decision_tree
features = [minimum, mean, StatsBase.cov, mode_5]

model = SoleXplorer.get_model(model_name)

valid_X = get_treatment(X, model, features; treatment=fixed_windows, nwindows=30)
tt_pairs = get_partition(y)

fit_model = SoleXplorer.get_fit(valid_X, y, tt_pairs, model; features=features, rng=rng)
dtree = SoleXplorer.get_test(valid_X, y, tt_pairs, model, fit_model)

rules = SoleXplorer.get_rules(dtree)

# ---------------------------------------------------------------------------- #
#                            get worlds: one window                            #
# ---------------------------------------------------------------------------- #
@info "Test 2: Decision Tree based on world filtering 'whole'"
model_name = :decision_tree
features = [minimum, mean, StatsBase.cov, mode_5]

model = SoleXplorer.get_model(model_name)

valid_X = get_treatment(X, model, features; treatment=whole)
tt_pairs = get_partition(y)

fit_model = SoleXplorer.get_fit(valid_X, y, tt_pairs, model; features=features, rng=rng)
dtree = SoleXplorer.get_test(valid_X, y, tt_pairs, model, fit_model)

rules = SoleXplorer.get_rules(dtree)

# ---------------------------------------------------------------------------- #
#                    get worlds: absolute moving window                        #
# ---------------------------------------------------------------------------- #
@info "Test 3: Decision Tree based on world filtering 'absolute_movingwindow'"
model_name = :decision_tree
features = [minimum, mean, StatsBase.cov, mode_5]

model = SoleXplorer.get_model(model_name)

valid_X = get_treatment(X, model, features; treatment=absolute_movingwindow, nwindows=10, overlap=2)
tt_pairs = get_partition(y)

fit_model = SoleXplorer.get_fit(valid_X, y, tt_pairs, model; features=features, rng=rng)
dtree = SoleXplorer.get_test(valid_X, y, tt_pairs, model, fit_model)

rules = SoleXplorer.get_rules(dtree)

# ---------------------------------------------------------------------------- #
#                     get worlds: absolute split window                        #
# ---------------------------------------------------------------------------- #
@info "Test 4: Decision Tree based on world filtering 'absolute_splitwindow'"
model_name = :decision_tree
features = [minimum, mean, StatsBase.cov, mode_5]

model = SoleXplorer.get_model(model_name)

valid_X = get_treatment(X, model, features; treatment=absolute_splitwindow, nwindows=10)
tt_pairs = get_partition(y)

fit_model = SoleXplorer.get_fit(valid_X, y, tt_pairs, model; features=features, rng=rng)
dtree = SoleXplorer.get_test(valid_X, y, tt_pairs, model, fit_model)

rules = SoleXplorer.get_rules(dtree)

# ---------------------------------------------------------------------------- #
#                    get worlds: relative moving window                        #
# ---------------------------------------------------------------------------- #
@info "Test 5: Decision Tree based on world filtering 'relative_movingwindow'"
model_name = :decision_tree
features = [minimum, mean, StatsBase.cov, mode_5]

model = SoleXplorer.get_model(model_name)

valid_X = get_treatment(X, model, features; treatment=realtive_movingwindow, nwindows=0.25, overlap=0.25)
tt_pairs = get_partition(y)

fit_model = SoleXplorer.get_fit(valid_X, y, tt_pairs, model; features=features, rng=rng)
dtree = SoleXplorer.get_test(valid_X, y, tt_pairs, model, fit_model)

rules = SoleXplorer.get_rules(dtree)

# ---------------------------------------------------------------------------- #
#                     get worlds: relative split windows                       #
# ---------------------------------------------------------------------------- #
@info "Test 6: Decision Tree based on world filtering 'relative_splitwindows'"
model_name = :decision_tree
features = [minimum, mean, StatsBase.cov, mode_5]

model = SoleXplorer.get_model(model_name)

valid_X = get_treatment(X, model, features; treatment=relative_splitwindow, nwindows=0.25)
tt_pairs = get_partition(y)

fit_model = SoleXplorer.get_fit(valid_X, y, tt_pairs, model; features=features, rng=rng)
dtree = SoleXplorer.get_test(valid_X, y, tt_pairs, model, fit_model)

rules = SoleXplorer.get_rules(dtree)