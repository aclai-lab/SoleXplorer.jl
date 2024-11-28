using Sole
using SoleXplorer
using Random, StatsBase
using DataFrames, JLD2

filename = "examples/respiratory_Pneumonia.jld2"
df = jldopen(filename)
X, y = df["X"], df["y"]
rng = Random.Xoshiro(1)

# ---------------------------------------------------------------------------- #
#                                 decision tree                                #
# ---------------------------------------------------------------------------- #
@info "Test 1: Decision Tree"
model_name = :decision_tree
features = [minimum, mean, StatsBase.cov, mode_5]

model = SoleXplorer.get_model(model_name)

valid_X = get_treatment(X, model, features)
tt_pairs = get_partition(y)

fit_model = SoleXplorer.get_fit(valid_X, y, tt_pairs, model; features=features, rng=rng)
dtree = SoleXplorer.get_test(valid_X, y, tt_pairs, model, fit_model)

dt_rules = SoleXplorer.get_rules(dtree)

# ---------------------------------------------------------------------------- #
#                    decision tree based on world filtering                    #
# ---------------------------------------------------------------------------- #
@info "Test 2: Decision Tree based on world filtering 'fixednumber_windows'"
model_name = :decision_tree
features = [minimum, mean, StatsBase.cov, mode_5]

model = SoleXplorer.get_model(model_name)

valid_X = get_treatment(X, model, features; treatment=SoleXplorer.fixednumber_windows, nwindows=3)
tt_pairs = get_partition(y)

fit_model = SoleXplorer.get_fit(valid_X, y, tt_pairs, model; features=features, rng=rng)
dtree = SoleXplorer.get_test(valid_X, y, tt_pairs, model, fit_model)

wfdt_rules = SoleXplorer.get_rules(dtree)

# ---------------------------------------------------------------------------- #
#                             modal decision tree                              #
# ---------------------------------------------------------------------------- #
@info "Test 3: Modal Decision Tree"
model_name = :modal_decision_tree
features = [minimum, mean]

model = SoleXplorer.get_model(model_name; relations=:IA7, features=features, set=X)

valid_X = get_treatment(X, model, features; nwindows = 20, relative_overlap=0.3)
tt_pairs = get_partition(y)

fit_model = SoleXplorer.get_fit(valid_X, y, tt_pairs, model; features=features, rng=rng)
dtree = SoleXplorer.get_test(valid_X, y, tt_pairs, model, fit_model)

mdt_rules = SoleXplorer.get_rules(dtree);

# ---------------------------------------------------------------------------- #
#                                 show results                                 #
# ---------------------------------------------------------------------------- #
@info "Results"
@show dt_rules
@show wfdt_rules
@show mdt_rules
