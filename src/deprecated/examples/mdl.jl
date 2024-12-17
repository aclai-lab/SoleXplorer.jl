using Sole, SoleBase
using SoleXplorer
using Random, StatsBase, JLD2, DataFrames
using RDatasets

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

# ---------------------------------------------------------------------------- #
#                         basic modal decision list                            #
# ---------------------------------------------------------------------------- #
@info "Test 16: Modal Decision List"
model_name = :modal_decision_list
features = [mean]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name)
ds = SoleXplorer.preprocess_dataset(X, y, model)

SoleXplorer.modelfit!(model, ds; features=features, rng=rng)

# dtree = SoleXplorer.get_test(model, valid_X, y, tt_pairs)

# @show SoleXplorer.get_rules(dtree);
# @show SoleXplorer.get_predict(fitted_model, valid_X, y, tt_pairs);

# ---------------------------------------------------------------------------- #
#                                                                              #
#                            examples based on vectors                         #
#                                                                              #
# ---------------------------------------------------------------------------- #
X, y = SoleData.load_arff_dataset("NATOPS")
train_seed = 11;

# ---------------------------------------------------------------------------- #
#                         basic modal decision list                            #
# ---------------------------------------------------------------------------- #
@info "Test 17: Modal Decision List on time series"
model_name = :modal_decision_list
features = [mean]
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name)
ds = SoleXplorer.preprocess_dataset(X, y, model; treatment=SoleXplorer.wholewindow)

SoleXplorer.modelfit!(model, ds; features=features, rng=rng)

