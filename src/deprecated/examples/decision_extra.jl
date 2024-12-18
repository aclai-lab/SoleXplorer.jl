using DecisionTree

features, labels = load_data("iris")    # also see "adult" and "digits" datasets

# the data loaded are of type Array{Any}
# cast them to concrete types for better performance
features = float.(features)
labels   = string.(labels)

# train adaptive-boosted stumps, using 7 iterations
model, coeffs = build_adaboost_stumps(labels, features, 7);
# apply learned model
apply_adaboost_stumps(model, coeffs, [5.9,3.0,5.1,1.9])
# get the probability of each label
apply_adaboost_stumps_proba(model, coeffs, [5.9,3.0,5.1,1.9], ["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
# run 3-fold cross validation for boosted stumps, using 7 iterations
n_iterations=7; n_folds=3
accuracy = nfoldCV_stumps(labels, features,
                          n_folds,
                          n_iterations;
                          verbose = true)

######################################################################################
using DecisionTree
using Sole
using SoleXplorer
using Random, StatsBase, JLD2, DataFrames
using RDatasets, MLJ

# ---------------------------------------------------------------------------- #
table = RDatasets.dataset("datasets", "iris")
y = table[:, :Species]
X = select(table, Not([:Species]));
train_seed = 11;
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

##########################################################################################
features = Matrix(selectrows(ds.X, ds.tt.train))
labels = string.(ds.y[ds.tt.train])

# train adaptive-boosted stumps, using 7 iterations
model, coeffs = build_adaboost_stumps(labels, features, 10);
# apply learned model
Xtest = Matrix(selectrows(ds.X, ds.tt.test))
apply_adaboost_stumps(model, coeffs, Xtest)
# get the probability of each label
acc=apply_adaboost_stumps_proba(model, coeffs, Xtest, ["setosa", "versicolor", "virginica"])
# run 3-fold cross validation for boosted stumps, using 7 iterations
n_iterations=7; n_folds=3
accuracy = nfoldCV_stumps(labels, features,
                          n_folds,
                          n_iterations;
                          verbose = false);

##########################################################################################
n, m = 10^3, 5
features = randn(n, m)
weights = rand(-2:2, m)
labels = features * weights

# train regression tree
model = build_tree(labels, features)
# apply learned model
apply_tree(model, [-0.9,3.0,5.1,1.9,0.0])
# run 3-fold cross validation, returns array of coefficients of determination (R^2)
n_folds = 3
r2 = nfoldCV_tree(labels, features, n_folds)

# set of regression parameters and respective default values
# pruning_purity: purity threshold used for post-pruning (default: 1.0, no pruning)
# max_depth: maximum depth of the decision tree (default: -1, no maximum)
# min_samples_leaf: the minimum number of samples each leaf needs to have (default: 5)
# min_samples_split: the minimum number of samples in needed for a split (default: 2)
# min_purity_increase: minimum purity needed for a split (default: 0.0)
# n_subfeatures: number of features to select at random (default: 0, keep all)
# keyword rng: the random number generator or seed to use (default Random.GLOBAL_RNG)
n_subfeatures = 0; max_depth = -1; min_samples_leaf = 5
min_samples_split = 2; min_purity_increase = 0.0; pruning_purity = 1.0 ; seed=3

model = build_tree(labels, features,
                   n_subfeatures,
                   max_depth,
                   min_samples_leaf,
                   min_samples_split,
                   min_purity_increase;
                   rng = seed)

r2 =  nfoldCV_tree(labels, features,
                   n_folds,
                   pruning_purity,
                   max_depth,
                   min_samples_leaf,
                   min_samples_split,
                   min_purity_increase;
                   verbose = true,
                   rng = seed)

using MLJ
doc("DecisionTreeClassifier", pkg="DecisionTree")
                   
                          
