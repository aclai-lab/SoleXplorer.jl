using Sole
using SoleXplorer
using Random, StatsBase, JLD2, DataFrames
using XGBoost, CategoricalArrays, OrderedCollections, MLBase, AbstractTrees
using MLJ, MLJBase, MLJXGBoostInterface

# ---------------------------------------------------------------------------- #
#                                  XGBoost.jl                                  #
# ---------------------------------------------------------------------------- #
X, y = SoleData.load_arff_dataset("NATOPS")
train_seed = 11;
model_name = :decision_tree
features = catch9
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name)

valid_X = get_treatment(X, model, features)
tt_pairs = get_partition(y)

# da inserire in get get_partition
y_code = @. CategoricalArrays.levelcode(y) - 1 # convert to 0-based indexing
dtrain = XGBoost.DMatrix((valid_X[tt_pairs.train, :], y_code[tt_pairs.train]), feature_names=names(valid_X))
dtest = XGBoost.DMatrix((valid_X[tt_pairs.test, :], y_code[tt_pairs.test]), feature_names=names(valid_X))

# create and train a gradient boosted tree model of 5 trees
bst = XGBoost.xgboost(dtrain, num_round=5, max_depth=6, objective="reg:squarederror")
y_predict = XGBoost.predict(bst, dtest)

# early stopping
bst = XGBoost.xgboost(dtrain, 
    num_round = 100, 
    eval_metric = "rmse", 
    watchlist = OrderedDict(["train" => dtrain, "eval" => dtest]), 
    early_stopping_rounds = 5, 
    max_depth=6, 
    η=0.3
)
# get the best iteration and use it for prediction
y_predict = XGBoost.predict(bst, dtest, ntree_limit = bst.best_iteration)

bst = XGBoost.xgboost(dtrain, num_round=1; XGBoost.randomforest()...)

# we can also retain / use the best score (based on eval_metric) which is stored in the booster
println("Best RMSE from model training $(round((bst.best_score), digits = 8)).")

prediction_rounded = round.(Int, y_predict)

MLBase.errorrate(y_code[tt_pairs.test], prediction_rounded)
MLBase.confusmat(length(levels(y_code)), Array(y_code[tt_pairs.test] .+ 1), Array(prediction_rounded) .+ 1)

XGBoost.importancetable(bst)

# return AbstractTrees.jl compatible tree objects describing the model
bst_tree = XGBoost.trees(bst)

AbstractTrees.repr_tree(bst_tree)
AbstractTrees.print_tree(bst_tree)

# ---------------------------------------------------------------------------- #
#                            MLJXGBoostInterface.jl                            #
# ---------------------------------------------------------------------------- #
using Sole
using SoleXplorer
using Random, StatsBase, JLD2, DataFrames
using XGBoost, CategoricalArrays, OrderedCollections, MLBase, AbstractTrees
using MLJ, MLJBase, MLJXGBoostInterface

filename = "examples/respiratory_Pneumonia.jld2"
df = jldopen(filename)
X, y = df["X"], df["y"]
rng = Random.Xoshiro(1)
train_seed = 11;
model_name = :decision_tree
features = catch9
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = SoleXplorer.get_model(model_name)
ds = SoleXplorer.preprocess_dataset(X, y, model)

plain_classifier = MLJXGBoostInterface.XGBoostClassifier(
    # ds.tt.train,
    num_round = 100, 
    eval_metric = ["rmse"], 
    watchlist = OrderedDict(["train" => ds.tt.train, "eval" => ds.tt.test]), 
    early_stopping_rounds = 5, 
    max_depth=6, 
    # η=0.3
)    
# num_round=5, max_depth=6, objective="reg:squarederror")

model = MLJ.machine(plain_classifier, ds.X[ds.tt.train, :], ds.y[ds.tt.train])
fit!(model, verbosity=0)
