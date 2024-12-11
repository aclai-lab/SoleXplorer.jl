using Sole
using SoleXplorer
using Random, RDatasets
# using CategoricalArrays
# import MLJModelInterface as MMI
# import XGBoost as XGB
# using AbstractTrees
# import DecisionTree
# using MLJ

# X, y = SoleData.load_arff_dataset("NATOPS")
# train_seed = 11;
# rng = Random.Xoshiro(train_seed)
# Random.seed!(train_seed)

table = RDatasets.dataset("datasets", "iris")
y = table[:, :Species]
X = select(table, Not([:Species]));

features = [mean, minimum]

model_name = :xgboost
x = SoleXplorer.get_model(model_name; num_round=1)
ds = SoleXplorer.preprocess_dataset(X, y, x)

SoleXplorer.modelfit!(x, ds; features=features, rng=rng)
xm = SoleXplorer.modeltest(x, ds);

# @show SoleXplorer.get_rules(dtree);
# @show SoleXplorer.get_predict(model, ds);

###########################################################
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model_name = :decision_tree
d = SoleXplorer.get_model(model_name)
ds = SoleXplorer.preprocess_dataset(X, y, d)

SoleXplorer.modelfit!(d, ds; features=features, rng=rng)
# d = SoleXplorer.modeltest(d, ds);

# learn_method = (mach, X, y) -> (dt = solemodel(MLJ.fitted_params(mach).tree); apply!(dt, X, y); dt)
# learn_method = (mach, X, y) -> (dt = solemodel(MLJ.fitted_params(mach).tree); apply!(dt, X, y); dt)

# using MLJ

# df=MLJ.fitted_params(d.mach)

# function MMI.fitted_params(::XGBoostAbstractClassifier, fitresult)
#     raw_tree = fitresult[1]
#     encoding = get_encoding(fitresult[2])
#     features = fitresult[4]
#     classlabels = MLJDecisionTreeInterface.classlabels(encoding)
#     tree = DecisionTree.wrap(
#         _node_or_leaf(raw_tree),
#         (featurenames=features, classlabels),
#     )
#     (; tree, raw_tree, encoding, features)
# end

xx=XGB.trees(x.mach.fitresult[1])
dd=d.mach.fitresult[1]

AbstractTrees.print_tree(xx)
AbstractTrees.print_tree(dd)

DecisionTree.wrap(dd)

###########################################################àà

a=MLJ.fitted_params(d.mach).tree

b=MLJXGBoostInterface.solemodel(MLJ.fitted_params(x.mach)...)
bb=MLJ.fitted_params(x.mach)
aa=solemodel(a)