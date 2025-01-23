using Sole
using SoleXplorer
using Random, RDatasets

table = RDatasets.dataset("datasets", "iris")
y = table[:, :Species]
X = select(table, Not([:Species]));

features = [mean, minimum]

model_name = :xgboost
xgb = SoleXplorer.get_model(model_name; num_round=10, objective="multi:softmax")
ds = SoleXplorer.prepare_dataset(X, y, xgb)

SoleXplorer.modelfit!(xgb, ds; features=features)
xm = SoleXplorer.modeltest(xgb, ds)

# @show SoleXplorer.get_rules(dtree);
# @show SoleXplorer.get_predict(model, ds);
