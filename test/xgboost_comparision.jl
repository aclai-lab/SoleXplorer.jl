using Test
using MLJ, SoleXplorer
using DataFrames, Random
using SoleData
using XGBoost

Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

Xts, yts = SoleData.load_arff_dataset("NATOPS")

# ---------------------------------------------------------------------------- #
#                             regression experiments                           #
# ---------------------------------------------------------------------------- #
bst = xgboost((Xr, yr), num_round=5, max_depth=6, objective="reg:squarederror")
ŷ = XGBoost.predict(bst, Xr)

soler = train_test(Xr, yr; model=(;type=:xgboost), preprocess=(;rng=Xoshiro(1)))
@test modelts isa SoleXplorer.Modelset
@test modelts.measures.measures isa Vector