using Test
using Sole, ModalDecisionTrees
import SoleXplorer as SX
using SoleXplorer
using Random, StatsBase, JLD2, DataFrames
using RDatasets
using MLJ, MLJDecisionTreeInterface
using DecisionTree
using MLJModelInterface

# ---------------------------------------------------------------------------- #
X, y = SoleData.load_arff_dataset("NATOPS")
train_ratio = 0.8
shuffle = true
train_seed = 11
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

m1 = SX.AdaBoostModel()
m1.learn_method = m1.learn_method[1]
ds1 = prepare_dataset(X, y, m1)
c1 = get_model(m1, ds1)
mc1 = modelfit(m1, c1, ds1)
fp1 = MLJ.fitted_params(mc1)
r1 = modeltest(m1, mc1, ds1)
f1 = SX.get_predict(mc1, ds1)

m2 = SX.ModalAdaBoostModel()
m2.learn_method = m2.learn_method[1]
ds2 = prepare_dataset(X, y, m2)
c2 = get_model(m2, ds2)
mc2 = modelfit(m2, c2, ds2)
fp2 = MLJ.fitted_params(mc2)
rp2 = MLJ.report(mc2)
r2 = modeltest(m2, mc2, ds2)
f2 = SX.get_predict(mc2, ds2)

m3 = SX.ModalRandomForestModel()
m3.learn_method = m3.learn_method[1]
ds3 = prepare_dataset(X, y, m3)
c3 = get_model(m3, ds3)
mc3 = modelfit(m3, c3, ds3)
fp3 = MLJ.fitted_params(mc3)
rp3 = MLJ.report(mc3)
r3 = modeltest(m3, mc3, ds3)
f3 = SX.get_predict(mc3, ds3)

filename = "respiratory_Pneumonia.jld2"
filepath = joinpath(@__DIR__, filename)
df = jldopen(filepath)
X, y = df["X"], df["y"]
train_ratio = 0.8
shuffle = true
train_seed = 11
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

m1 = SX.AdaBoostModel()
m1.learn_method = m1.learn_method[1]
ds1 = prepare_dataset(X, y, m1)
c1 = get_model(m1, ds1)
mc1 = modelfit(m1, c1, ds1)
fp1 = MLJ.fitted_params(mc1)
r1 = modeltest(m1, mc1, ds1)
f1 = SX.get_predict(mc1, ds1)

m2 = SX.ModalAdaBoostModel()
m2.learn_method = m2.learn_method[1]
ds2 = prepare_dataset(X, y, m2)
c2 = get_model(m2, ds2)
mc2 = modelfit(m2, c2, ds2)
fp2 = MLJ.fitted_params(mc2)
rp2 = MLJ.report(mc2)
r2 = modeltest(m2, mc2, ds2)
f2 = SX.get_predict(mc2, ds2)

# m3 = SX.ModalRandomForestModel()
# m3.learn_method = m3.learn_method[1]
# ds3 = prepare_dataset(X, y, m3)
# c3 = get_model(m3, ds3)
# mc3 = modelfit(m3, c3, ds3)
# fp3 = MLJ.fitted_params(mc3)
# rp3 = MLJ.report(mc3)
# r3 = modeltest(m3, mc3, ds3)
# f3 = SX.get_predict(mc3, ds3)


