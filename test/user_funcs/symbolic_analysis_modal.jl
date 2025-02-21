using Test
using Sole
import SoleXplorer as SX
using SoleXplorer
using Random, StatsBase, JLD2, DataFrames
using RDatasets

# ---------------------------------------------------------------------------- #
X, y = SoleData.load_arff_dataset("NATOPS")
train_seed = 11;
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

# ---------------------------------------------------------------------------- #
#                           modal symbolic analysis                            #
# ---------------------------------------------------------------------------- #

tree_results = symbolic_analysis(X, y; models=(type=:decisiontree,))
tree_results = symbolic_analysis(X, y; models=(type=:modaldecisiontree, features=[mean, cov]))
tree_results = symbolic_analysis(X, y; models=(type=:modaldecisiontree, features=[maximum], winparams=(type=adaptivewindow, nwindows=3)))


forest_results = symbolic_analysis(X, y; models=(type=:randomforest,))
forest_results = symbolic_analysis(X, y; models=(type=:modalrandomforest, features=[mean, cov]))
forest_results = symbolic_analysis(X, y; models=(type=:modalrandomforest, features=[maximum], winparams=(type=adaptivewindow, nwindows=3)))

results = symbolic_analysis(X, y; models=(type=:modaladaboost,))
