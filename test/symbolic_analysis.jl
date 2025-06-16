using Test
using MLJ, SoleXplorer
using DataFrames, Random
using SoleData

Xts, yts = SoleData.load_arff_dataset("NATOPS")

# ---------------------------------------------------------------------------- #
#                           propositional time series                          #
# ---------------------------------------------------------------------------- #
modelts = symbolic_analysis(
    Xts, yts;
    model=(;type=:xgboost),
    preprocess=(;rng=Xoshiro(1)),
    measures=(accuracy,)
)
@test modelts isa SoleXplorer.Modelset

# ---------------------------------------------------------------------------- #
#                               modal time series                              #
# ---------------------------------------------------------------------------- #
modelts = symbolic_analysis(
    Xts, yts;
    model=(;type=:modaldecisiontree),
    preprocess=(;rng=Xoshiro(1)),
    features=(minimum, maximum),
    measures=(accuracy,)
)
@test modelts isa SoleXplorer.Modelset