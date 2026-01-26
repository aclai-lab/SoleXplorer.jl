using CSV, JLD2, DataFrames
using StatsBase, Random, MLJ
using CategoricalArrays
using SoleXplorer

# ---------------------------------------------------------------------------- #
#                                  load data                                   #
# ---------------------------------------------------------------------------- #
data  = JLD2.load(joinpath(@__DIR__, "respiratory_pneumonia.jld2"))
Xc = data["X"]
# this is imperative: some algos accept only categorical value
# TODO automate in solexplorer if it's a CLabel ?
yc = CategoricalArrays.CategoricalArray{String,1,UInt32}(data["y"])

Xlight = Xc[:, 3:18]
Xlumen = Xc[:, 3:10]

# ---------------------------------------------------------------------------- #
#                                sole xplorer                                  #
# ---------------------------------------------------------------------------- #
dtc = symbolic_analysis(
    Xc, yc;
    model=DecisionTreeClassifier(),
    resampling=StratifiedCV(nfolds=20, shuffle=true),
    seed=12345,
    # extractor=InTreesRuleExtractor(),
    measures=(accuracy,)      
)

rfc = symbolic_analysis(
    Xlight, yc;
    model=RandomForestClassifier(n_trees=30),
    resampling=StratifiedCV(nfolds=20, shuffle=true),
    seed=12345,
    extractor=LumenRuleExtractor(),
    measures=(accuracy,)      
)

# ---------------------------------------------------------------------------- #
#                                   lumen                                      #
# ---------------------------------------------------------------------------- #
lfc = symbolic_analysis(
    Xlumen, yc;
    model=RandomForestClassifier(n_trees=2),
    resampling=Holdout(fraction_train=0.7, shuffle=true),
    seed=12345,
    extractor=LumenRuleExtractor(),
    measures=(accuracy,)      
)

lXc = symbolic_analysis(
    Xlight, yc;
    model=XGBoostClassifier(num_round=4),
    resampling=Holdout(fraction_train=0.7, shuffle=true),
    seed=12345,
    extractor=LumenRuleExtractor(),
    measures=(accuracy,)      
)

# test with actual MLJ model
Tree = @MLJ.load XGBoostClassifier pkg=XGBoost
tree = Tree(;num_round=4)

mlj = evaluate(
    tree, Xlight, yc;
    resampling=Holdout(fraction_train=0.7, shuffle=true, seed=12345),
    measures=[accuracy],
    per_observation=false,
    verbosity=0
)

@test lXc.measures.measures_values[1] == mlj.measurement[1]

# ---------------------------------------------------------------------------- #
#                              serialize results                               #
# ---------------------------------------------------------------------------- #
# save data using JLD2
jldsave("forest_juliacon2025.jld2"; X=rfc)

lumenresult = lfc.rules[1]
jldsave("lumen_randomforest_juliacon2025.jld2"; lumenresult)
lumenresult = lXc.rules[1]
jldsave("lumen_xgboost_juliacon2025.jld2"; lumenresult)

# ---------------------------------------------------------------------------- #
#                                  test data                                   #
# ---------------------------------------------------------------------------- #
data  = JLD2.load("forest_juliacon2025.jld2")
test_model = data["X"]
