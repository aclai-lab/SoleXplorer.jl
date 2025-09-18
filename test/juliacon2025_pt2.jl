using CSV, JLD2, DataFrames
using StatsBase, Random, MLJ
using SoleXplorer

# ---------------------------------------------------------------------------- #
#                                  load data                                   #
# ---------------------------------------------------------------------------- #
data  = JLD2.load(joinpath(@__DIR__, "respiratory_juliacon2025.jld2"))
Xc = data["X"]
# this is imperative: some algos accept only categorical value
# TODO automate in solexplorer if it's a CLabel ?
yc = MLJ.CategoricalArray{String,1,UInt32}(data["y"])

Xlight = Xc[:, 3:18]
Xlumen = Xc[:, 3:10]

# ---------------------------------------------------------------------------- #
#                                sole xplorer                                  #
# ---------------------------------------------------------------------------- #
dtc = symbolic_analysis(
    Xc, yc;
    model=DecisionTreeClassifier(),
    resample=StratifiedCV(nfolds=20, shuffle=true),
    rng=Xoshiro(12345),
    # extractor=InTreesRuleExtractor(),
    measures=(accuracy,)      
)

rfc = symbolic_analysis(
    Xlight, yc;
    model=RandomForestClassifier(n_trees=30),
    resample=StratifiedCV(nfolds=20, shuffle=true),
    rng=Xoshiro(12345),
    extractor=LumenRuleExtractor(),
    measures=(accuracy,)      
)

# ---------------------------------------------------------------------------- #
#                                   lumen                                      #
# ---------------------------------------------------------------------------- #
lfc = symbolic_analysis(
    Xlumen, yc;
    model=RandomForestClassifier(n_trees=2),
    resample=Holdout(fraction_train=0.7, shuffle=true),
    rng=Xoshiro(12345),
    extractor=LumenRuleExtractor(),
    measures=(accuracy,)      
)

# ---------------------------------------------------------------------------- #
#                              serialize results                               #
# ---------------------------------------------------------------------------- #
# save data using JLD2
jldsave("forest_juliacon2025.jld2"; X=rfc)

# save lumen random forest decision set
jldsave("lumen_randomforest_juliacon2025.jld2"; lr=rfc)

# ---------------------------------------------------------------------------- #
#                                  test data                                   #
# ---------------------------------------------------------------------------- #
data  = JLD2.load("forest_juliacon2025.jld2")
test_model = data["X"]
