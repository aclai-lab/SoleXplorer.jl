using JLD2, Random, MLJ
using SoleXplorer

data  = JLD2.load("respiratory_juliacon2025.jld2")
Xc = data["X"]
yc = data["y"]

Xlight = Xc[:, 3:18]

# ---------------------------------------------------------------------------- #
#                                sole xplorer                                  #
# ---------------------------------------------------------------------------- #
modelc = symbolic_analysis(
    Xc, yc;
    model=DecisionTreeClassifier(),
    resample=StratifiedCV(nfolds=20, shuffle=true),
    rng=Xoshiro(12345),
    # extractor=InTreesRuleExtractor(),
    measures=(accuracy,)      
)

# modelc = symbolic_analysis(
#     Xlight, yc;
#     model=RandomForestClassifier(n_trees=30),
#     resample=StratifiedCV(nfolds=20, shuffle=true),
#     rng=Xoshiro(12345),
#     # extractor=InTreesRuleExtractor(),
#     measures=(accuracy,)      
# )

modelc = symbolic_analysis(
    Xc, yc;
    model=XGBoostClassifier(early_stopping_rounds=20),
    valid_ratio=0.2,
    resample=StratifiedCV(nfolds=20, shuffle=true),
    rng=Xoshiro(12345),
    measures=(accuracy,)      
)

Xd, yd = @load_iris
# Xd = DataFrame(Xd)
modelc = symbolic_analysis(
    Xd, yd;
    model=XGBoostClassifier(),
    resample=StratifiedCV(nfolds=20, shuffle=true),
    rng=Xoshiro(12345),
    measures=(accuracy,) 
);