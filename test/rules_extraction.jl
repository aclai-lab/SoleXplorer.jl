using Test
using SoleXplorer
using MLJ
using DataFrames, Random
const SX = SoleXplorer

Xc, yc = @load_iris
Xc = DataFrame(Xc)

# ---------------------------------------------------------------------------- #
#                          in trees rules extraction                           #
# ---------------------------------------------------------------------------- #
dsc = setup_dataset(
    Xc, yc;
    model=DecisionTreeClassifier(),
    resample=Holdout(;shuffle=true),
    rng=Xoshiro(1),   
)
solemc = train_test(dsc)

modelc = symbolic_analysis(
    dsc, solemc;
    extractor=InTreesRuleExtractor()
)
@test rules(modelc) isa SX.DecisionSet

modelc = symbolic_analysis(
    dsc, solemc;
    extractor=InTreesRuleExtractor(min_coverage=0.25)
)
@test rules(modelc) isa SX.DecisionSet

@test_throws MethodError  symbolic_analysis(
    dsc, solemc;
    extractor=InTreesRuleExtractor(invalid=true)
)
