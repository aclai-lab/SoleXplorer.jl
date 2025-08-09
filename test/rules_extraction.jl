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
    extractor=InTreesRuleExtractor(min_coverage=1.0)
)
@test rules(modelc) isa SX.DecisionSet

@test_throws MethodError  symbolic_analysis(
    dsc, solemc;
    extractor=InTreesRuleExtractor(;invalid=true)
)

# ---------------------------------------------------------------------------- #
#                           lumen rules extraction                             #
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
    extractor=LumenRuleExtractor()
)
@test rules(modelc) isa SX.DecisionSet

modelc = symbolic_analysis(
    dsc, solemc;
    extractor=LumenRuleExtractor(minimization_scheme=:mitespresso)
)
@test rules(modelc) isa SX.DecisionSet

@test_throws MethodError  symbolic_analysis(
    dsc, solemc;
    extractor=LumenRuleExtractor(invalid=true)
)

dsc = setup_dataset(
    Xc, yc;
    model=RandomForestClassifier(n_trees=2),
    resample=Holdout(;shuffle=true),
    rng=Xoshiro(1),   
)
solemc = train_test(dsc)

modelc = symbolic_analysis(
    dsc, solemc;
    extractor=LumenRuleExtractor()
)
@test rules(modelc) isa SX.DecisionSet

modelc = symbolic_analysis(
    dsc, solemc;
    extractor=LumenRuleExtractor(minimization_scheme=:mitespresso)
)
@test rules(modelc) isa SX.DecisionSet

@test_throws MethodError  symbolic_analysis(
    dsc, solemc;
    extractor=LumenRuleExtractor(invalid=true)
)

# ---------------------------------------------------------------------------- #
#                          batrees rules extraction                            #
# ---------------------------------------------------------------------------- #
dsc = setup_dataset(
    Xc, yc;
    model=RandomForestClassifier(n_trees=2),
    resample=Holdout(;shuffle=true),
    rng=Xoshiro(1),   
)
solemc = train_test(dsc)

modelc = symbolic_analysis(
    dsc, solemc;
    extractor=BATreesRuleExtractor(dataset_name="Sole_Analysis")
)
@test rules(modelc) isa SX.DecisionSet

modelc = symbolic_analysis(
    dsc, solemc;
    extractor=BATreesRuleExtractor(dataset_name="Sole_Analysis", num_trees=5)
)
@test rules(modelc) isa SX.DecisionSet

@test_throws MethodError  symbolic_analysis(
    dsc, solemc;
    extractor=BATreesRuleExtractor(invalid=true)
)

# ---------------------------------------------------------------------------- #
#                         rulecosi rules extraction                            #
# ---------------------------------------------------------------------------- #
dsc = setup_dataset(
    Xc, yc;
    model=RandomForestClassifier(n_trees=2),
    resample=Holdout(;shuffle=true),
    rng=Xoshiro(1),   
)
solemc = train_test(dsc)

modelc = symbolic_analysis(
    dsc, solemc;
    extractor=RULECOSIPLUSRuleExtractor()
)
@test rules(modelc) isa SX.DecisionSet

@test_throws MethodError  symbolic_analysis(
    dsc, solemc;
    extractor=RULECOSIPLUSRuleExtractor(invalid=true)
)