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
    resampling=Holdout(;shuffle=true),
    rng=Xoshiro(1),   
)
solemc = train_test(dsc)

modelc = symbolic_analysis(
    dsc, solemc;
    extractor=InTreesRuleExtractor()
)
@test SX.rules(modelc) isa Vector{SX.DecisionSet}

modelc = symbolic_analysis(
    dsc, solemc;
    extractor=InTreesRuleExtractor(min_coverage=1.0)
)
@test SX.rules(modelc) isa Vector{SX.DecisionSet}

@test_throws MethodError  symbolic_analysis(
    dsc, solemc;
    extractor=InTreesRuleExtractor(;invalid=true)
)

# ---------------------------------------------------------------------------- #
#                           lumen rules extraction                             #
# ---------------------------------------------------------------------------- #
dsc = setup_dataset(
    Xc, yc;
    model=XGBoostClassifier(),
    resampling=Holdout(;shuffle=true),
    rng=Xoshiro(1),   
)
solemc = train_test(dsc)

modelc = symbolic_analysis(
    dsc, solemc;
    extractor=LumenRuleExtractor()
)
@test SX.rules(modelc) isa Vector{SX.LumenResult}

modelc = symbolic_analysis(
    dsc, solemc;
    extractor=LumenRuleExtractor(minimization_scheme=:mitespresso)
)
@test SX.rules(modelc) isa Vector{SX.LumenResult}

@test_throws MethodError  symbolic_analysis(
    dsc, solemc;
    extractor=LumenRuleExtractor(invalid=true)
)

dsc = setup_dataset(
    Xc, yc;
    model=RandomForestClassifier(n_trees=2),
    resampling=Holdout(;shuffle=true),
    rng=Xoshiro(1),   
)
solemc = train_test(dsc)

modelc = symbolic_analysis(
    dsc, solemc;
    extractor=LumenRuleExtractor()
)
@test SX.rules(modelc) isa Vector{SX.LumenResult}

modelc = symbolic_analysis(
    dsc, solemc;
    extractor=LumenRuleExtractor(minimization_scheme=:mitespresso)
)
@test SX.rules(modelc) isa Vector{SX.LumenResult}

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
    resampling=Holdout(;shuffle=true),
    rng=Xoshiro(1),   
)
solemc = train_test(dsc)

modelc = symbolic_analysis(
    dsc, solemc;
    extractor=BATreesRuleExtractor(dataset_name="Sole_Analysis")
)
@test SX.rules(modelc) isa Vector{SX.DecisionSet}

modelc = symbolic_analysis(
    dsc, solemc;
    extractor=BATreesRuleExtractor(dataset_name="Sole_Analysis", num_trees=5)
)
@test SX.rules(modelc) isa Vector{SX.DecisionSet}

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
    resampling=Holdout(;shuffle=true),
    rng=Xoshiro(1),   
)
solemc = train_test(dsc)

modelc = symbolic_analysis(
    dsc, solemc;
    extractor=RULECOSIPLUSRuleExtractor()
)
@test SX.rules(modelc) isa Vector{SX.DecisionSet}

@test_throws MethodError  symbolic_analysis(
    dsc, solemc;
    extractor=RULECOSIPLUSRuleExtractor(invalid=true)
)

# ---------------------------------------------------------------------------- #
#                           refne rules extraction                             #
# ---------------------------------------------------------------------------- #
dsc = setup_dataset(
    Xc, yc;
    model=RandomForestClassifier(n_trees=2),
    resampling=Holdout(;shuffle=true),
    rng=Xoshiro(1),   
)
solemc = train_test(dsc)

modelc = symbolic_analysis(
    dsc, solemc;
    extractor=REFNERuleExtractor(L=2)
)
@test SX.rules(modelc) isa Vector{SX.DecisionSet}

@test_throws MethodError  symbolic_analysis(
    dsc, solemc;
    extractor=REFNERuleExtractor(invalid=true)
)

# ---------------------------------------------------------------------------- #
#                          trepan rules extraction                             #
# ---------------------------------------------------------------------------- #
dsc = setup_dataset(
    Xc, yc;
    model=RandomForestClassifier(n_trees=2),
    resampling=Holdout(;shuffle=true),
    rng=Xoshiro(1),   
)
solemc = train_test(dsc)

modelc = symbolic_analysis(
    dsc, solemc;
    extractor=TREPANRuleExtractor()
)
@test SX.rules(modelc) isa Vector{SX.DecisionSet}

@test_throws MethodError  symbolic_analysis(
    dsc, solemc;
    extractor=TREPANRuleExtractor(invalid=true)
)

# ---------------------------------------------------------------------------- #
#                                    XGBoost                                   #
# ---------------------------------------------------------------------------- #
# dsc = setup_dataset(
#     Xc, yc;
#     model=XGBoostClassifier(),
#     resampling=Holdout(;shuffle=true),
#     rng=Xoshiro(1),   
# )
# solemc = train_test(dsc)

# modelc = symbolic_analysis(
#     dsc, solemc;
#     extractor=LumenRuleExtractor()
# )
# @test SX.rules(modelc) isa Vector{SX.LumenResult}