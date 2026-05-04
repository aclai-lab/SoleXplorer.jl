using Test
using SoleXplorer
const SX = SoleXplorer

using MLJ
using DataFrames

Xc, yc = @load_iris
Xc = DataFrame(Xc)

solex = solexplorer(
    Xc, yc;
    model=SX.RandomForestClassifier(max_depth=5, n_trees=10),
    resampling=Holdout(;shuffle=true),
    rng=42,   
)

# ---------------------------------------------------------------------------- #
#                          in trees rules extraction                           #
# ---------------------------------------------------------------------------- #
solexplorer!(
    solex;
    extractor=InTreesRuleExtractor()
)
get_rules(solex)
@test get_rules(solex) isa Vector{SX.DecisionSet}

solexplorer!(
    solex;
    extractor=InTreesRuleExtractor(min_coverage=0.3)
)
get_rules(solex)
@test get_rules(solex) isa Vector{SX.DecisionSet}

@test_throws MethodError solexplorer!(
    solex;
    extractor=InTreesRuleExtractor(;invalid=true)
)

# ---------------------------------------------------------------------------- #
#                           lumen rules extraction                             #
# ---------------------------------------------------------------------------- #
solexplorer!(
    solex;
    extractor=LumenRuleExtractor()
)
get_rules(solex)
@test get_rules(solex) isa Vector{SX.DecisionSet}

# takes too long on randomforest
soledt = solexplorer(
    Xc, yc;
    model=SX.DecisionTreeClassifier(max_depth=5,),
    resampling=Holdout(;shuffle=true),
    rng=42,   
    extractor=LumenRuleExtractor(minimization_scheme=:mitespresso)
)
@test get_rules(soledt) isa Vector{SX.DecisionSet}

@test_throws MethodError solexplorer!(
    solex;
    extractor=LumenRuleExtractor(invalid=true)
)

# ---------------------------------------------------------------------------- #
#                          batrees rules extraction                            #
# ---------------------------------------------------------------------------- #
# remember to install g++, clang and make
# sudo apt update
# sudo apt install clang
# sudo apt install make
# sudo apt install build-essential
solexplorer!(
    solex;
    extractor=BATreesRuleExtractor(;dataset_name="Sole_Analysis")
)
@test get_rules(solex) isa Vector{SX.DecisionSet}

solexplorer!(
    solex;
    extractor=BATreesRuleExtractor(dataset_name="Sole_Analysis", num_trees=5)
)
@test get_rules(solex) isa Vector{SX.DecisionSet}

@test_throws MethodError solexplorer!(
    solex;
    extractor=BATreesRuleExtractor(;invalid=true)
)

# ---------------------------------------------------------------------------- #
#                         rulecosi rules extraction                            #
# ---------------------------------------------------------------------------- #
solexplorer!(
    solex;
    extractor=RULECOSIPLUSRuleExtractor()
)
@test get_rules(solex) isa Vector{SX.DecisionSet}

@test_throws MethodError  solexplorer!(
    solex;
    extractor=RULECOSIPLUSRuleExtractor(invalid=true)
)

# ---------------------------------------------------------------------------- #
#                           refne rules extraction                             #
# ---------------------------------------------------------------------------- #
solexplorer!(
    solex;
    extractor=REFNERuleExtractor(;L=2)
)
@test get_rules(solex) isa Vector{SX.DecisionSet}

@test_throws MethodError solexplorer!(
    solex;
    extractor=REFNERuleExtractor(invalid=true)
)

# ---------------------------------------------------------------------------- #
#                          trepan rules extraction                             #
# ---------------------------------------------------------------------------- #
solexplorer!(
    solex;
    extractor=TREPANRuleExtractor()
)
@test get_rules(solex) isa Vector{SX.DecisionSet}

@test_throws MethodError solexplorer!(
    solex;
    extractor=TREPANRuleExtractor(invalid=true)
)
