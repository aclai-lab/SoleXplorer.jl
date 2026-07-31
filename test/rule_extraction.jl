using Test
using SoleXplorer
const SX = SoleXplorer

using SoleData
using SoleModels

using SolePostHoc

using MLJ
using DataFrames, Random

# ---------------------------------------------------------------------------- #
#                                load dataset                                  #
# ---------------------------------------------------------------------------- #
Xc, yc = @load_iris
Xc = DataFrame(Xc)

# ---------------------------------------------------------------------------- #
#                        I'm easy like sunday morning                          #
# ---------------------------------------------------------------------------- #
dsh = setup_dataset(Xc, yc; model=RandomForestClassifier(n_trees=20), rng=42)
modelh = solexplorer(dsh)
@test modelh isa SX.ModelSet
@test SX.get_sole(modelh) isa Vector{SX.AbstractModel}
@test length(SX.get_sole(modelh)) == 1
@test get_X(get_ds(modelh), :test) isa Vector{<:SubDataFrame}
@test length(get_X(get_ds(modelh), :test)) == 1
@test get_y(get_ds(modelh), :test) isa Vector
@test length(get_y(get_ds(modelh), :test)) == 1

dscv = setup_dataset(
    Xc,
    yc;
    model=RandomForestClassifier(n_trees=20),
    resampling=CV(nfolds=3, shuffle=true),
    rng=42
)
modelcv = solexplorer(dscv)
@test modelcv isa SX.ModelSet
@test SX.get_sole(modelcv) isa Vector{SX.AbstractModel}
@test length(SX.get_sole(modelcv)) == 3
@test get_X(get_ds(modelcv), :test) isa Vector{<:SubDataFrame}
@test length(get_X(get_ds(modelcv), :test)) == 3
@test get_y(get_ds(modelcv), :test) isa Vector
@test length(get_y(get_ds(modelcv), :test)) == 3

# ---------------------------------------------------------------------------- #
#                                    intrees                                   #
# ---------------------------------------------------------------------------- #
config = InTreesConfig(
    pruning=PruningConfig(
        prune_rules=true,
        decay_threshold=0.05,
        percentage_degradation=true,
        s=1.0e-6
    ),
    rule_selection=CBC(
        threshold=0.1,
        nsubfeatures=0,
        ntrees=50,
        partial_sampling=0.7,
        max_depth=10
    ),
    post_process=STEL(;
        min_coverage=0.01
    ),
    complexity_metric=:natoms,
    max_rules=0,
    dns=false,
    rng=Xoshiro(42)
)

function extract_rules(m::ModelSet, idx::Int=1)
    return (
        get_sole(m)[idx],
        scalarlogiset(get_X(get_ds(m), :test)[idx]; allow_propositional=true),
        get_y(get_ds(m), :test)[idx]
    )
end

intrees(config, extract_rules(modelh)...)
