using Test
using SoleXplorer
using MLJ
using DataFrames, Random
const SX = SoleXplorer

Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

natopsloader = NatopsLoader()
Xts, yts = SX.load(natopsloader)

path = @__DIR__

# ---------------------------------------------------------------------------- #
#                             save dataset setup                               #
# ---------------------------------------------------------------------------- #
r1 = SX.range(:(oversampler.k), lower=3, upper=10)
r2 = SX.range(:(undersampler.min_ratios), lower=0.1, upper=0.9)

dsc = setup_dataset(
    Xc, yc;
    model=DecisionTreeClassifier(max_depth=3),
    resampling=StratifiedCV(nfolds=5, shuffle=true),
    seed=11,
    balancing=(
        oversampler=SMOTENC(k=5, ratios=1.0),
        undersampler=TomekUndersampler(min_ratios=0.5)),
    tuning=GridTuning(goal=4, range=(r1,r2))
)
solesave(dsc; path, name="test1")

solemc = train_test(dsc)
solesave(solemc; path, name="test1.jld2")

modelc = symbolic_analysis(
    dsc, solemc,
    extractor=LumenRuleExtractor(minimization_scheme=:mitespresso),
    measures=(accuracy, log_loss, kappa)
)
solesave(modelc; path, name="test1")

@test_throws ArgumentError solesave(modelc; path, name="test1")

# ---------------------------------------------------------------------------- #
#                             load dataset setup                               #
# ---------------------------------------------------------------------------- #
name="test1"
path = @__DIR__
dsc_load = load(path, )