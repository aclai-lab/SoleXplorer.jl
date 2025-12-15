using Test
using SoleXplorer
const SX = SoleXplorer

using MLJ
using DataFrames, Random

Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

natopsloader = NatopsLoader()
Xts, yts = SX.load(natopsloader)

path = @__DIR__

# ---------------------------------------------------------------------------- #
#                                 save model                                   #
# ---------------------------------------------------------------------------- #
r1 = SX.range(:(oversampler.k), lower=3, upper=10)
r2 = SX.range(:(undersampler.min_ratios), lower=0.1, upper=0.9)

dsc = setup_dataset(
    Xc, yc;
    model=SX.DecisionTreeClassifier(max_depth=3),
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
    measures=(SX.accuracy, log_loss, kappa)
)
solesave(modelc; path, name="test1")

@test_throws ArgumentError solesave(modelc; path, name="test1")

# ---------------------------------------------------------------------------- #
#                                 load model                                   #
# ---------------------------------------------------------------------------- #
ds_name        = "soleds_test1.jld2"
solemodel_name = "solemodel_test1.jld2"
analysis_name  = "soleanalysis_test1.jld2"

dsc_loaded      = soleload(path, ds_name)
model_loaded    = soleload(path, solemodel_name)
analysis_loaded = soleload(path, analysis_name)

@test_throws ArgumentError soleload(path, "invalid")

@test dsc_loaded      isa PropositionalDataSet
@test model_loaded    isa SX.SoleModel
@test analysis_loaded isa ModelSet

# ---------------------------------------------------------------------------- #
#                                 cleanup                                      #
# ---------------------------------------------------------------------------- #
# Delete created test files
test_files = [
    joinpath(path, ds_name),
    joinpath(path, solemodel_name),
    joinpath(path, analysis_name)
]

for file in test_files
    if isfile(file)
        rm(file; force=true)
        @info "Deleted test file: $file"
    elseif isdir(file)
        rm(file; recursive=true, force=true)
        @info "Deleted test directory: $file"
    end
end