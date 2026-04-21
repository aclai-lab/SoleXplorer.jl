using Test

using SoleXplorer
const SX = SoleXplorer

using MLJ
using DataFrames, Random, CSV

Xc, yc = @load_iris
Xc = DataFrame(Xc)

iris = DT.load_dataset(Xc, yc)

# natopsloader = SX.NatopsLoader()
# Xts, yts = SX.load(natopsloader)

# natops = DT.load_dataset(
#     Xts, yts,
#     TreatmentGroup(
#         dims=1,
#         aggrfunc=SX.aggregate(
#             # win=SX.adaptivewindow(nwindows=3, overlap=0.3),
#             # features=(maximum, mean)
#             win=SX.wholewindow(),
#             features=mean
#         )
#     )
# )

# ---------------------------------------------------------------------------- #
#                                random forest                                 #
# ---------------------------------------------------------------------------- #
@btime a=SX.solexplorer(
    iris,
    model=SX.RandomForestClassifier(n_trees=50,),
    # extractor=SX.LumenRuleExtractor(minimization_scheme=:abc),
    # extractor=SX.LumenRuleExtractor(),
    seed=42
)
#   210.685 s (727569802 allocations: 403.57 GiB)
# ModelSet{DataSet{RandomForestClassifier, Int64}}:
#   Dataset: DataSet{RandomForestClassifier, Int64}
#   Models:  1 symbolic models
#   Rules: 3 extracted rules per model
#   Measures:
#     Accuracy() = 0.9555555555555556
#     Kappa() = 0.9317147192716237

# ---------------------------------------------------------------------------- #
#                           random forest float32                              #
# ---------------------------------------------------------------------------- #
@btime SX.solexplorer(
    iris,
    model=SX.RandomForestClassifier(n_trees=50,),
    # extractor=SX.LumenRuleExtractor(minimization_scheme=:abc),
    extractor=SX.LumenRuleExtractor(float_type=Float32,),
    seed=42
)
#   96.081 s (325824597 allocations: 197.66 GiB)
# ModelSet{DataSet{RandomForestClassifier, Int64}}:
#   Dataset: DataSet{RandomForestClassifier, Int64}
#   Models:  1 symbolic models
#   Rules: 3 extracted rules per model
#   Measures:
#     Accuracy() = 0.9555555555555556
#     Kappa() = 0.9317147192716237

# ---------------------------------------------------------------------------- #
#                                   xgboost                                    #
# ---------------------------------------------------------------------------- #
@btime a=SX.solexplorer(
    iris,
    model=SX.XGBoostClassifier(num_round=100,),
    # extractor=SX.LumenRuleExtractor(minimization_scheme=:abc),
    # extractor=SX.LumenRuleExtractor(),
    rng=42
)
#   271.988 ms (1839800 allocations: 158.14 MiB)
# ModelSet{DataSet{XGBoostClassifier, Int64}}:
#   Dataset: DataSet{XGBoostClassifier, Int64}
#   Models:  1 symbolic models
#   Rules: 3 extracted rules per model
#   Measures:
#     Accuracy() = 0.9555555555555556
#     Kappa() = 0.9310872894333843

# ---------------------------------------------------------------------------- #
#                               xgboost float32                                #
# ---------------------------------------------------------------------------- #
@btime SX.solexplorer(
    iris,
    model=SX.XGBoostClassifier(num_round=100,),
    # extractor=SX.LumenRuleExtractor(minimization_scheme=:abc),
    extractor=SX.LumenRuleExtractor(float_type=Float32,),
    seed=42
)
#   284.528 ms (1840659 allocations: 133.86 MiB)
# ModelSet{DataSet{XGBoostClassifier, Int64}}:
#   Dataset: DataSet{XGBoostClassifier, Int64}
#   Models:  1 symbolic models
#   Rules: 3 extracted rules per model
#   Measures:
#     Accuracy() = 0.9555555555555556
#     Kappa() = 0.9310872894333843

# ---------------------------------------------------------------------------- #
#                       random decision list ensemble                          #
# ---------------------------------------------------------------------------- #
SX.solexplorer(
    iris,
    model=SX.RandomDecisionListClassifier(num_models=50,),
    # extractor=SX.LumenRuleExtractor(minimization_scheme=:abc),
    # extractor=SX.LumenRuleExtractor(),
    seed=42
)


rng = Xoshiro(42)
Xc, yc = @load_iris
Xc = DataFrame(Xc)

featurenames = Symbol.(names(Xc))
logiset = scalarlogiset(Xc; featurenames, allow_propositional=true)
ensemble_model = build_ensemble(logiset, yc, 50; featurenames, model_wrapper)

@btime lumen(ensemble_model);
# 422.738 s (567032367 allocations: 97.72 GiB)

# ---------------------------------------------------------------------------- #
#                   random decision list ensemble float32                      #
# ---------------------------------------------------------------------------- #
rng = Xoshiro(42)
Xc, yc = @load_iris
Xc = DataFrame(Xc)

featurenames = Symbol.(names(Xc))
logiset = scalarlogiset(Xc; featurenames, allow_propositional=true)
ensemble_model = build_ensemble(logiset, yc, 50; featurenames, model_wrapper)

@btime lumen(ensemble_model; float_type=Float32);
# 366.494 s (538308022 allocations: 52.31 GiB)


dataset_name = "banknote.csv"

# datatreatments parameters
rng = 42
treatment = SX.TreatmentGroup(
    impute=(SX.Interpolate(r=RoundNearest), SX.LOCF(), SX.NOCB()),
)
balance = (SX.SMOTENC(ratios=0.75; rng), SX.RandomUndersampler(;rng))
float_type = Float32
# solexplorer parameters
models = Dict(
    "RF" => SX.RandomForestClassifier(n_trees=25,),
    "XGB" => SX.XGBoostClassifier(num_round=50,),
    "DL" => SX.RandomDecisionListClassifier(num_models=25,)
)
resampling = SX.CV(; nfolds=10, shuffle=true, rng)
measures = (SX.Accuracy(), SX.FScore())
model = models["DL"]

filepath = joinpath(@__DIR__, "dataframes/", dataset_name)
df = CSV.read(filepath, DataFrame)
X = df[:,1:end-1]
y = df[:,end]

@info "load dataset..."
dt = SX.load_dataset(
    X, y,
    treatment;
    balance,
    float_type
)

@info "processing dataset..."
solemodel = SX.solexplorer(
    dt;
    model=SX.RandomForestClassifier(n_trees=25,),
    resampling,
    measures,
    rng
)
#   Measures:
#     Accuracy() = 0.9926229508196721
#     FScore(beta = 1.0, …) = 0.9921491777949815

function model_wrapper(X, y, w; featurenames, rng, iteration, kwargs...)
    ripperk(X, y; featurenames, max_k=1, rng, min_rule_coverage=3)
end

featurenames = Symbol.(names(X))
logiset = scalarlogiset(X; featurenames, allow_propositional=true)
ensemble_model = build_ensemble(logiset, string.(y), 50; featurenames, model_wrapper)
apply!(ensemble_model, logiset, string.(y))

predictions = ensemble_model.models[1].info.supporting_predictions
labels = string.(y)
accuracy = sum(predictions .== labels) / length(labels)
# 0.8112244897959183

model = ensemble_model.models[1]
model_preds = apply(model, X)

labels = string.(y)
accuracy = sum(model_preds .== labels) / length(labels)
# 0.8112244897959183
