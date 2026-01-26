# ---------------------------------------------------------------------------- #
# Usage Example: Bronze
# I have a dataset composed of a matrix (or dataframe) of measures, and a vector of labels.
# I don't know if there's something interesting there, let's check it out...

using SoleXplorer, MLJ, JLD2, CategoricalArrays

data_path = joinpath(@__DIR__, "respiratory_pneumonia.jld2")
data  = JLD2.load(data_path)
X, y = data["X"], CategoricalArrays.CategoricalArray{String,1,UInt32}(data["y"])

model = symbolic_analysis(X, y, seed=123);
show_measures(model)

# Performance Measures:
#   Accuracy() = 0.81
#   Kappa() = 0.6

# ---------------------------------------------------------------------------- #
# Usage Example: Silver
# I would like to test various models, to see which one suites for my experiment...

model = symbolic_analysis(X, y; model=ModalRandomForest(), seed=123);
show_measures(model)

model = symbolic_analysis(X, y; model=XGBoostClassifier(), seed=123);
show_measures(model)

# Performance Measures:
#   Accuracy() = 0.86
#   Kappa() = 0.69

# ---------------------------------------------------------------------------- #
# Usage Example: Gold
# Now it's time to tweak hyperparameters too find the best setting for the choosen model...

range = SoleXplorer.range(:max_depth; lower=1, upper=10)
model = symbolic_analysis(
    X, y;
    model=XGBoostClassifier(),
    seed=123,
    resampling=CV(nfolds=5, shuffle=true),
    tuning=GridTuning(resolution=5, resampling=CV(nfolds=5), range=range, measure=accuracy, repeats=5),
    measures=(accuracy, log_loss, confusion_matrix, kappa)    
)
show_measures(model)

# Performance Measures:
#   Accuracy() = 0.86
#   LogLoss(tol = 2.22045e-16) = 5.15
#   ConfusionMatrix(levels = nothing, …) = ConfusionMatrix{2}([31 6; 4 29])
#   Kappa() = 0.71

# ---------------------------------------------------------------------------- #
# Usage Example: Platinum
# Would be nice to dig into Rules Extraction, no need to train the model again...

range = (
    SoleXplorer.range(:max_depth; lower=1, upper=3),
    SoleXplorer.range(:num_round; lower=1, upper=10))
model = symbolic_analysis(
    X, y;
    model=XGBoostClassifier(),
    seed=123,
    tuning=AdaptiveTuning(range=range, resampling=CV(nfolds=5), measure=accuracy, repeats=10),
    extractor=LumenRuleExtractor()
)

# ▣ (V3 < 0.0084) ∧ (V2 ≥ 0.0238) ∧ (V4 ≥ 0.0031) -> healthy
# ▣ (V3 ≥ 0.0087) ∧ (V5 < 0.0045) -> pneumonia

# ---------------------------------------------------------------------------- #
# Usage Example: diamond
# We've selected some rules that sounds interesting. Finally would be nice to see if there's some associations among them...

manual_p = Atom(ScalarCondition(VariableMin(3), ≥, 0.0087))
manual_q = Atom(ScalarCondition(VariableMin(5), <, 0.0045))
manual_r = Atom(ScalarCondition(VariableMax(4), <, 0.0031))
manual_lp = box(IA_L)(manual_p)
manual_lq = diamond(IA_L)(manual_q)
manual_lr = box(IA_L)(manual_r)

symbolic_analysis!(
    model,
    association=FPGrowth(
        Vector{Item}([manual_p, manual_q, manual_r]),
        [(gsupport, 0.1, 0.1)],
        [(gconfidence, 0.2, 0.2)],
    )
)
associations(model)

#  min[V3] ≥ 0.0087 => min[V5] < 0.0045
#  min[V5] < 0.0045 => min[V3] ≥ 0.0087
#  min[V5] < 0.0045 => max[V4] < 0.0031
#  max[V4] < 0.0031 => min[V5] < 0.0045