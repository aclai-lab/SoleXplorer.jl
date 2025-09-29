# ---------------------------------------------------------------------------- #
# Usage Example: Bronze
# I have a dataset composed of a matrix (or dataframe) of measures, and a vector of labels.
# I don't know if there's something interesting there, let's check it out...

using SoleXplorer, MLJ, JLD2

data_path = joinpath(@__DIR__, "respiratory_pneumonioa.jld2")
data  = JLD2.load(data_path)
X, y = data["X"], MLJ.CategoricalArray{String,1,UInt32}(data["y"])

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
    # model=XGBoostClassifier(max_depth=3, early_stopping_rounds=20),
    model=XGBoostClassifier(),
    seed=123,
    resampling=CV(nfolds=5, shuffle=true),
    tuning=GridTuning(resolution=5, resampling=CV(nfolds=5), range=range, measure=accuracy, repeats=5),
    # extractor=InTreesRuleExtractor(),
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

symbolic_analysis!(
    model,
    extractor=LumenRuleExtractor()
)

# ---------------------------------------------------------------------------- #
# Usage Example: Diamond