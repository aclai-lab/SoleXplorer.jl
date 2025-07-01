using Test
using MLJ, SoleXplorer
using DataFrames, Random
using SoleData
const SX = SoleXplorer

Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

Xts, yts = SoleData.load_arff_dataset("NATOPS")

# ---------------------------------------------------------------------------- #
#                            classification models                             #
# ---------------------------------------------------------------------------- #
model, _, _ = symbolic_analysis(
    Xc, yc;
    model=(;type=:decisiontree),
    resample = (type=Holdout, params=(;shuffle=true)),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    measures=(log_loss, accuracy, kappa, confusion_matrix),
)
@test model isa SoleXplorer.Modelset

model, _, _ = symbolic_analysis(
    Xc, yc;
    model=(;type=:randomforest, params=(;n_trees=50)),
    resample = (type=Holdout, params=(;shuffle=true)),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    measures=(log_loss, accuracy, kappa, confusion_matrix),
)
@test model isa SoleXplorer.Modelset

model, _, _ = symbolic_analysis(
    Xc, yc;
    model=(;type=:adaboost),
    resample = (type=Holdout, params=(;shuffle=true)),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    measures=(log_loss, accuracy, kappa, confusion_matrix),
)
@test model isa SoleXplorer.Modelset

model, _, _ = symbolic_analysis(
    Xc, yc;
    model=(;type=:modaldecisiontree),
    resample = (type=Holdout, params=(;shuffle=true)),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    measures=(log_loss, accuracy, kappa, confusion_matrix),
)
@test model isa SoleXplorer.Modelset

model, _, _ = symbolic_analysis(
    Xc, yc;
    model=(;type=:modalrandomforest),
    resample = (type=Holdout, params=(;shuffle=true)),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    measures=(log_loss, accuracy, kappa, confusion_matrix),
)
@test model isa SoleXplorer.Modelset

model, _, _ = symbolic_analysis(
    Xc, yc;
    model=(;type=:modaladaboost),
    resample = (type=Holdout, params=(;shuffle=true)),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    measures=(log_loss, accuracy, kappa, confusion_matrix),
)
@test model isa SoleXplorer.Modelset

model, _, _ = symbolic_analysis(
    Xc, yc;
    model=(;type=:xgboost),
    resample = (type=Holdout, params=(;shuffle=true)),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    measures=(log_loss, accuracy, kappa, confusion_matrix),
)
@test model isa SoleXplorer.Modelset

# ---------------------------------------------------------------------------- #
#                classification models with tuning parameters                  #
# ---------------------------------------------------------------------------- #
model, _, _ = symbolic_analysis(
    Xc, yc;
    model=(;type=:decisiontree),
    resample = (type=Holdout, params=(;shuffle=true)),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    tuning=true,
    measures=(log_loss, accuracy, kappa, confusion_matrix),
)
@test model isa SoleXplorer.Modelset

model, _, _ = symbolic_analysis(
    Xc, yc;
    model=(;type=:randomforest, params=(;n_trees=50)),
    resample = (type=Holdout, params=(;shuffle=true)),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    tuning=true,
    measures=(log_loss, accuracy, kappa, confusion_matrix),
)
@test model isa SoleXplorer.Modelset

model, _, _ = symbolic_analysis(
    Xc, yc;
    model=(;type=:adaboost),
    resample = (type=Holdout, params=(;shuffle=true)),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    tuning=true,
    measures=(log_loss, accuracy, kappa, confusion_matrix),
)
@test model isa SoleXplorer.Modelset

model, _, _ = symbolic_analysis(
    Xc, yc;
    model=(;type=:modaldecisiontree),
    resample = (type=Holdout, params=(;shuffle=true)),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    tuning=true,
    measures=(log_loss, accuracy, kappa, confusion_matrix),
)
@test model isa SoleXplorer.Modelset

model, _, _ = symbolic_analysis(
    Xc, yc;
    model=(;type=:modalrandomforest),
    resample = (type=Holdout, params=(;shuffle=true)),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    tuning=true,
    measures=(log_loss, accuracy, kappa, confusion_matrix),
)
@test model isa SoleXplorer.Modelset

model, _, _ = symbolic_analysis(
    Xc, yc;
    model=(;type=:modaladaboost),
    resample = (type=Holdout, params=(;shuffle=true)),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    tuning=true,
    measures=(log_loss, accuracy, kappa, confusion_matrix),
)
@test model isa SoleXplorer.Modelset

model, _, _ = symbolic_analysis(
    Xc, yc;
    model=(;type=:xgboost),
    resample = (type=Holdout, params=(;shuffle=true)),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    tuning=true,
    measures=(log_loss, accuracy, kappa, confusion_matrix),
)
@test model isa SoleXplorer.Modelset

# ---------------------------------------------------------------------------- #
#                              regression models                               #
# ---------------------------------------------------------------------------- #
model, _, _ = symbolic_analysis(
    Xr, yr;
    model=(;type=:decisiontree),
    resample = (type=Holdout, params=(;shuffle=true)),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    measures=(rms, l1, l2, mae, mav),
)
@test model isa SoleXplorer.Modelset

model, _, _ = symbolic_analysis(
    Xr, yr;
    model=(;type=:randomforest),
    resample = (type=Holdout, params=(;shuffle=true)),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    measures=(rms, l1, l2, mae, mav),
)
@test model isa SoleXplorer.Modelset

model, _, _ = symbolic_analysis(
    Xr, yr;
    model=(;type=:xgboost),
    resample = (type=Holdout, params=(;shuffle=true)),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    measures=(rms, l1, l2, mae, mav),
)
@test model isa SoleXplorer.Modelset

# ---------------------------------------------------------------------------- #
#                  regression models with tuning parameters                    #
# ---------------------------------------------------------------------------- #
model, _, _ = symbolic_analysis(
    Xr, yr;
    model=(;type=:decisiontree),
    resample = (type=Holdout, params=(;shuffle=true)),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    tuning=true,
    measures=(rms, l1, l2, mae, mav),
)
@test model isa SoleXplorer.Modelset

model, _, _ = symbolic_analysis(
    Xr, yr;
    model=(;type=:randomforest),
    resample = (type=Holdout, params=(;shuffle=true)),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    tuning=true,
    measures=(rms, l1, l2, mae, mav),
)
@test model isa SoleXplorer.Modelset

model, _, _ = symbolic_analysis(
    Xr, yr;
    model=(;type=:xgboost),
    resample = (type=Holdout, params=(;shuffle=true)),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    tuning=true,
    measures=(rms, l1, l2, mae, mav),
)
@test model isa SoleXplorer.Modelset

# ---------------------------------------------------------------------------- #
#                               time series models                             #
# ---------------------------------------------------------------------------- #
model, _, _ = symbolic_analysis(
    Xts, yts;
    model=(;type=:modaldecisiontree),
    resample = (type=Holdout, params=(;shuffle=true)),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    measures=(log_loss, accuracy, kappa, confusion_matrix),
)
@test model isa SoleXplorer.Modelset

model, _, _ = symbolic_analysis(
    Xts, yts;
    model=(;type=:modalrandomforest),
    resample = (type=Holdout, params=(;shuffle=true)),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    measures=(log_loss, accuracy, kappa, confusion_matrix),
)
@test model isa SoleXplorer.Modelset

model, _, _ = symbolic_analysis(
    Xts, yts;
    model=(;type=:modaladaboost),
    resample = (type=Holdout, params=(;shuffle=true)),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    measures=(log_loss, accuracy, kappa, confusion_matrix),
)
@test model isa SoleXplorer.Modelset

# ---------------------------------------------------------------------------- #
#                  time series models with tuning parameters                   #
# ---------------------------------------------------------------------------- #
model, _, _ = symbolic_analysis(
    Xts, yts;
    model=(;type=:modaldecisiontree),
    resample = (type=Holdout, params=(;shuffle=true)),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    tuning=true,
    measures=(log_loss, accuracy, kappa, confusion_matrix),
)
@test model isa SoleXplorer.Modelset

model, _, _ = symbolic_analysis(
    Xts, yts;
    model=(;type=:modalrandomforest),
    resample = (type=Holdout, params=(;shuffle=true)),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    tuning=true,
    measures=(log_loss, accuracy, kappa, confusion_matrix),
)
@test model isa SoleXplorer.Modelset

model, _, _ = symbolic_analysis(
    Xts, yts;
    model=(;type=:modaladaboost),
    resample = (type=Holdout, params=(;shuffle=true)),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    tuning=true,
    measures=(log_loss, accuracy, kappa, confusion_matrix),
)
@test model isa SoleXplorer.Modelset
