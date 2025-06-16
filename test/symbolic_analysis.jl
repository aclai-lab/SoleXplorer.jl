using Test
using MLJ, SoleXplorer
using DataFrames, Random
using SoleData

Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

Xts, yts = SoleData.load_arff_dataset("NATOPS")

# ---------------------------------------------------------------------------- #
#                           propositional time series                          #
# ---------------------------------------------------------------------------- #
modelts = symbolic_analysis(
    Xts, yts;
    model=(;type=:xgboost),
    preprocess=(;rng=Xoshiro(1)),
    measures=(accuracy,)
)
@test modelts isa SoleXplorer.Modelset

# ---------------------------------------------------------------------------- #
#                               modal time series                              #
# ---------------------------------------------------------------------------- #
modelts = symbolic_analysis(
    Xts, yts;
    model=(;type=:modaldecisiontree),
    preprocess=(;rng=Xoshiro(1)),
    features=(minimum, maximum),
    measures=(accuracy,)
)
@test modelts isa SoleXplorer.Modelset

# ---------------------------------------------------------------------------- #
#             xgboost makewatchlist for early stopping technique               #
# ---------------------------------------------------------------------------- #
early_stop  = symbolic_analysis(
    Xc, yc; 
    model=(
        type=:xgboost_classifier,
        params=(
            num_round=100,
            max_depth=6,
            eta=0.1, 
            objective="multi:softprob",
            # early_stopping parameters
            early_stopping_rounds=10,
            watchlist=makewatchlist
        )
    ),
    # with early stopping a validation set is required
    preprocess=(valid_ratio = 0.3, rng=Xoshiro(1))
)