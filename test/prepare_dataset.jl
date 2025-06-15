using Test
using MLJ, SoleXplorer
using DataFrames, Random

Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

# ---------------------------------------------------------------------------- #
#                        prepare dataset usage examples                        #
# ---------------------------------------------------------------------------- #
modelc, dsc = prepare_dataset(Xc, yc)
@test modelc isa SoleXplorer.Modelset
@test dsc    isa SoleXplorer.Dataset
modelr, dsr = prepare_dataset(Xr, yr)
@test modelr isa SoleXplorer.Modelset
@test dsr    isa SoleXplorer.Dataset

# dsc = prepare_dataset(
#     Xc, yc;
#     model=(;type=:decisiontree),
#     preprocess=(;train_ratio=0.9, rng=Xoshiro(1)),
#     measures=(log_loss, accuracy),
# )

# dsc = prepare_dataset(
#     Xc, yc;
#     model=(;type=:decisiontree),
#     preprocess=(;train_ratio=0.9, rng=Xoshiro(1)),
#     measures=(log_loss, accuracy),
# )