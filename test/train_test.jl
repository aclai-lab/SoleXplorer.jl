using Test
using MLJ, SoleXplorer
using DataFrames, Random

Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

# ---------------------------------------------------------------------------- #
#                        train and test usage examples                         #
# ---------------------------------------------------------------------------- #
# basic setup
modelc = train_test(Xc, yc)
@test modelc isa SoleXplorer.Modelset
modelr = train_test(Xr, yr)
@test modelr isa SoleXplorer.Modelset
