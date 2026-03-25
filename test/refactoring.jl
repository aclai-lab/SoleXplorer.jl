using Test
using SoleXplorer
const SX = SoleXplorer

using DataTreatments
const DT = DataTreatments

using MLJ
using DataFrames, Random

Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

natopsloader = SX.NatopsLoader()
Xts, yts = SX.load(natopsloader)

# ---------------------------------------------------------------------------- #
dt = DT.load_dataset(Xc, yc)

@test is_tabular(dt) == true
@test is_multidim(dt) == false

dt = DT.load_dataset(Xts, yts)

@test is_tabular(dt) == true
@test is_multidim(dt) == false

dt = DT.load_dataset(Xts, yts, TreatmentGroup(aggrfunc=reducesize(win=(splitwindow(nwindows=3),)),))
@test is_tabular(dt) == false
@test is_multidim(dt) == true
# ---------------------------------------------------------------------------- #
# setup_dataset(Xc, yc)