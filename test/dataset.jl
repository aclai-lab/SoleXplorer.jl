using Test
using SoleXplorer
const SX = SoleXplorer

using SoleData

using MLJ
using DataFrames, Random

Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

natopsloader = SoleData.Artifacts.NatopsLoader()
Xts, yts = SoleData.Artifacts.load(natopsloader)

Xts, yts = SoleData.Artifacts.load(SoleData.Artifacts.LibrasLoader())

rng = Random.Xoshiro(42)
Ximg = DataFrame(
    [round.(rand(rng, Float32, 15, 15); digits = 2) for _ in 1:20, _ in 1:12],
    :auto
)
yimg = rand(rng, 1:3, 20)

# ---------------------------------------------------------------------------- #
#                           model type specification                           #
# ---------------------------------------------------------------------------- #
dsc = setup_dataset(
    Xc, yc;
    model=SX.DecisionTreeClassifier()
)
@test dsc isa SX.DataSet{SX.DecisionTreeClassifier}

dsc = setup_dataset(
    Xc, yc;
    model=SX.RandomForestClassifier()
)
@test dsc isa SX.DataSet{SX.RandomForestClassifier}

dsc = setup_dataset(
    Xc, yc;
    model=SX.AdaBoostStumpClassifier()
)
@test dsc isa SX.DataSet{SX.AdaBoostStumpClassifier}

dsr = setup_dataset(
    Xr, yr;
    model=SX.DecisionTreeRegressor()
)
@test dsr isa SX.DataSet{SX.DecisionTreeRegressor}

dsr = setup_dataset(
    Xr, yr;
    model=SX.RandomForestRegressor()
)
@test dsr isa SX.DataSet{SX.RandomForestRegressor}

dsc = setup_dataset(
    Xc, yc;
    model=SX.XGBoostClassifier()
)
@test dsc isa SX.DataSet{SX.XGBoostClassifier}

dsr = setup_dataset(
    Xr, yr;
    model=SX.XGBoostRegressor()
)
@test dsr isa SX.DataSet{SX.XGBoostRegressor}

dsts = setup_dataset(
    Xts, yts;
    model=SX.ModalDecisionTree(),
)
@test dsts isa SX.DataSet{SX.ModalDecisionTree}
@test dsts.mach.args[1].data[1,1] isa Vector

dsts = setup_dataset(
    Xts, yts;
    model=SX.ModalRandomForest(),
)
@test dsts isa SX.DataSet{SX.ModalRandomForest}
@test dsts.mach.args[1].data[1,1] isa Vector

dsts = setup_dataset(
    Xts, yts;
    model=SX.ModalAdaBoost(),
)
@test dsts isa SX.DataSet{SX.ModalAdaBoost}
@test dsts.mach.args[1].data[1,1] isa Vector

dsimg = setup_dataset(
    Ximg, yimg;
    model=SX.ModalDecisionTree(),
)
@test dsimg isa SX.DataSet{SX.ModalDecisionTree}
@test dsimg.mach.args[1].data[1,1] isa Matrix

dsimg = setup_dataset(
    Ximg, yimg;
    model=SX.ModalRandomForest(),
)
@test dsimg isa SX.DataSet{SX.ModalRandomForest}
@test dsimg.mach.args[1].data[1,1] isa Matrix

dsimg = setup_dataset(
    Ximg, yimg;
    model=SX.ModalAdaBoost(),
)
@test dsimg isa SX.DataSet{SX.ModalAdaBoost}
@test dsimg.mach.args[1].data[1,1] isa Matrix

# ---------------------------------------------------------------------------- #
#                     aggrfunc=reducesized parameter usage                     #
# ---------------------------------------------------------------------------- #
# windowing functions
dsts = setup_dataset(
    Xts, yts; model=SX.ModalDecisionTree(),
    aggrfunc=reducesize(win=(wholewindow()))
)
@test dsts isa SX.DataSet{SX.ModalDecisionTree}
@test dsts.mach.args[1].data[1,1] isa Vector

dsts = setup_dataset(
    Xts, yts; model=SX.ModalDecisionTree(),
    aggrfunc=reducesize(win=(splitwindow(nwindows=3)))
)
@test dsts isa SX.DataSet{SX.ModalDecisionTree}
@test dsts.mach.args[1].data[1,1] isa Vector

dsts = setup_dataset(
    Xts, yts; model=SX.ModalDecisionTree(),
    aggrfunc=reducesize(win=(adaptivewindow(nwindows=3, overlap=0.2)))
)
@test dsts isa SX.DataSet{SX.ModalDecisionTree}
@test dsts.mach.args[1].data[1,1] isa Vector

# ---------------------------------------------------------------------------- #
dsimg = setup_dataset(
    Ximg, yimg; model=SX.ModalDecisionTree(),
    aggrfunc=reducesize(win=(wholewindow()))
)
@test dsimg isa SX.DataSet{SX.ModalDecisionTree}
@test dsimg.mach.args[1].data[1,1] isa Matrix

dsimg = setup_dataset(
    Ximg, yimg; model=SX.ModalDecisionTree(),
    aggrfunc=reducesize(win=(splitwindow(nwindows=3)))
)
@test dsimg isa SX.DataSet{SX.ModalDecisionTree}
@test dsimg.mach.args[1].data[1,1] isa Matrix

dsimg = setup_dataset(
    Ximg, yimg; model=SX.ModalDecisionTree(),
    aggrfunc=reducesize(win=(adaptivewindow(nwindows=3, overlap=0.2)))
)
@test dsimg isa SX.DataSet{SX.ModalDecisionTree}
@test dsimg.mach.args[1].data[1,1] isa Matrix

# ---------------------------------------------------------------------------- #
# reducefunc
dsts = setup_dataset(
    Xts, yts; model=SX.ModalDecisionTree(),
    aggrfunc=reducesize(reducefunc=mean)
)
@test dsts isa SX.DataSet{SX.ModalDecisionTree}
@test dsts.mach.args[1].data[1,1] isa Vector

dsts = setup_dataset(
    Xts, yts; model=SX.ModalDecisionTree(),
    aggrfunc=reducesize(reducefunc=minimum)
)
@test dsts isa SX.DataSet{SX.ModalDecisionTree}
@test dsts.mach.args[1].data[1,1] isa Vector

dsts = setup_dataset(
    Xts, yts; model=SX.ModalDecisionTree(),
    aggrfunc=reducesize(reducefunc=maximum)
)
@test dsts isa SX.DataSet{SX.ModalDecisionTree}
@test dsts.mach.args[1].data[1,1] isa Vector

# ---------------------------------------------------------------------------- #
dsimg = setup_dataset(
    Ximg, yimg; model=SX.ModalDecisionTree(),
    aggrfunc=reducesize(reducefunc=mean)
)
@test dsimg isa SX.DataSet{SX.ModalDecisionTree}
@test dsimg.mach.args[1].data[1,1] isa Matrix

dsimg = setup_dataset(
    Ximg, yimg; model=SX.ModalDecisionTree(),
    aggrfunc=reducesize(reducefunc=minimum)
)
@test dsimg isa SX.DataSet{SX.ModalDecisionTree}
@test dsimg.mach.args[1].data[1,1] isa Matrix

dsimg = setup_dataset(
    Ximg, yimg; model=SX.ModalDecisionTree(),
    aggrfunc=reducesize(reducefunc=maximum)
)
@test dsimg isa SX.DataSet{SX.ModalDecisionTree}
@test dsimg.mach.args[1].data[1,1] isa Matrix

# ---------------------------------------------------------------------------- #
#                      aggrfunc=aggregate parameter usage                      #
# ---------------------------------------------------------------------------- #
# windowing functions
dsts = setup_dataset(
    Xts, yts; model=SX.DecisionTreeClassifier(),
    aggrfunc=SX.aggregate(win=(wholewindow()))
)
@test dsts isa SX.DataSet{SX.DecisionTreeClassifier}
@test dsts.mach.args[1].data[1,1] isa Real

dsts = setup_dataset(
    Xts, yts; model=SX.DecisionTreeClassifier(),
    aggrfunc=SX.aggregate(win=(splitwindow(nwindows=3)))
)
@test dsts isa SX.DataSet{SX.DecisionTreeClassifier}
@test dsts.mach.args[1].data[1,1] isa Real

dsts = setup_dataset(
    Xts, yts; model=SX.DecisionTreeClassifier(),
    aggrfunc=SX.aggregate(win=(adaptivewindow(nwindows=3, overlap=0.2)))
)
@test dsts isa SX.DataSet{SX.DecisionTreeClassifier}
@test dsts.mach.args[1].data[1,1] isa Real

# ---------------------------------------------------------------------------- #
dsimg = setup_dataset(
    Ximg, yimg; model=SX.DecisionTreeClassifier(),
    aggrfunc=SX.aggregate(win=(wholewindow()))
)
@test dsimg isa SX.DataSet{SX.DecisionTreeClassifier}
@test dsimg.mach.args[1].data[1,1] isa Real

dsimg = setup_dataset(
    Ximg, yimg; model=SX.DecisionTreeClassifier(),
    aggrfunc=SX.aggregate(win=(splitwindow(nwindows=3)))
)
@test dsimg isa SX.DataSet{SX.DecisionTreeClassifier}
@test dsimg.mach.args[1].data[1,1] isa Real

dsimg = setup_dataset(
    Ximg, yimg; model=SX.DecisionTreeClassifier(),
    aggrfunc=SX.aggregate(win=(adaptivewindow(nwindows=3, overlap=0.2)))
)
@test dsimg isa SX.DataSet{SX.DecisionTreeClassifier}
@test dsimg.mach.args[1].data[1,1] isa Real

# ---------------------------------------------------------------------------- #
# featuresets
dsts = setup_dataset(
    Xts, yts; model=SX.DecisionTreeClassifier(),
    aggrfunc=SX.aggregate(features=mean)
)
@test dsts isa SX.DataSet{SX.DecisionTreeClassifier}
@test dsts.mach.args[1].data[1,1] isa Real

dsts = setup_dataset(
    Xts, yts; model=SX.DecisionTreeClassifier(),
    aggrfunc=SX.aggregate(features=(maximum, minimum, mean))
)
@test dsts isa SX.DataSet{SX.DecisionTreeClassifier}
@test dsts.mach.args[1].data[1,1] isa Real

# ---------------------------------------------------------------------------- #
dsimg = setup_dataset(
    Ximg, yimg; model=SX.DecisionTreeClassifier(),
    aggrfunc=SX.aggregate(features=mean)
)
@test dsimg isa SX.DataSet{SX.DecisionTreeClassifier}
@test dsimg.mach.args[1].data[1,1] isa Real

dsimg = setup_dataset(
    Ximg, yimg; model=SX.DecisionTreeClassifier(),
    aggrfunc=SX.aggregate(features=(maximum, minimum, mean))
)
@test dsimg isa SX.DataSet{SX.DecisionTreeClassifier}
@test dsimg.mach.args[1].data[1,1] isa Real

# ---------------------------------------------------------------------------- #
#                           resamplig strategies                               #
# ---------------------------------------------------------------------------- #
dsc = setup_dataset(
    Xc, yc; model=SX.DecisionTreeClassifier(),
    resampling=CV(nfolds=3, shuffle=true),
)
@test dsc isa SX.DataSet{SX.DecisionTreeClassifier}
@test dsc.pinfo.type isa MLJ.CV

dsc = setup_dataset(
    Xc, yc; model=SX.DecisionTreeClassifier(),
    resampling=Holdout(fraction_train=0.7, shuffle=true),
)
@test dsc isa SX.DataSet{SX.DecisionTreeClassifier}
@test dsc.pinfo.type isa MLJ.Holdout

dsc = setup_dataset(
    Xc, yc; model=SX.DecisionTreeClassifier(),
    resampling=StratifiedCV(nfolds=4, shuffle=true),
)
@test dsc isa SX.DataSet{SX.DecisionTreeClassifier}
@test dsc.pinfo.type isa MLJ.StratifiedCV

dsc = setup_dataset(
    Xc, yc; model=SX.DecisionTreeClassifier(),
    resampling=TimeSeriesCV(nfolds=4),
)
@test dsc isa SX.DataSet{SX.DecisionTreeClassifier}
@test dsc.pinfo.type isa MLJ.TimeSeriesCV

# ---------------------------------------------------------------------------- #
#                               normalization                                  #
# ---------------------------------------------------------------------------- #
# propositional
dsc = setup_dataset(
    Xc, yc; model=SX.DecisionTreeClassifier(),
    norm=ZScore
)
@test dsc isa SX.DataSet{SX.DecisionTreeClassifier}
@test dsts.mach.args[1].data[1,1] isa Real

dsc = setup_dataset(
    Xc, yc; model=SX.DecisionTreeClassifier(),
    norm=MinMax
)
@test dsc isa SX.DataSet{SX.DecisionTreeClassifier}
@test dsts.mach.args[1].data[1,1] isa Real

dsc = setup_dataset(
    Xc, yc; model=SX.DecisionTreeClassifier(),
    norm=Center
)
@test dsc isa SX.DataSet{SX.DecisionTreeClassifier}
@test dsts.mach.args[1].data[1,1] isa Real

dsc = setup_dataset(
    Xc, yc; model=SX.DecisionTreeClassifier(),
    norm=Sigmoid
)
@test dsc isa SX.DataSet{SX.DecisionTreeClassifier}
@test dsts.mach.args[1].data[1,1] isa Real

dsc = setup_dataset(
    Xc, yc; model=SX.DecisionTreeClassifier(),
    norm=UnitPower
)
@test dsc isa SX.DataSet{SX.DecisionTreeClassifier}
@test dsts.mach.args[1].data[1,1] isa Real

dsc = setup_dataset(
    Xc, yc; model=SX.DecisionTreeClassifier(),
    norm=Scale
)
@test dsc isa SX.DataSet{SX.DecisionTreeClassifier}
@test dsts.mach.args[1].data[1,1] isa Real

dsc = setup_dataset(
    Xc, yc; model=SX.DecisionTreeClassifier(),
    norm=ScaleMad
)
@test dsc isa SX.DataSet{SX.DecisionTreeClassifier}
@test dsts.mach.args[1].data[1,1] isa Real

dsc = setup_dataset(
    Xc, yc; model=SX.DecisionTreeClassifier(),
    norm=ScaleFirst
)
@test dsc isa SX.DataSet{SX.DecisionTreeClassifier}
@test dsts.mach.args[1].data[1,1] isa Real

dsc = setup_dataset(
    Xc, yc; model=SX.DecisionTreeClassifier(),
    norm=PNorm1
)
@test dsc isa SX.DataSet{SX.DecisionTreeClassifier}
@test dsts.mach.args[1].data[1,1] isa Real

dsc = setup_dataset(
    Xc, yc; model=SX.DecisionTreeClassifier(),
    norm=PNormInf
)
@test dsc isa SX.DataSet{SX.DecisionTreeClassifier}
@test dsts.mach.args[1].data[1,1] isa Real

# ---------------------------------------------------------------------------- #
# time-series
dsts = setup_dataset(
    Xts, yts; model=SX.ModalDecisionTree(),
    norm=ZScore
)
@test dsts isa SX.DataSet{SX.ModalDecisionTree}
@test dsts.mach.args[1].data[1,1] isa Vector

dsts = setup_dataset(
    Xts, yts; model=SX.ModalDecisionTree(),
    aggrfunc=reducesize(win=(splitwindow(nwindows=3))),
    norm=MinMax
)
@test dsts isa SX.DataSet{SX.ModalDecisionTree}
@test dsts.mach.args[1].data[1,1] isa Vector

dsts = setup_dataset(
    Xts, yts; model=SX.ModalDecisionTree(),
    norm=Center
)
@test dsts isa SX.DataSet{SX.ModalDecisionTree}
@test dsts.mach.args[1].data[1,1] isa Vector

dsts = setup_dataset(
    Xts, yts; model=SX.ModalDecisionTree(),
    norm=Sigmoid
)
@test dsts isa SX.DataSet{SX.ModalDecisionTree}
@test dsts.mach.args[1].data[1,1] isa Vector

dsts = setup_dataset(
    Xts, yts; model=SX.ModalDecisionTree(),
    norm=UnitPower
)
@test dsts isa SX.DataSet{SX.ModalDecisionTree}
@test dsts.mach.args[1].data[1,1] isa Vector

dsts = setup_dataset(
    Xts, yts; model=SX.ModalDecisionTree(),
    norm=Scale
)
@test dsts isa SX.DataSet{SX.ModalDecisionTree}
@test dsts.mach.args[1].data[1,1] isa Vector

dsts = setup_dataset(
    Xts, yts; model=SX.ModalDecisionTree(),
    norm=ScaleMad
)
@test dsts isa SX.DataSet{SX.ModalDecisionTree}
@test dsts.mach.args[1].data[1,1] isa Vector

dsts = setup_dataset(
    Xts, yts; model=SX.ModalDecisionTree(),
    norm=ScaleFirst
)
@test dsts isa SX.DataSet{SX.ModalDecisionTree}
@test dsts.mach.args[1].data[1,1] isa Vector

dsts = setup_dataset(
    Xts, yts; model=SX.ModalDecisionTree(),
    norm=PNorm1
)
@test dsts isa SX.DataSet{SX.ModalDecisionTree}
@test dsts.mach.args[1].data[1,1] isa Vector

dsts = setup_dataset(
    Xts, yts; model=SX.ModalDecisionTree(),
    norm=PNormInf
)
@test dsts isa SX.DataSet{SX.ModalDecisionTree}
@test dsts.mach.args[1].data[1,1] isa Vector

# ---------------------------------------------------------------------------- #
# images
dsimg = setup_dataset(
    Ximg, yimg; model=SX.ModalDecisionTree(),
    norm=ZScore
)
@test dsimg isa SX.DataSet{SX.ModalDecisionTree}
@test dsimg.mach.args[1].data[1,1] isa Matrix

dsimg = setup_dataset(
    Ximg, yimg; model=SX.ModalDecisionTree(),
    aggrfunc=reducesize(win=(splitwindow(nwindows=3))),
    norm=MinMax
)
@test dsimg isa SX.DataSet{SX.ModalDecisionTree}
@test dsimg.mach.args[1].data[1,1] isa Matrix

dsimg = setup_dataset(
    Ximg, yimg; model=SX.ModalDecisionTree(),
    norm=Center
)
@test dsimg isa SX.DataSet{SX.ModalDecisionTree}
@test dsimg.mach.args[1].data[1,1] isa Matrix

dsimg = setup_dataset(
    Ximg, yimg; model=SX.ModalDecisionTree(),
    norm=Sigmoid
)
@test dsimg isa SX.DataSet{SX.ModalDecisionTree}
@test dsimg.mach.args[1].data[1,1] isa Matrix

dsimg = setup_dataset(
    Ximg, yimg; model=SX.ModalDecisionTree(),
    norm=UnitPower
)
@test dsimg isa SX.DataSet{SX.ModalDecisionTree}
@test dsimg.mach.args[1].data[1,1] isa Matrix

dsimg = setup_dataset(
    Ximg, yimg; model=SX.ModalDecisionTree(),
    norm=Scale
)
@test dsimg isa SX.DataSet{SX.ModalDecisionTree}
@test dsimg.mach.args[1].data[1,1] isa Matrix

dsimg = setup_dataset(
    Ximg, yimg; model=SX.ModalDecisionTree(),
    norm=ScaleMad
)
@test dsimg isa SX.DataSet{SX.ModalDecisionTree}
@test dsimg.mach.args[1].data[1,1] isa Matrix

dsimg = setup_dataset(
    Ximg, yimg; model=SX.ModalDecisionTree(),
    norm=ScaleFirst
)
@test dsimg isa SX.DataSet{SX.ModalDecisionTree}
@test dsimg.mach.args[1].data[1,1] isa Matrix

dsimg = setup_dataset(
    Ximg, yimg; model=SX.ModalDecisionTree(),
    norm=PNorm1
)
@test dsimg isa SX.DataSet{SX.ModalDecisionTree}
@test dsimg.mach.args[1].data[1,1] isa Matrix

dsimg = setup_dataset(
    Ximg, yimg; model=SX.ModalDecisionTree(),
    norm=PNormInf
)
@test dsimg isa SX.DataSet{SX.ModalDecisionTree}
@test dsimg.mach.args[1].data[1,1] isa Matrix

# ---------------------------------------------------------------------------- #
#                                  impute                                      #
# ---------------------------------------------------------------------------- #
df_prop = DataFrame([
    1.0  2.0  3.0;
    NaN  5.0  6.0;
    7.0  missing  9.0;
    1.0  2.0  3.0;
    4.0  5.0  6.0
], :auto)
target = ["a", "b", "a", "b", "a"]

dsc = setup_dataset(
    df_prop, target; model=SX.DecisionTreeClassifier(),
    impute=(LOCF(), NOCB())
)
@test dsc isa SX.DataSet{SX.DecisionTreeClassifier}

dsc = setup_dataset(
    df_prop, target; model=SX.DecisionTreeClassifier(),
    impute=(Interpolate())
)
@test dsc isa SX.DataSet{SX.DecisionTreeClassifier}

dsc = setup_dataset(
    df_prop, target; model=SX.DecisionTreeClassifier(),
    impute=(SVD(init=Substitute(), rank=0, maxiter=100, tol=1e-10)),
    float_type=Float64 # SVD cannot work with float32 datasets
)
@test dsc isa SX.DataSet{SX.DecisionTreeClassifier}

dsc = setup_dataset(
    df_prop, target; model=SX.DecisionTreeClassifier(),
    impute=(Substitute(statistic=mean))
)
@test dsc isa SX.DataSet{SX.DecisionTreeClassifier}

# ---------------------------------------------------------------------------- #
# time-series
df_ts = DataFrame(
    ts1 = Any[
        Union{Missing, Float64}[1.0, NaN, 3.0, 4.0],
        Union{Missing, Float64}[2.0, 3.0, missing, 5.0],
        missing,
        Union{Missing, Float64}[5.0, NaN, 7.0, 8.0],
    ],
    ts2 = Any[
        Union{Missing, Float64}[missing, 2.0, 3.0, NaN],
        NaN,
        Union{Missing, Float64}[5.0, missing, 7.0, 8.0],
        Union{Missing, Float64}[NaN, 2.0, 3.0, 4.0],
    ],
)
target = ["a", "b", "a", "b"]

dsts = setup_dataset(
    df_ts, target; model=SX.ModalDecisionTree(),
    impute=(LOCF(), NOCB())
)
@test dsts isa SX.DataSet{SX.ModalDecisionTree}

dsts = setup_dataset(
    df_ts, target; model=SX.ModalDecisionTree(),
    impute=(Substitute(statistic=mean))
)
@test dsts isa SX.DataSet{SX.ModalDecisionTree}

# ---------------------------------------------------------------------------- #
# images
df_img = DataFrame(
    img1 = Any[
        Matrix{Union{Missing, Float64}}([1.0 NaN; missing 4.0]),
        Matrix{Union{Missing, Float64}}([2.0 3.0; 4.0 missing]),
        Matrix{Union{Missing, Float64}}([NaN 6.0; 7.0 8.0]),
        Matrix{Union{Missing, Float64}}([9.0 missing; 11.0 12.0]),
    ],
    img2 = Any[
        Matrix{Union{Missing, Float64}}([missing 2.0; 3.0 NaN]),
        Matrix{Union{Missing, Float64}}([4.0 5.0; NaN 7.0]),
        Matrix{Union{Missing, Float64}}([8.0 9.0; 10.0 missing]),
        Matrix{Union{Missing, Float64}}([11.0 12.0; 13.0 14.0]),
    ],
)

dsimg = setup_dataset(
    df_img, target; model=SX.ModalDecisionTree(),
    aggrfunc=reducesize(win=(splitwindow(nwindows=2))),
    impute=(LOCF(), NOCB())
)
@test dsimg isa SX.DataSet{SX.ModalDecisionTree}

dsimg = setup_dataset(
    df_img, target; model=SX.ModalDecisionTree(),
    aggrfunc=reducesize(win=(splitwindow(nwindows=2))),
    impute=(Substitute(statistic=mean))
)
@test dsimg isa SX.DataSet{SX.ModalDecisionTree}

# ---------------------------------------------------------------------------- #
#                                 imbalance                                    #
# ---------------------------------------------------------------------------- #
# create imbalanced dataset
Ximb = vcat(Xc[1:25, :], Xc[51:100, :], Xc[101:135, :])
yimb = vcat(yc[1:25], yc[51:100], yc[101:135])

dsc = setup_dataset(
    Ximb, yimb; model=SX.DecisionTreeClassifier(),
    balance=SX.RandomOversampler()
)
@test dsc isa SX.DataSet{SX.DecisionTreeClassifier}

dsc = setup_dataset(
    Ximb, yimb; model=SX.DecisionTreeClassifier(),
    balance=SX.RandomWalkOversampler()
)
@test dsc isa SX.DataSet{SX.DecisionTreeClassifier}

dsc = setup_dataset(
    Ximb, yimb; model=SX.DecisionTreeClassifier(),
    balance=SX.ROSE()
)
@test dsc isa SX.DataSet{SX.DecisionTreeClassifier}

dsc = setup_dataset(
    Ximb, yimb; model=SX.DecisionTreeClassifier(),
    balance=SX.SMOTE(k=5)
)
@test dsc isa SX.DataSet{SX.DecisionTreeClassifier}

dsc = setup_dataset(
    Ximb, yimb; model=SX.DecisionTreeClassifier(),
    balance=SX.BorderlineSMOTE1()
)
@test dsc isa SX.DataSet{SX.DecisionTreeClassifier}

dsc = setup_dataset(
    Ximb, yimb; model=SX.DecisionTreeClassifier(),
    balance=SX.SMOTENC(k=5)
)
@test dsc isa SX.DataSet{SX.DecisionTreeClassifier}

dsc = setup_dataset(
    Ximb, yimb; model=SX.DecisionTreeClassifier(),
    balance=SX.RandomUndersampler()
)
@test dsc isa SX.DataSet{SX.DecisionTreeClassifier}

dsc = setup_dataset(
    Ximb, yimb; model=SX.DecisionTreeClassifier(),
    balance=SX.ClusterUndersampler()
)
@test dsc isa SX.DataSet{SX.DecisionTreeClassifier}

dsc = setup_dataset(
    Ximb, yimb; model=SX.DecisionTreeClassifier(),
    balance=SX.ENNUndersampler()
)
@test dsc isa SX.DataSet{SX.DecisionTreeClassifier}

dsc = setup_dataset(
    Ximb, yimb; model=SX.DecisionTreeClassifier(),
    balance=SX.TomekUndersampler()
)
@test dsc isa SX.DataSet{SX.DecisionTreeClassifier}

# ---------------------------------------------------------------------------- #
#                              rng propagation                                 #
# ---------------------------------------------------------------------------- #
dsc = setup_dataset(
    Xc, yc; model=SX.DecisionTreeClassifier(),
    resampling=CV(nfolds=10, shuffle=true),
    rng=1
)
@test dsc isa SX.DataSet{SX.DecisionTreeClassifier}
@test dsc.mach.model.rng isa Xoshiro
@test dsc.pinfo.rng isa Xoshiro

# ---------------------------------------------------------------------------- #
#                            validate modelsetup                               #
# ---------------------------------------------------------------------------- #
dsc = setup_dataset(
    Xc, yc; model=SX.DecisionTreeClassifier(;max_depth=5)
)
@test dsc isa SX.DataSet{SX.DecisionTreeClassifier}
@test dsc.mach.model.max_depth == 5

@test_throws UndefVarError setup_dataset(
    Xc, yc; model=Invalid(;max_depth=5)
)

@test_throws MethodError setup_dataset(
    Xc, yc;
    model=SX.DecisionTreeClassifier(;invalid=5)
)

@test_throws MethodError setup_dataset(
    Xc, yc; model=SX.DecisionTreeClassifier(),
    resampling=Holdout(fraction_train=0.5, shuffle=true),
    invalid=maximum
)

# ---------------------------------------------------------------------------- #
#                                  Base.show                                   #
# ---------------------------------------------------------------------------- #
@testset "Base.show tests for partition.jl" begin
    # Setup test data
    rng = Xoshiro(42)
    y = ["A", "B", "A", "B", "A", "B", "A", "B"]
    resampling = CV(nfolds=3)
    
    @testset "PartitionInfo show methods" begin
        pinfo = SX.PartitionInfo(resampling, 0.2, rng)
        
        # Test Base.show(io::IO, info::PartitionInfo)
        io = IOBuffer()
        show(io, pinfo)
        output = String(take!(io))
        
        @test occursin("PartitionInfo:", output)
        @test occursin("type:", output)
        @test occursin("valid_ratio:", output)
        @test occursin("rng:", output)
        @test occursin("0.2", output)
        
        # Test Base.show(io::IO, ::MIME"text/plain", info::PartitionInfo)
        io = IOBuffer()
        show(io, MIME("text/plain"), pinfo)
        plain_output = String(take!(io))
        
        @test plain_output == output  # Should be identical
    end
    
    @testset "PartitionIdxs show methods" begin
        # Create test PartitionIdxs
        train_idxs = [1, 2, 3, 4]
        valid_idxs = [5, 6]
        test_idxs = [7, 8]
        pidx = SX.PartitionIdxs(train_idxs, valid_idxs, test_idxs)
        
        # Test Base.show(io::IO, pidx::PartitionIdxs{T})
        io = IOBuffer()
        show(io, pidx)
        output = String(take!(io))
        
        @test occursin("PartitionIdxs{Int", output)
        @test occursin("Total samples: 8", output)
        @test occursin("Train: 4", output)
        @test occursin("Valid: 2", output)
        @test occursin("Test: 2", output)
        
        # Test Base.show(io::IO, ::MIME"text/plain", pidx::PartitionIdxs{T})
        io = IOBuffer()
        show(io, MIME("text/plain"), pidx)
        plain_output = String(take!(io))
        
        @test plain_output == output  # Should be identical
        
        # Test with empty valid set
        pidx_no_valid = SX.PartitionIdxs([1, 2, 3, 4, 5], Int[], [6, 7, 8])
        io = IOBuffer()
        show(io, pidx_no_valid)
        output_no_valid = String(take!(io))
        
        @test occursin("Total samples: 8", output_no_valid)
        @test occursin("Train: 5", output_no_valid)
        @test occursin("Valid: 0", output_no_valid)
        @test occursin("Test: 3", output_no_valid)
    end
    
    @testset "Integration test with partition function" begin
        # Test show methods with actual partition results
        pidxs, pinfo = SX.partition(y; resampling, valid_ratio=0.2, rng=rng)
        
        # Test PartitionInfo show
        io = IOBuffer()
        show(io, pinfo)
        pinfo_output = String(take!(io))
        @test occursin("PartitionInfo:", pinfo_output)
        @test occursin("CV", pinfo_output)
        
        # Test PartitionIdxs show for first fold
        io = IOBuffer()
        show(io, pidxs[1])
        pidx_output = String(take!(io))
        @test occursin("PartitionIdxs{Int", pidx_output)
        @test occursin("Total samples:", pidx_output)
        @test occursin("Train:", pidx_output)
        @test occursin("Valid:", pidx_output)
        @test occursin("Test:", pidx_output)
    end
end

@testset "pCV Tests" begin
    # Test constructor validation
    @test_throws ArgumentError pCV(nfolds=1, fraction_train=0.7)
    @test_throws ArgumentError pCV(nfolds=0, fraction_train=0.7)
    
    # Test default constructor
    cv = pCV()
    @test cv.nfolds == 6
    @test cv.fraction_train == 0.7
    
    # Test custom parameters
    cv = pCV(nfolds=10, fraction_train=0.6, shuffle=true, rng=Xoshiro(42))
    @test cv.nfolds == 10
    @test cv.fraction_train == 0.6
    @test cv.shuffle == true
    
    # Test train_test_pairs
    rows = 1:100
    pairs = MLJ.MLJBase.train_test_pairs(cv, rows)
    
    @test length(pairs) == 10  # nfolds
    
    for (train, test) in pairs
        # Check sizes roughly match fraction_train
        @test length(train) ≈ 60 atol=5
        @test length(test) ≈ 40 atol=5
        
        # Check no overlap
        @test isempty(intersect(train, test))
        
        # Check all indices covered
        @test length(union(train, test)) == 100
    end
    
    # Test reproducibility with same RNG
    cv1 = pCV(nfolds=5, fraction_train=0.7, shuffle=true, rng=Xoshiro(123))
    cv2 = pCV(nfolds=5, fraction_train=0.7, shuffle=true, rng=Xoshiro(123))
    
    pairs1 = MLJ.MLJBase.train_test_pairs(cv1, rows)
    pairs2 = MLJ.MLJBase.train_test_pairs(cv2, rows)
    
    @test pairs1 == pairs2
end
