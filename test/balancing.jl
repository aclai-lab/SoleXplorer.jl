using Test
using SoleXplorer
const SX = SoleXplorer

using MLJ
using DataFrames, Random

using Distributions

# ---------------------------------------------------------------------------- #
#                                load dataset                                  #
# ---------------------------------------------------------------------------- #
Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

natopsloader = SX.NatopsLoader()
Xts, yts = SX.load(natopsloader)

# ---------------------------------------------------------------------------- #
#                                  balancing                                   #
# ---------------------------------------------------------------------------- #
seed = 42
model = SX.DecisionTreeClassifier()
measures=(SX.accuracy, SX.kappa)
resampling=SX.StratifiedCV(nfolds=5, shuffle=true)

balancing = (
    oversampler=SX.BorderlineSMOTE1(),
    undersampler=SX.ClusterUndersampler()
)

m = solexplorer(
    Xc, yc;
    model,
    resampling,
    seed,
    balancing
)
@test m isa SX.ModelSet
@test m.ds.mach.model.oversampler isa SX.BorderlineSMOTE1
@test m.ds.mach.model.undersampler isa SX.ClusterUndersampler

balancing = (
    oversampler=SX.BorderlineSMOTE1(
        m=6,
        k=4,
        ratios=1.1
    ),
    undersampler=SX.ClusterUndersampler(
        mode="center",
        ratios=0.9,
        maxiter=75
    )
)

m = solexplorer(
    Xc, yc;
    model,
    resampling,
    seed,
    balancing
)
@test m isa SX.ModelSet
@test m.ds.mach.model.oversampler isa SX.BorderlineSMOTE1
@test m.ds.mach.model.undersampler isa SX.ClusterUndersampler
@test m.ds.mach.model.oversampler.m == 6
@test m.ds.mach.model.oversampler.k == 4
@test m.ds.mach.model.oversampler.ratios ≈ 1.1
@test m.ds.mach.model.undersampler.mode == "center"
@test m.ds.mach.model.undersampler.ratios ≈ 0.9
@test m.ds.mach.model.undersampler.maxiter == 75

# ---------------------------------------------------------------------------- #
seed = 42
model = SX.RandomForestClassifier()
measures=(SX.accuracy, SX.kappa)
resampling=SX.StratifiedCV(nfolds=5, shuffle=true)

balancing=(
    oversampler=SX.ENNUndersampler(k=7),
    undersampler=SX.ROSE()
)
m = solexplorer(
    Xc, yc;
    model,
    resampling,
    balancing,
    measures
)
@test m isa SX.ModelSet
@test m.ds.mach.model.oversampler isa SX.ENNUndersampler
@test m.ds.mach.model.undersampler isa SX.ROSE

balancing=(
    oversampler=SX.ROSE(),
    undersampler=SX.ENNUndersampler()
)
m = solexplorer(
    Xc, yc;
    model,
    resampling,
    balancing,
    measures
)
@test m isa SX.ModelSet
@test m.ds.mach.model.oversampler isa SX.ROSE
@test m.ds.mach.model.undersampler isa SX.ENNUndersampler

balancing=(
    oversampler=SX.ROSE(
        s=2.0,
        ratios=1.9
    ),
    undersampler=SX.ENNUndersampler(
        k=7,
        keep_condition="exists",
        # "mode" (default): the class of the point is one of the most
        # frequent classes of the neighbors (there may be many)
        # "exists": the point has at least one neighbor from the same class
        # "only mode": the class of the point is the single most frequent class
        # of the neighbors
        # "all": the class of the point is the same as all the neighbors
        min_ratios=0.7,
        force_min_ratios=true
    )
)
m = solexplorer(
    Xc, yc;
    model,
    resampling,
    balancing,
    measures
)
@test m isa SX.ModelSet
@test m.ds.mach.model.oversampler isa SX.ROSE
@test m.ds.mach.model.undersampler isa SX.ENNUndersampler
@test m.ds.mach.model.oversampler.s == 2.0
@test m.ds.mach.model.oversampler.ratios == 1.9
@test m.ds.mach.model.undersampler.k == 7
@test m.ds.mach.model.undersampler.keep_condition == "exists"
@test m.ds.mach.model.undersampler.min_ratios ==0.7
@test m.ds.mach.model.undersampler.force_min_ratios == true

# ---------------------------------------------------------------------------- #
m = solexplorer(
    Xc, yc;
    model=SX.AdaBoostStumpClassifier(),
    resampling=StratifiedCV(nfolds=5, shuffle=true),
    balancing=(
        oversampler=RandomOversampler(),
        undersampler=RandomUndersampler()),
    measures=(SX.accuracy, )
)
@test m isa SX.ModelSet
@test m.ds.mach.model.oversampler isa SX.BorderlineSMOTE1
@test m.ds.mach.model.undersampler isa SX.ClusterUndersampler

# ---------------------------------------------------------------------------- #
m = solexplorer(
    Xc, yc;
    model=SX.ModalDecisionTree(),
    resampling=StratifiedCV(nfolds=5, shuffle=true),
    balancing=(
        oversampler=RandomWalkOversampler(),
        undersampler=SMOTE()),
    measures=(SX.accuracy, )
)
@test m isa SX.ModelSet

# ---------------------------------------------------------------------------- #
m = solexplorer(
    Xc, yc;
    model=SX.ModalRandomForest(),
    resampling=StratifiedCV(nfolds=5, shuffle=true),
    balancing=(
        oversampler=SMOTE(),
        undersampler=RandomUndersampler()),
    measures=(SX.accuracy, )
)
@test m isa SX.ModelSet

# ---------------------------------------------------------------------------- #
m = solexplorer(
    Xc, yc;
    model=ModalAdaBoost(),
    resampling=StratifiedCV(nfolds=5, shuffle=true),
    balancing=(
        oversampler=SMOTENC(),
        undersampler=TomekUndersampler()),
    measures=(SX.accuracy, )
)
@test m isa SX.ModelSet

# ---------------------------------------------------------------------------- #
m = solexplorer(
    Xc, yc;
    model=SX.XGBoostClassifier(),
    resampling=StratifiedCV(nfolds=5, shuffle=true),
    balancing=(
        oversampler=SMOTENC(),
        undersampler=TomekUndersampler()),
    measures=(SX.accuracy, )
)
@test m isa SX.ModelSet
