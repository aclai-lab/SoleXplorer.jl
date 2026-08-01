using Test
using SoleXplorer
const SX = SoleXplorer

using SoleData

using MLJ
using DataFrames, Random

# ---------------------------------------------------------------------------- #
#                                load dataset                                  #
# ---------------------------------------------------------------------------- #
Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

natopsloader = SoleData.Artifacts.NatopsLoader()
Xts, yts = SoleData.Artifacts.load(natopsloader)

function create_image(seed::Int; n=6)
    Random.seed!(seed)
    rand(Float64, n, n)
end

# ---------------------------------------------------------------------------- #
#                        I'm easy like sunday morning                          #
# ---------------------------------------------------------------------------- #
dsc = setup_dataset(Xc, yc; model=RandomForestClassifier(n_trees=20), rng=42)
modelc = solexplorer(dsc)
@test modelc isa SX.ModelSet

dsr = setup_dataset(Xr, yr; model=XGBoostRegressor(), rng=42)
modelr = solexplorer(dsr)
@test modelc isa SX.ModelSet

dsts = setup_dataset(Xts, yts; model=ModalDecisionTree(), rng=42)
modelts = solexplorer(dsts)
@test modelc isa SX.ModelSet

# ---------------------------------------------------------------------------- #
#                              usage examples                                  #
# ---------------------------------------------------------------------------- #
# classification models
@testset "DecisionTreeClassifier parametrizations" begin
    for (resampling, seed) in [
        (CV(nfolds=3, shuffle=true), 1),
        (CV(nfolds=5, shuffle=true), 42),
        (Holdout(fraction_train=0.7, shuffle=true), 7),
        (StratifiedCV(nfolds=4, shuffle=true), 99),
    ]
        m = solexplorer(
            Xc, yc;
            model=SX.DecisionTreeClassifier(),
            resampling,
            rng=seed,
            measures=(
                SX.Accuracy(),
                SX.LogLoss(),
                SX.ConfusionMatrix(),
                SX.Kappa()
            )
        )
        @test m isa SX.ModelSet
    end
end

@testset "RandomForestClassifier parametrizations" begin
    for (resampling, seed) in [
        (CV(nfolds=3, shuffle=true), 1),
        (CV(nfolds=5, shuffle=true), 42),
        (Holdout(fraction_train=0.75, shuffle=true), 7),
        (StratifiedCV(nfolds=4, shuffle=true), 99),
    ]
        m = solexplorer(
            Xc, yc;
            model=SX.RandomForestClassifier(),
            resampling=resampling,
            rng=seed,
            measures=(
                SX.Accuracy(),
                SX.LogLoss(),
                SX.ConfusionMatrix(),
                SX.Kappa()
            )
        )
        @test m isa SX.ModelSet
    end
end

@testset "AdaBoostStumpClassifier parametrizations" begin
    for (resampling, seed) in [
        (CV(nfolds=3, shuffle=true), 1),
        (Holdout(fraction_train=0.7, shuffle=true), 7),
        (StratifiedCV(nfolds=4, shuffle=true), 99),
    ]
        m = solexplorer(
            Xc, yc;
            model=SX.AdaBoostStumpClassifier(),
            resampling=resampling,
            rng=seed,
            measures=(
                SX.Accuracy(),
                SX.LogLoss(),
                SX.ConfusionMatrix(),
                SX.Kappa()
            )
        )
        @test m isa SX.ModelSet
    end
end

@testset "ModalDecisionTree classification parametrizations" begin
    for (resampling, seed) in [
        (CV(nfolds=3, shuffle=true), 1),
        (Holdout(fraction_train=0.7, shuffle=true), 7),
        (StratifiedCV(nfolds=4, shuffle=true), 99),
    ]
        m = solexplorer(
            Xts, yts;
            model=SX.ModalDecisionTree(),
            aggrfunc=reducesize(
                reducefunc=mean,
                win=(splitwindow(nwindows=5),)
            ),
            resampling=resampling,
            rng=seed,
            measures=(SX.Accuracy(),)
        )
        @test m isa SX.ModelSet
    end
end

@testset "ModalRandomForest classification parametrizations" begin
    for (resampling, seed) in [
        (CV(nfolds=3, shuffle=true), 1),
        (Holdout(fraction_train=0.75, shuffle=true), 7),
        (StratifiedCV(nfolds=4, shuffle=true), 99),
    ]
        m = solexplorer(
            Xts, yts;
            model=SX.ModalRandomForest(),
            aggrfunc=reducesize(
                reducefunc=mean,
                win=(splitwindow(nwindows=3),)
            ),
            resampling=resampling,
            rng=seed,
            measures=(SX.Accuracy(),)
        )
        @test m isa SX.ModelSet
    end
end

@testset "ModalAdaBoost classification parametrizations" begin
    for (resampling, seed) in [
        (CV(nfolds=3, shuffle=true), 1),
        (Holdout(fraction_train=0.7, shuffle=true), 7),
        (StratifiedCV(nfolds=4, shuffle=true), 99),
    ]
        m = solexplorer(
            Xts, yts;
            model=ModalAdaBoost(),
            aggrfunc=reducesize(
                reducefunc=mean,
                win=(splitwindow(nwindows=5),)
            ),
            resampling=resampling,
            rng=seed,
            float_type=Float64,
            measures=(SX.Accuracy(),)
        )
        @test m isa SX.ModelSet
    end
end

@testset "XGBoostClassifier parametrizations" begin
    for (resampling, seed) in [
        (CV(nfolds=3, shuffle=true), 1),
        (Holdout(fraction_train=0.7, shuffle=true), 7),
        (StratifiedCV(nfolds=4, shuffle=true), 99),
    ]
        m = solexplorer(
            Xc, yc;
            model=SX.XGBoostClassifier(),
            resampling=resampling,
            rng=seed,
            measures=(SX.Accuracy(), SX.ConfusionMatrix(), SX.Kappa())
        )
        @test m isa SX.ModelSet
    end

    # with early stopping
    m = solexplorer(
        Xc, yc;
        model=SX.XGBoostClassifier(early_stopping_rounds=10),
        resampling=CV(nfolds=3, shuffle=true),
        valid_ratio=0.2,
        rng=42,
        measures=(SX.Accuracy(), SX.ConfusionMatrix())
    )
    @test m isa SX.ModelSet
end

# ---------------------------------------------------------------------------- #
# Regression models
@testset "DecisionTreeRegressor parametrizations" begin
    for (resampling, seed) in [
        (CV(nfolds=3, shuffle=true), 1),
        (CV(nfolds=5, shuffle=true), 42),
        (Holdout(fraction_train=0.7, shuffle=true), 7),
    ]
        m = solexplorer(
            Xr, yr;
            model=SX.DecisionTreeRegressor(),
            resampling=resampling,
            rng=seed,
            measures=(SX.RootMeanSquaredError(), SX.LPLoss())
        )
        @test m isa SX.ModelSet
    end
end

@testset "RandomForestRegressor parametrizations" begin
    for (resampling, seed) in [
        (CV(nfolds=3, shuffle=true), 1),
        (CV(nfolds=5, shuffle=true), 42),
        (Holdout(fraction_train=0.75, shuffle=true), 7),
    ]
        m = solexplorer(
            Xr, yr;
            model=SX.RandomForestRegressor(),
            resampling=resampling,
            rng=seed,
            measures=(SX.RootMeanSquaredError(), SX.LPLoss())
        )
        @test m isa SX.ModelSet
    end
end

@testset "XGBoostRegressor parametrizations" begin
    for (resampling, seed) in [
        (CV(nfolds=3, shuffle=true), 1),
        (Holdout(fraction_train=0.7, shuffle=true), 7),
    ]
        m = solexplorer(
            Xr, yr;
            model=SX.XGBoostRegressor(),
            resampling=resampling,
            rng=seed,
            measures=(SX.RootMeanSquaredError(), SX.LPLoss())
        )
        @test m isa SX.ModelSet
    end

    # with early stopping
    m = solexplorer(
        Xr, yr;
        model=SX.XGBoostRegressor(early_stopping_rounds=10),
        resampling=CV(nfolds=3, shuffle=true),
        valid_ratio=0.2,
        rng=42,
        measures=(SX.RootMeanSquaredError(), SX.LPLoss())
    )
    @test m isa SX.ModelSet
end
