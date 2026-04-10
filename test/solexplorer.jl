using Test
using SoleXplorer
const SX = SoleXplorer

# using DataTreatments
# const DT = DataTreatments

using MLJ
using DataFrames, Random

# ---------------------------------------------------------------------------- #
#                                load dataset                                  #
# ---------------------------------------------------------------------------- #

Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

natopsloader = SX.NatopsLoader()
Xts, yts = SX.load(natopsloader)

function create_image(seed::Int; n=6)
    Random.seed!(seed)
    rand(Float64, n, n)
end

function build_2_bananas_df()
    DataFrame(
        str_col=[missing, "blue", "green", "red", "blue"],
        sym_col=[:circle, :square, :triangle, :square, missing],
        img4=[i == 3 ? missing : create_image(i + 30) for i in 1:5],
        int_col=Int[10, 20, 30, 40, 50],
        V1=[NaN, missing, 3.0, 4.0, 5.6],
        V2=[2.5, missing, 4.5, 5.5, NaN],
        ts1=[
            NaN, collect(2.0:7.0),
            missing, collect(4.0:9.0),
            collect(5.0:10.0)
        ],
        V4=[4.1, NaN, NaN, 7.1, 5.5],
        V5=[5.0, 6.0, 7.0, 8.0, 1.8],
        ts2=[
            collect(2.0:0.5:5.5),
            collect(1.0:0.5:4.5),
            collect(3.0:0.5:6.5),
            collect(4.0:0.5:7.5),
            NaN
        ],
        ts3=[
            [1.0, 1.2, 1.2, 2.6, NaN, 4.0, 4.2],
            NaN, NaN, missing,
            [3.0, NaN, 4.4, missing, 5.8, 7.0, 7.2]
        ],
        V3=[3.2, 4.2, 5.2, missing, 2.4],
        ts4=[
            [6.0, 5.2, missing, 4.4, 1.2, 3.6, 2.8],
            missing,
            [5.0, 4.2, NaN, 3.4, missing, 2.6, 1.8],
            [8.0, 7.2, missing, 6.4, NaN, 5.6, 4.8],
            [9.0, NaN, 8.2, missing, 7.4, 6.6, 5.8]
        ],
        img1=[create_image(i) for i in 1:5],
        cat_col=categorical(["small", "medium", missing, "small", "large"]),
        uint_col=UInt32[1, 2, 3, 4, 5],
        img2=[i == 1 ? NaN : create_image(i + 10) for i in 1:5],
        img3=[create_image(i + 20) for i in 1:5],
    )
end

function build_test_df()
    n = 20
    Random.seed!(42)
    DataFrame(
        str_col=rand([missing, "blue", "green", "red"], n),
        sym_col=rand([:circle, :square, :triangle, missing], n),
        img4=[i % 5 == 0 ? missing : create_image(i + 30) for i in 1:n],
        int_col=Int.(collect(10:10:10*n)),
        V1=collect(1.0:Float64(n)),
        V2=collect(1.0:Float64(n)) .* 2.5,
        ts1=[collect(Float64(i):Float64(i)+5) for i in 1:n],
        V4=collect(4.0:0.5:4.0+0.5*(n-1)),
        V5=collect(5.0:0.5:5.0+0.5*(n-1)),
        ts2=[collect(Float64(i):0.5:Float64(i)+3.5) for i in 1:n],
        ts3=[collect(Float64(i):0.2:Float64(i)+1.2) for i in 1:n],
        V3=collect(3.0:0.5:3.0+0.5*(n-1)),
        ts4=[collect(Float64(i)+6:-0.8:Float64(i)) for i in 1:n],
        img1=[create_image(i) for i in 1:n],
        cat_col=categorical(rand(["small", "medium", "large"], n)),
        uint_col=UInt32.(1:n),
        img2=[create_image(i + 10) for i in 1:n],
        img3=[create_image(i + 20) for i in 1:n],
    )
end

d_bananas = build_2_bananas_df()
tb_classif = ["classA", "classB", "classC", "classA", "classB"]
tb_regress = [1.0, 5.0, 10.0, 15.0, 20.0]

df = build_test_df()
t_classif = repeat(["classA", "classB", "classC", "classA", "classB"], 4)
t_regress = collect(1.0:1.0:20.0)  

# ---------------------------------------------------------------------------- #
#                        I'm easy like sunday morning                          #
# ---------------------------------------------------------------------------- #
modelc = solexplorer(Xc, yc)
@test modelc isa SX.ModelSet

modelr = solexplorer(Xr, yr)
@test modelc isa SX.ModelSet

modelts = solexplorer(Xts, yts)
@test modelc isa SX.ModelSet

modeldb = solexplorer(d_bananas, tb_classif, 
    TreatmentGroup(impute=(SX.LOCF(), SX.NOCB()))
)
@test modelc isa SX.ModelSet

modeldb = solexplorer(df, t_regress,
    TreatmentGroup(impute=(SX.LOCF(), SX.NOCB()))
)
@test modelc isa SX.ModelSet

# ---------------------------------------------------------------------------- #
#                         all models parametrizations                          #
# ---------------------------------------------------------------------------- #

# --- Classification models ---
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
            seed,
            measures=(SX.accuracy, log_loss, confusion_matrix, kappa)
        )
        @test m isa SX.ModelSet
    end

    # with tuning
    range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
    for tuning in [
        GridTuning(resolution=5,  resampling=CV(nfolds=3), range=range, measure=SX.accuracy),
        RandomTuning(n=5,         resampling=CV(nfolds=3), range=range, measure=SX.accuracy),
    ]
        m = solexplorer(
            Xc, yc;
            model=SX.DecisionTreeClassifier(),
            resampling=CV(nfolds=3, shuffle=true),
            seed=1,
            tuning=tuning,
            measures=(SX.accuracy, kappa)
        )
        @test m isa SX.ModelSet
    end

    # with rule extraction
    m = solexplorer(
        Xc, yc;
        model=SX.DecisionTreeClassifier(),
        resampling=CV(nfolds=3, shuffle=true),
        seed=1,
        extractor=SX.InTreesRuleExtractor(),
        measures=(SX.accuracy, log_loss, confusion_matrix, kappa)
    )
    @test m isa SX.ModelSet
end

@testset "RandomForestClassifier parametrizations" begin
    for (resampling, seed) in [
        (CV(nfolds=3, shuffle=true),               1),
        (CV(nfolds=5, shuffle=true),               42),
        (Holdout(fraction_train=0.75, shuffle=true), 7),
        (StratifiedCV(nfolds=4, shuffle=true),     99),
    ]
        m = solexplorer(
            Xc, yc;
            model=SX.RandomForestClassifier(),
            resampling=resampling,
            seed=seed,
            measures=(SX.accuracy, log_loss, confusion_matrix, kappa)
        )
        @test m isa SX.ModelSet
    end

    # with tuning
    range = SX.range(:n_trees; lower=10, upper=50)
    m = solexplorer(
        Xc, yc;
        model=SX.RandomForestClassifier(),
        resampling=CV(nfolds=3, shuffle=true),
        seed=1,
        tuning=GridTuning(resolution=3, resampling=CV(nfolds=3), range=range, measure=SX.accuracy),
        measures=(SX.accuracy, kappa)
    )
    @test m isa SX.ModelSet
end

@testset "AdaBoostStumpClassifier parametrizations" begin
    for (resampling, seed) in [
        (CV(nfolds=3, shuffle=true),               1),
        (Holdout(fraction_train=0.7, shuffle=true),  7),
        (StratifiedCV(nfolds=4, shuffle=true),     99),
    ]
        m = solexplorer(
            Xc, yc;
            model=SX.AdaBoostStumpClassifier(),
            resampling=resampling,
            seed=seed,
            measures=(SX.accuracy, log_loss, confusion_matrix, kappa)
        )
        @test m isa SX.ModelSet
    end
end

@testset "ModalDecisionTree classification parametrizations" begin
    for (resampling, seed) in [
        (CV(nfolds=3, shuffle=true),               1),
        (Holdout(fraction_train=0.7, shuffle=true),  7),
        (StratifiedCV(nfolds=4, shuffle=true),     99),
    ]
        m = solexplorer(
            Xc, yc;
            model=SX.ModalDecisionTree(),
            resampling=resampling,
            seed=seed,
            measures=(SX.accuracy,)
        )
        @test m isa SX.ModelSet
    end
end

@testset "ModalRandomForest classification parametrizations" begin
    for (resampling, seed) in [
        (CV(nfolds=3, shuffle=true),               1),
        (Holdout(fraction_train=0.75, shuffle=true), 7),
        (StratifiedCV(nfolds=4, shuffle=true),     99),
    ]
        m = solexplorer(
            Xc, yc;
            model=SX.ModalRandomForest(),
            resampling=resampling,
            seed=seed,
            measures=(SX.accuracy,)
        )
        @test m isa SX.ModelSet
    end
end

@testset "ModalAdaBoost classification parametrizations" begin
    for (resampling, seed) in [
        (CV(nfolds=3, shuffle=true),               1),
        (Holdout(fraction_train=0.7, shuffle=true),  7),
        (StratifiedCV(nfolds=4, shuffle=true),     99),
    ]
        m = solexplorer(
            Xc, yc;
            model=ModalAdaBoost(),
            resampling=resampling,
            seed=seed,
            measures=(SX.accuracy,)
        )
        @test m isa SX.ModelSet
    end
end

@testset "XGBoostClassifier parametrizations" begin
    for (resampling, seed) in [
        (CV(nfolds=3, shuffle=true),               1),
        (Holdout(fraction_train=0.7, shuffle=true),  7),
        (StratifiedCV(nfolds=4, shuffle=true),     99),
    ]
        m = solexplorer(
            Xc, yc;
            model=SX.XGBoostClassifier(),
            resampling=resampling,
            seed=seed,
            measures=(SX.accuracy, confusion_matrix, kappa)
        )
        @test m isa SX.ModelSet
    end

    # with early stopping
    m = solexplorer(
        Xc, yc;
        model=SX.XGBoostClassifier(early_stopping_rounds=10),
        resampling=CV(nfolds=3, shuffle=true),
        valid_ratio=0.2,
        seed=1,
        measures=(SX.accuracy, confusion_matrix)
    )
    @test m isa SX.ModelSet

    # with tuning
    range = SX.range(:num_round; lower=10, unit=10, upper=50)
    m = solexplorer(
        Xc, yc;
        model=SX.XGBoostClassifier(),
        resampling=CV(nfolds=3, shuffle=true),
        seed=1,
        tuning=GridTuning(resolution=3, resampling=CV(nfolds=3), range=range, measure=SX.accuracy),
        measures=(SX.accuracy, kappa)
    )
    @test m isa SX.ModelSet
end

# --- Regression models ---
@testset "DecisionTreeRegressor parametrizations" begin
    for (resampling, seed) in [
        (CV(nfolds=3, shuffle=true),               1),
        (CV(nfolds=5, shuffle=true),               42),
        (Holdout(fraction_train=0.7, shuffle=true),  7),
    ]
        m = solexplorer(
            Xr, yr;
            model=SX.DecisionTreeRegressor(),
            resampling=resampling,
            seed=seed,
            measures=(rms, l1, l2, mae, mav)
        )
        @test m isa SX.ModelSet
    end

    # with tuning
    range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
    for tuning in [
        GridTuning(resolution=5, resampling=CV(nfolds=3), range=range, measure=rms),
        RandomTuning(n=5,        resampling=CV(nfolds=3), range=range, measure=rms),
    ]
        m = solexplorer(
            Xr, yr;
            model=SX.DecisionTreeRegressor(),
            resampling=CV(nfolds=3, shuffle=true),
            seed=1,
            tuning=tuning,
            measures=(rms, mae)
        )
        @test m isa SX.ModelSet
    end
end

@testset "RandomForestRegressor parametrizations" begin
    for (resampling, seed) in [
        (CV(nfolds=3, shuffle=true),               1),
        (CV(nfolds=5, shuffle=true),               42),
        (Holdout(fraction_train=0.75, shuffle=true), 7),
    ]
        m = solexplorer(
            Xr, yr;
            model=SX.RandomForestRegressor(),
            resampling=resampling,
            seed=seed,
            measures=(rms, l1, l2, mae, mav)
        )
        @test m isa SX.ModelSet
    end

    # with tuning
    range = SX.range(:n_trees; lower=10, upper=50)
    m = solexplorer(
        Xr, yr;
        model=SX.RandomForestRegressor(),
        resampling=CV(nfolds=3, shuffle=true),
        seed=1,
        tuning=GridTuning(resolution=3, resampling=CV(nfolds=3), range=range, measure=rms),
        measures=(rms, mae)
    )
    @test m isa SX.ModelSet
end

@testset "XGBoostRegressor parametrizations" begin
    for (resampling, seed) in [
        (CV(nfolds=3, shuffle=true),               1),
        (Holdout(fraction_train=0.7, shuffle=true),  7),
    ]
        m = solexplorer(
            Xr, yr;
            model=SX.XGBoostRegressor(),
            resampling=resampling,
            seed=seed,
            measures=(rms, l1, l2, mae, mav)
        )
        @test m isa SX.ModelSet
    end

    # with early stopping
    m = solexplorer(
        Xr, yr;
        model=SX.XGBoostRegressor(early_stopping_rounds=10),
        resampling=CV(nfolds=3, shuffle=true),
        valid_ratio=0.2,
        seed=1,
        measures=(rms, mae)
    )
    @test m isa SX.ModelSet

    # with tuning
    range = SX.range(:num_round; lower=10, unit=10, upper=50)
    m = solexplorer(
        Xr, yr;
        model=SX.XGBoostRegressor(),
        resampling=CV(nfolds=3, shuffle=true),
        seed=1,
        tuning=GridTuning(resolution=3, resampling=CV(nfolds=3), range=range, measure=rms),
        measures=(rms, mae)
    )
    @test m isa SX.ModelSet
end
