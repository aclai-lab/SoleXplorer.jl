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

function create_image(seed::Int; n=6)
    Random.seed!(seed)
    rand(Float64, n, n)
end

function build_test_df()
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

df = build_test_df()
t_classif = ["classA", "classB", "classC", "classA", "classB"]
t_regress = [1.2, 3.4, 2.2, 4.8, 0.9]

# ---------------------------------------------------------------------------- #
ds = SX.setup_dataset(Xc, yc)
solem = SX._train_test(ds)
solexplorer(Xc, yc)

ds = SX.setup_dataset(Xr, yr; float_type=Float32)
solem = SX._train_test(ds)
solexplorer(Xr, yr; float_type=Float32)

ds = SX.setup_dataset(Xts, yts)
solem = SX._train_test(ds)
solexplorer(Xts, yts)

ds = SX.setup_dataset(
    Xts,
    yts,
    model=ModalDecisionTree(
        features=[mean, maximum]
    ),
    TreatmentGroup(aggrfunc=reducesize(win=(splitwindow(nwindows=3),)),)
)
solem = SX._train_test(ds)
solexplorer(
    Xts,
    yts,
    TreatmentGroup(aggrfunc=reducesize(win=(splitwindow(nwindows=3),)),);
    model=ModalDecisionTree(
        features=[mean, maximum]
    )
)

# if dataset has missing, should be imputed first TODO
@test_throws MethodError SX.setup_dataset(df)

# ---------------------------------------------------------------------------- #
dt = load_dataset(Xc, yc; float_type=Float32)
get_tabular(dt)
get_multidim(dt)

dt = load_dataset(Xr, yr; float_type=Float32)
get_tabular(dt)
get_multidim(dt)

dt = load_dataset(Xts, yts; float_type=Float32)
get_tabular(dt)
get_multidim(dt)

dt = load_dataset(
    Xts,
    TreatmentGroup(
        dims=1, aggrfunc=reducesize(win=(splitwindow(nwindows=3),)),
    );
    float_type=Float32
)
get_tabular(dt)
get_multidim(dt)

dt = load_dataset(Xc)

dt = load_dataset(
    df,
    TreatmentGroup(
        dims=1, aggrfunc=reducesize(win=(splitwindow(nwindows=3),)),
    );
    float_type=Float32
)
get_tabular(dt)
get_multidim(dt)

dt = load_dataset(
    df,
    TreatmentGroup(
        name_expr=["str_col", "sym_col", "cat_col", "int_col", "uint_col"],
    )
)
get_tabular(dt)
get_multidim(dt)

dt = load_dataset(
    df,
    TreatmentGroup(dims=1,
    )
)
get_tabular(dt)
get_multidim(dt)

dt = load_dataset(
    df,
    TreatmentGroup(dims=2,
    )
)
get_tabular(dt)
get_multidim(dt)

dt = load_dataset(
    df,
    TreatmentGroup(
        dims=2, aggrfunc=reducesize(win=(splitwindow(nwindows=3),)),
    );
    float_type=Float32
)
get_tabular(dt)
get_multidim(dt)