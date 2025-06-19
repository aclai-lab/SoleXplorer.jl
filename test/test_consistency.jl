using Test
using MLJ, SoleXplorer
using DataFrames, Random
const SX = SoleXplorer

Xc, yc = @load_iris
Xc = DataFrame(Xc)

# ---------------------------------------------------------------------------- #
#                         check train test partitions                          #
# ---------------------------------------------------------------------------- #
Tree = @load DecisionTreeClassifier pkg=DecisionTree
tree = Tree()

@test_nowarn begin
    for seed in 1:25
        for fraction_train in 0.5:0.1:0.9
            Random.seed!(1234)
            _, ds = prepare_dataset(
                Xc, yc;
                model=(;type=:decisiontree),
                # don't pass fraction_ratio to resample, it goes to preprocess
                resample = (type=Holdout, params=(;shuffle=true)),
                preprocess=(train_ratio=fraction_train, rng=Xoshiro(seed)),
            )

            Random.seed!(1234)
            mljm = evaluate(
                tree, Xc, yc;
                resampling=Holdout(;fraction_train, shuffle=true, rng=Xoshiro(seed)),
                per_observation=false,
                verbosity=0,
            )
            @assert ds.tt[1].test == mljm.train_test_rows[1][2]
        end
    end
end

@test_nowarn begin
    for seed in 1:25
        for nfolds in 2:25
            Random.seed!(1234)
            _, ds = prepare_dataset(
                Xc, yc;
                model=(;type=:decisiontree),
                resample = (type=CV, params=(;nfolds, shuffle=true)),
                preprocess=(;train_ratio=0.7, rng=Xoshiro(seed)),
            )

            Random.seed!(1234)
            mljm = evaluate(
                tree, Xc, yc;
                resampling=CV(;nfolds, shuffle=true, rng=Xoshiro(seed)),
                per_observation=false,
                verbosity=0,
            )

            @assert all(ds.tt[i].test == mljm.train_test_rows[i][2] for i in 1:nfolds)
        end
    end
end

@test_nowarn begin
    for seed in 1:25
        for nfolds in 2:25
            Random.seed!(1234)
            _, ds = prepare_dataset(
                Xc, yc;
                model=(;type=:decisiontree),
                resample = (type=StratifiedCV, params=(;nfolds, shuffle=true)),
                preprocess=(;train_ratio=0.7, rng=Xoshiro(seed)),
            )

            Random.seed!(1234)
            mljm = evaluate(
                tree, Xc, yc;
                resampling=StratifiedCV(;nfolds, shuffle=true, rng=Xoshiro(seed)),
                per_observation=false,
                verbosity=0,
            )

            @assert all(ds.tt[i].test == mljm.train_test_rows[i][2] for i in 1:nfolds)
        end
    end
end

@test_nowarn begin
    for seed in 1:25
        for nfolds in 2:25
            Random.seed!(1234)
            _, ds = prepare_dataset(
                Xc, yc;
                model=(;type=:decisiontree),
                resample = (type=TimeSeriesCV, params=(;nfolds)),
                preprocess=(;train_ratio=0.7, rng=Xoshiro(seed)),
            )

            Random.seed!(1234)
            mljm = evaluate(
                tree, Xc, yc;
                resampling=TimeSeriesCV(;nfolds),
                per_observation=false,
                verbosity=0,
            )

            @assert all(ds.tt[i].test == mljm.train_test_rows[i][2] for i in 1:nfolds)
        end
    end
end