using Test
using SoleXplorer
using DataFrames
using StatsBase: sample
using DecisionTree: load_data
using RDatasets
    
@testset "check utility: check_dataframe_type" begin
    df_valid = DataFrame(a = [1.0, 2.0], b = [3, 4])
    df_invalid = DataFrame(a = ["a", "b"], b = [1, 2])
    
    @test SoleXplorer.check_dataset_type(df_valid) == true
    @test SoleXplorer.check_dataset_type(df_invalid) == false
    @test SoleXplorer.check_dataset_type(Matrix(df_valid)) == true
    @test SoleXplorer.check_dataset_type(Matrix(df_invalid)) == false
end

@testset "check utility: hasnans" begin
    df = DataFrame(a = [1.0, 2.0], b = [3, 4])
    df_hasnans = DataFrame(a = [1.0, NaN], b = [3, 4])
    
    @test SoleXplorer.hasnans(df) == false
    @test SoleXplorer.hasnans(df_hasnans) == true
    @test SoleXplorer.hasnans(Matrix(df)) == false
    @test SoleXplorer.hasnans(Matrix(df_hasnans)) == true
end

# ---------------------------------------------------------------------------- #
#                                numeric dataset                               #
# ---------------------------------------------------------------------------- #
X, y = load_data("iris")
X = DataFrame(Float64.(X), :auto)
y = String.(y)
rng = Xoshiro(11)

# ---------------------------------------------------------------------------- #
#                               prepare_dataset                                #
# ---------------------------------------------------------------------------- #
no_parameters = prepare_dataset(X, y)
model_type = prepare_dataset(X, y; model=(type=:modaldecisiontree,))
parametrized_model_type = prepare_dataset(X, y; 
    model=(type=:xgboost,
            params=(
                num_round=20, 
                booster="gbtree", 
                eta=0.5,
                num_parallel_tree=10, 
                max_depth=8, 
            )
    )
)

reducefunc = prepare_dataset(X, y; model=(type=:modaldecisiontree,), reducefunc=median)

resample = prepare_dataset(X, y; resample=(type=CV,))
parametrized_resample = prepare_dataset(X, y; resample=(type=StratifiedCV, params=(nfolds=10,)))

win = prepare_dataset(X, y; win=(type=adaptivewindow,))
parametrized_win = prepare_dataset(X, y; win=(type=adaptivewindow, params=(nwindows = 3, relative_overlap = 0.1)))

features = prepare_dataset(X, y; features=(mean, maximum, entropy_pairs))
features = prepare_dataset(X, y; features=(catch9))

tuning = prepare_dataset(X, y; tuning=true)
rng_tuning = prepare_dataset(X, y; tuning=true, preprocess=(;rng))
parametrized_tuning = prepare_dataset(X, y;
    tuning=(
        method=(type=grid, params=(resolution=25,)), 
        params=(repeats=35, n=10),
        ranges=(
            SoleXplorer.range(:merge_purity_threshold, lower=0.1, upper=2.0),
            SoleXplorer.range(:feature_importance, values=[:impurity, :split])
        )
    ), 
    preprocess=(;rng)
)

preprocess = prepare_dataset(X, y; preprocess=(valid_ratio=0.5,))

# ---------------------------------------------------------------------------- #
#                                 train_test                                   #
# ---------------------------------------------------------------------------- #
no_parameters = train_test(X, y)
model_type = train_test(X, y; model=(type=:modaldecisiontree,))
parametrized_model_type = train_test(X, y; 
    model=(type=:xgboost,
            params=(
                num_round=20, 
                booster="gbtree", 
                eta=0.5,
                num_parallel_tree=10, 
                max_depth=8, 
            )
    )
)

reducefunc = train_test(X, y; model=(type=:modaldecisiontree,), reducefunc=median)

resample = train_test(X, y; resample=(type=CV,))
parametrized_resample = train_test(X, y; resample=(type=StratifiedCV, params=(nfolds=10,)))

win = train_test(X, y; win=(type=adaptivewindow,))
parametrized_win = train_test(X, y; win=(type=adaptivewindow, params=(nwindows = 3, relative_overlap = 0.1)))

features = train_test(X, y; features=(mean, maximum, entropy_pairs))
features = train_test(X, y; features=(catch9))

tuning = train_test(X, y; tuning=true)
rng_tuning = train_test(X, y; tuning=true, preprocess=(;rng))
parametrized_tuning = train_test(X, y;
    tuning=(
        method=(type=grid, params=(resolution=25,)), 
        params=(repeats=35, n=10),
        ranges=(
            SoleXplorer.range(:merge_purity_threshold, lower=0.1, upper=2.0),
            SoleXplorer.range(:feature_importance, values=[:impurity, :split])
        )
    ), 
    preprocess=(;rng)
)

model_check_1 = train_test(X, y; model=(type=:decisiontree,), tuning=true, preprocess=(;rng))
model_check_2 = train_test(X, y; model=(type=:randomforest,), tuning=true, preprocess=(;rng))
model_check_3 = train_test(X, y; model=(type=:adaboost,), tuning=true, preprocess=(;rng))
model_check_4 = train_test(X, y; model=(type=:modaldecisiontree,), tuning=true, preprocess=(;rng))
model_check_5 = train_test(X, y; model=(type=:modalrandomforest,), tuning=true, preprocess=(;rng))
model_check_6 = train_test(X, y; model=(type=:modaladaboost,), tuning=true, preprocess=(;rng))
model_check_7 = train_test(X, y; model=(type=:xgboost,), tuning=true, preprocess=(;rng))

parametrized_model_type = train_test(X, y; 
    model=(type=:xgboost,
            params=(
                num_round=20, 
                booster="gbtree", 
                eta=0.5,
                num_parallel_tree=10, 
                max_depth=8, 
            )
    )
)

early_stop  = train_test(X, y; 
    model=(type=:xgboost_classifier,
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
    preprocess=(valid_ratio = 0.7,)
)

preprocess = train_test(X, y; preprocess=(valid_ratio=0.5,))

# ---------------------------------------------------------------------------- #
#                             time series dataset                              #
# ---------------------------------------------------------------------------- #

X, y = load_arff_dataset("NATOPS")
num_cols_to_sample, num_rows_to_sample, rng = 10, 50, Xoshiro(11)
chosen_cols = sample(rng, 1:size(X, 2), num_cols_to_sample; replace=false)
chosen_rows = sample(rng, 1:size(X, 1), num_rows_to_sample; replace=false)
X = X[chosen_rows, chosen_cols]
y = y[chosen_rows]

# ---------------------------------------------------------------------------- #
#                               prepare_dataset                                #
# ---------------------------------------------------------------------------- #
no_parameters = prepare_dataset(X, y)
model_type = prepare_dataset(X, y; model=(type=:modaldecisiontree,))
parametrized_model_type = prepare_dataset(X, y; 
    model=(type=:xgboost,
            params=(
                num_round=20, 
                booster="gbtree", 
                eta=0.5,
                num_parallel_tree=10, 
                max_depth=8, 
            )
    )
)

reducefunc = prepare_dataset(X, y; model=(type=:modaldecisiontree,), reducefunc=median)

resample = prepare_dataset(X, y; resample=(type=CV,))
parametrized_resample = prepare_dataset(X, y; resample=(type=StratifiedCV, params=(nfolds=10,)))

win = prepare_dataset(X, y; win=(type=adaptivewindow,))
parametrized_win = prepare_dataset(X, y; win=(type=adaptivewindow, params=(nwindows = 3, relative_overlap = 0.1)))

features = prepare_dataset(X, y; features=(mean, maximum, entropy_pairs))
features = prepare_dataset(X, y; features=(catch9))

tuning = prepare_dataset(X, y; tuning=true)
rng_tuning = prepare_dataset(X, y; tuning=true, preprocess=(;rng))
parametrized_tuning = prepare_dataset(X, y;
    tuning=(
        method=(type=grid, params=(resolution=25,)), 
        params=(repeats=35, n=10),
        ranges=(
            SoleXplorer.range(:merge_purity_threshold, lower=0.1, upper=2.0),
            SoleXplorer.range(:feature_importance, values=[:impurity, :split])
        )
    ), 
    preprocess=(;rng)
)

preprocess = prepare_dataset(X, y; preprocess=(valid_ratio=0.5,))

# ---------------------------------------------------------------------------- #
#                                 train_test                                   #
# ---------------------------------------------------------------------------- #
no_parameters = train_test(X, y)
model_type = train_test(X, y; model=(type=:modaldecisiontree,))
parametrized_model_type = train_test(X, y; 
    model=(type=:xgboost,
            params=(
                num_round=20, 
                booster="gbtree", 
                eta=0.5,
                num_parallel_tree=10, 
                max_depth=8, 
            )
    )
)

reducefunc = train_test(X, y; model=(type=:modaldecisiontree,), reducefunc=median)

resample = train_test(X, y; resample=(type=CV,))
parametrized_resample = train_test(X, y; resample=(type=StratifiedCV, params=(nfolds=10,)))

win = train_test(X, y; win=(type=adaptivewindow,))
parametrized_win = train_test(X, y; win=(type=adaptivewindow, params=(nwindows = 3, relative_overlap = 0.1)))

features = train_test(X, y; features=(mean, maximum, entropy_pairs))
features = train_test(X, y; features=(catch9))

tuning = train_test(X, y; tuning=true)
rng_tuning = train_test(X, y; tuning=true, preprocess=(;rng))
parametrized_tuning = train_test(X, y;
    tuning=(
        method=(type=grid, params=(resolution=25,)), 
        params=(repeats=35, n=10),
        ranges=(
            SoleXplorer.range(:merge_purity_threshold, lower=0.1, upper=2.0),
            SoleXplorer.range(:feature_importance, values=[:impurity, :split])
        )
    ), 
    preprocess=(;rng)
)

model_check_1 = train_test(X, y; model=(type=:decisiontree,), tuning=true, preprocess=(;rng))
model_check_2 = train_test(X, y; model=(type=:randomforest,), tuning=true, preprocess=(;rng))
model_check_3 = train_test(X, y; model=(type=:adaboost,), tuning=true, preprocess=(;rng))
model_check_4 = train_test(X, y; model=(type=:modaldecisiontree,), tuning=true, preprocess=(;rng))
model_check_5 = train_test(X, y; model=(type=:modalrandomforest,), tuning=true, preprocess=(;rng))
model_check_6 = train_test(X, y; model=(type=:modaladaboost,), tuning=true, preprocess=(;rng))
model_check_7 = train_test(X, y; model=(type=:xgboost,), tuning=true, preprocess=(;rng))

parametrized_model_type = train_test(X, y; 
    model=(type=:xgboost,
            params=(
                num_round=20, 
                booster="gbtree", 
                eta=0.5,
                num_parallel_tree=10, 
                max_depth=8, 
            )
    )
)

early_stop  = train_test(X, y; 
    model=(type=:xgboost_classifier,
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
    preprocess=(valid_ratio = 0.7,)
)

preprocess = train_test(X, y; preprocess=(valid_ratio=0.5,))

# ---------------------------------------------------------------------------- #
#                                 regression                                   #
# ---------------------------------------------------------------------------- #
table = RDatasets.dataset("datasets", "LifeCycleSavings")
y = table[:, :DDPI]
X = select(table, Not([:DDPI, :Country]));
rng = Xoshiro(11)

# ---------------------------------------------------------------------------- #
#                                 train_test                                   #
# ---------------------------------------------------------------------------- #
no_parameters = train_test(X, y)
model_type = train_test(X, y; model=(type=:randomforest,))
parametrized_model_type = train_test(X, y; 
    model=(type=:randomforest,
            params=(
                n_trees             = 50,
                feature_importance  = :split,
            )
    )
)

resample = train_test(X, y; resample=(type=CV,))
parametrized_resample = train_test(X, y; resample=(type=StratifiedCV, params=(nfolds=10,)))

win = train_test(X, y; win=(type=adaptivewindow,))
parametrized_win = train_test(X, y; win=(type=adaptivewindow, params=(nwindows = 3, relative_overlap = 0.1)))

features = train_test(X, y; features=(mean, maximum, entropy_pairs))
features = train_test(X, y; features=(catch9))

tuning = train_test(X, y; tuning=true)
rng_tuning = train_test(X, y; tuning=true, preprocess=(;rng))
parametrized_tuning = train_test(X, y;
    tuning=(
        method=(type=grid, params=(resolution=25,)), 
        params=(repeats=35, n=10),
        ranges=(
            SoleXplorer.range(:merge_purity_threshold, lower=0.1, upper=2.0),
            SoleXplorer.range(:feature_importance, values=[:impurity, :split])
        )
    ), 
    preprocess=(;rng)
)

model_check_1 = train_test(X, y; model=(type=:decisiontree,), tuning=true, preprocess=(;rng))
model_check_2 = train_test(X, y; model=(type=:randomforest,), tuning=true, preprocess=(;rng))

preprocess = train_test(X, y; preprocess=(valid_ratio=0.5,))
