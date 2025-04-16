using Test
using SoleXplorer
using DataFrames
using StatsBase: sample
using DecisionTree: load_data
using RDatasets

# ---------------------------------------------------------------------------- #
#                                numeric dataset                               #
# ---------------------------------------------------------------------------- #
X, y = load_data("iris")
X = DataFrame(Float64.(X), :auto)
y = String.(y)
rng = Xoshiro(11)

# ---------------------------------------------------------------------------- #
#                             symbolic_analysis                                #
# ---------------------------------------------------------------------------- #
no_parameters = symbolic_analysis(X, y)
model_type = symbolic_analysis(X, y; model=(type=:modaldecisiontree,))
parametrized_model_type = symbolic_analysis(X, y; 
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

reducefunc = symbolic_analysis(X, y; model=(type=:modaldecisiontree,), reducefunc=median)

resample = symbolic_analysis(X, y; resample=(type=CV,))
parametrized_resample = symbolic_analysis(X, y; resample=(type=StratifiedCV, params=(nfolds=10,)))

win = symbolic_analysis(X, y; win=(type=adaptivewindow,))
parametrized_win = symbolic_analysis(X, y; win=(type=adaptivewindow, params=(nwindows = 3, relative_overlap = 0.1)))

features = symbolic_analysis(X, y; features=(mean, maximum, entropy_pairs))
features = symbolic_analysis(X, y; features=(catch9))

tuning = symbolic_analysis(X, y; tuning=true)
rng_tuning = symbolic_analysis(X, y; tuning=true, preprocess=(;rng))
parametrized_tuning = symbolic_analysis(X, y;
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

model_check_1 = symbolic_analysis(X, y; model=(type=:decisiontree,), tuning=true, preprocess=(;rng))
model_check_2 = symbolic_analysis(X, y; model=(type=:randomforest,), tuning=true, preprocess=(;rng))
model_check_3 = symbolic_analysis(X, y; model=(type=:adaboost,), tuning=true, preprocess=(;rng))
model_check_4 = symbolic_analysis(X, y; model=(type=:modaldecisiontree,), tuning=true, preprocess=(;rng))
model_check_5 = symbolic_analysis(X, y; model=(type=:modalrandomforest,), tuning=true, preprocess=(;rng))
model_check_6 = symbolic_analysis(X, y; model=(type=:modaladaboost,), tuning=true, preprocess=(;rng))
model_check_7 = symbolic_analysis(X, y; model=(type=:xgboost,), tuning=true, preprocess=(;rng))

parametrized_model_type = symbolic_analysis(X, y; 
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

early_stop  = symbolic_analysis(X, y; 
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

preprocess = symbolic_analysis(X, y; preprocess=(valid_ratio=0.5,))

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
#                               symbolic_analysis                                #
# ---------------------------------------------------------------------------- #
no_parameters = symbolic_analysis(X, y)
model_type = symbolic_analysis(X, y; model=(type=:modaldecisiontree,))
parametrized_model_type = symbolic_analysis(X, y; 
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

reducefunc = symbolic_analysis(X, y; model=(type=:modaldecisiontree,), reducefunc=median)

resample = symbolic_analysis(X, y; resample=(type=CV,))
parametrized_resample = symbolic_analysis(X, y; resample=(type=StratifiedCV, params=(nfolds=10,)))

win = symbolic_analysis(X, y; win=(type=adaptivewindow,))
parametrized_win = symbolic_analysis(X, y; win=(type=adaptivewindow, params=(nwindows = 3, relative_overlap = 0.1)))

features = symbolic_analysis(X, y; features=(mean, maximum, entropy_pairs))
features = symbolic_analysis(X, y; features=(catch9))

tuning = symbolic_analysis(X, y; tuning=true)
rng_tuning = symbolic_analysis(X, y; tuning=true, preprocess=(;rng))
parametrized_tuning = symbolic_analysis(X, y;
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

preprocess = symbolic_analysis(X, y; preprocess=(valid_ratio=0.5,))

# ---------------------------------------------------------------------------- #
#                                 symbolic_analysis                                   #
# ---------------------------------------------------------------------------- #
no_parameters = symbolic_analysis(X, y)
model_type = symbolic_analysis(X, y; model=(type=:modaldecisiontree,))
parametrized_model_type = symbolic_analysis(X, y; 
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

reducefunc = symbolic_analysis(X, y; model=(type=:modaldecisiontree,), reducefunc=median)

resample = symbolic_analysis(X, y; resample=(type=CV,))
parametrized_resample = symbolic_analysis(X, y; resample=(type=StratifiedCV, params=(nfolds=10,)))

win = symbolic_analysis(X, y; win=(type=adaptivewindow,))
parametrized_win = symbolic_analysis(X, y; win=(type=adaptivewindow, params=(nwindows = 3, relative_overlap = 0.1)))

features = symbolic_analysis(X, y; features=(mean, maximum, entropy_pairs))
features = symbolic_analysis(X, y; features=(catch9))

tuning = symbolic_analysis(X, y; tuning=true)
rng_tuning = symbolic_analysis(X, y; tuning=true, preprocess=(;rng))
parametrized_tuning = symbolic_analysis(X, y;
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

model_check_1 = symbolic_analysis(X, y; model=(type=:decisiontree,), tuning=true, preprocess=(;rng))
model_check_2 = symbolic_analysis(X, y; model=(type=:randomforest,), tuning=true, preprocess=(;rng))
model_check_3 = symbolic_analysis(X, y; model=(type=:adaboost,), tuning=true, preprocess=(;rng))
model_check_4 = symbolic_analysis(X, y; model=(type=:modaldecisiontree,), tuning=true, preprocess=(;rng))
model_check_5 = symbolic_analysis(X, y; model=(type=:modalrandomforest,), tuning=true, preprocess=(;rng))
model_check_6 = symbolic_analysis(X, y; model=(type=:modaladaboost,), tuning=true, preprocess=(;rng))
model_check_7 = symbolic_analysis(X, y; model=(type=:xgboost,), tuning=true, preprocess=(;rng))

parametrized_model_type = symbolic_analysis(X, y; 
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

early_stop  = symbolic_analysis(X, y; 
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

preprocess = symbolic_analysis(X, y; preprocess=(valid_ratio=0.5,))

# ---------------------------------------------------------------------------- #
#                                 regression                                   #
# ---------------------------------------------------------------------------- #
table = RDatasets.dataset("datasets", "LifeCycleSavings")
y = table[:, :DDPI]
X = select(table, Not([:DDPI, :Country]));
rng = Xoshiro(11)

# ---------------------------------------------------------------------------- #
#                                 symbolic_analysis                                   #
# ---------------------------------------------------------------------------- #
no_parameters = symbolic_analysis(X, y)
model_type = symbolic_analysis(X, y; model=(type=:randomforest,))
parametrized_model_type = symbolic_analysis(X, y; 
    model=(type=:randomforest,
            params=(
                n_trees             = 50,
                feature_importance  = :split,
            )
    )
)

resample = symbolic_analysis(X, y; resample=(type=CV,))
parametrized_resample = symbolic_analysis(X, y; resample=(type=StratifiedCV, params=(nfolds=10,)))

win = symbolic_analysis(X, y; win=(type=adaptivewindow,))
parametrized_win = symbolic_analysis(X, y; win=(type=adaptivewindow, params=(nwindows = 3, relative_overlap = 0.1)))

features = symbolic_analysis(X, y; features=(mean, maximum, entropy_pairs))
features = symbolic_analysis(X, y; features=(catch9))

tuning = symbolic_analysis(X, y; tuning=true)
rng_tuning = symbolic_analysis(X, y; tuning=true, preprocess=(;rng))
parametrized_tuning = symbolic_analysis(X, y;
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

model_check_1 = symbolic_analysis(X, y; model=(type=:decisiontree,), tuning=true, preprocess=(;rng))
model_check_2 = symbolic_analysis(X, y; model=(type=:randomforest,), tuning=true, preprocess=(;rng))

preprocess = symbolic_analysis(X, y; preprocess=(valid_ratio=0.5,))
