using Test
using SoleXplorer
using DataFrames
using StatsBase: sample

# ---------------------------------------------------------------------------- #
#                       numeric dataset classification                         #
# ---------------------------------------------------------------------------- #
using MLJBase

X, y = @load_iris
X = DataFrame(X)
rng = Xoshiro(11)

# ---------------------------------------------------------------------------- #
#                           decision tree classifier                           #
# ---------------------------------------------------------------------------- #
# decision tree
modelset = symbolic_analysis(
    X, y;
    model=(type=:decisiontree, params=(max_depth=3, min_samples_leaf=5, min_purity_increase=0.01)),
    preprocess=(;rng)
)
println("decision tree accuracy: ", get_accuracy(modelset))

# decision tree with resampling cross validation
modelset = symbolic_analysis(
    X, y;
    model=(type=:decisiontree, params=(max_depth=5, min_samples_leaf=2)),
    resample=(type=CV, params=(nfolds=10,)),
    preprocess=(;rng)
)
println("decision tree with cross validation accuracy: ", get_accuracy(modelset))

# decision tree with tuning strategy
modelset = symbolic_analysis(
    X, y;
    model=(;type=:decisiontree, params=(max_depth=5, min_samples_leaf=2)),
    tuning=(
        method=(;type=latinhypercube), 
        params=(repeats=25, n=10),
        ranges=(
            SoleXplorer.range(:max_depth, lower=2, upper=10),
            SoleXplorer.range(:feature_importance, values=[:impurity, :split])
        )
    ), 
    preprocess=(;rng)
)
println("decision tree with tuning strategy accuracy: ", get_accuracy(modelset))

# ---------------------------------------------------------------------------- #
#                           random forest classifier                           #
# ---------------------------------------------------------------------------- #
# random forest
modelset = symbolic_analysis(
    X, y;
    model=(type=:randomforest, params=(;max_depth=5)),
    preprocess=(;rng)
)
println("random forest accuracy: ", get_accuracy(modelset))

# random forest with resampling cross validation
modelset = symbolic_analysis(
    X, y;
    model=(type=:randomforest, params=(;max_depth=5)),
    resample=(type=StratifiedCV, params=(nfolds=10,)),
    preprocess=(;rng)
)
println("random forest with cross validation accuracy: ", get_accuracy(modelset))

# random forest with tuning strategy
modelset = symbolic_analysis(
    X, y;
    model=(;type=:randomforest, params=(;max_depth=5)),
    resample=(type=StratifiedCV, params=(nfolds=10,)),
    tuning=(
        method=(;type=latinhypercube), 
        params=(repeats=25, n=10),
        ranges=(
            SoleXplorer.range(:max_depth, lower=2, upper=10),
            SoleXplorer.range(:feature_importance, values=[:impurity, :split])
        )
    ), 
    preprocess=(;rng)
)
println("random forest with tuning strategy accuracy: ", get_accuracy(modelset))

# ---------------------------------------------------------------------------- #
#                             adaboost classifier                              #
# ---------------------------------------------------------------------------- #
# adaboost
modelset = symbolic_analysis(
    X, y;
    model=(type=:adaboost, params=(;n_iter=5)),
    preprocess=(;rng)
)
println("adaboost accuracy: ", get_accuracy(modelset))

# adaboost with resampling cross validation
modelset = symbolic_analysis(
    X, y;
    model=(type=:adaboost, params=(;n_iter=5)),
    resample=(type=CV, params=(nfolds=10,)),
    preprocess=(;rng)
)
println("adaboost with cross validation accuracy: ", get_accuracy(modelset))

# adaboost with tuning strategy
modelset = symbolic_analysis(
    X, y;
    model=(;type=:adaboost, params=(;n_iter=5)),
    tuning=(
        method=(;type=latinhypercube), 
        params=(repeats=25, n=10),
        ranges=(SoleXplorer.range(:n_iter, lower=2, upper=10),)
    ), 
    preprocess=(;rng)
)
println("adaboost with tuning strategy accuracy: ", get_accuracy(modelset))

# ---------------------------------------------------------------------------- #
#                                 regression                                   #
# ---------------------------------------------------------------------------- #
using RDatasets

table = RDatasets.dataset("datasets", "LifeCycleSavings")
y = table[:, :DDPI]
X = DataFrames.select(table, Not([:DDPI, :Country]));
rng = Xoshiro(11)

# ---------------------------------------------------------------------------- #
#                           decision tree regressor                            #
# ---------------------------------------------------------------------------- #
# decision tree regression
modelset = symbolic_analysis(
    X, y;
    model=(type=:decisiontree, params=(max_depth=5, min_samples_leaf=2)),
    preprocess=(;rng)
)
println("decision tree regression accuracy: ", get_accuracy(modelset))

# decision tree regression with resampling cross validation
modelset = symbolic_analysis(
    X, y;
    model=(type=:decisiontree, params=(max_depth=5, min_samples_leaf=2)),
    resample=(type=CV, params=(nfolds=10,)),
    preprocess=(;rng)
)
println("decision tree regression with cross validation accuracy: ", get_accuracy(modelset))

# decision tree regression with tuning strategy
modelset = symbolic_analysis(
    X, y;
    model=(;type=:decisiontree, params=(max_depth=5, min_samples_leaf=2)),
    tuning=(
        method=(;type=latinhypercube), 
        params=(repeats=25, n=10),
        ranges=(
            SoleXplorer.range(:max_depth, lower=2, upper=10),
            SoleXplorer.range(:feature_importance, values=[:impurity, :split])
        )
    ), 
    preprocess=(;rng)
)
println("decision tree regression with tuning strategy accuracy: ", get_accuracy(modelset))

# ---------------------------------------------------------------------------- #
#                           random forest regressor                            #
# ---------------------------------------------------------------------------- #
# random forest regression
modelset = symbolic_analysis(
    X, y;
    model=(type=:randomforest, params=(;max_depth=5)),
    preprocess=(;rng)
)
println("random forest regression accuracy: ", get_accuracy(modelset))

# random forest regression with resampling cross validation
modelset = symbolic_analysis(
    X, y;
    model=(type=:randomforest, params=(;max_depth=5)),
    resample=(type=StratifiedCV, params=(nfolds=10,)),
    preprocess=(;rng)
)
println("random forest regression with cross validation accuracy: ", get_accuracy(modelset))

# random forest regression with tuning strategy
modelset = symbolic_analysis(
    X, y;
    model=(;type=:randomforest, params=(;max_depth=5)),
    resample=(type=StratifiedCV, params=(nfolds=10,)),
    tuning=(
        method=(;type=latinhypercube), 
        params=(repeats=25, n=10),
        ranges=(
            SoleXplorer.range(:max_depth, lower=2, upper=10),
            SoleXplorer.range(:feature_importance, values=[:impurity, :split])
        )
    ), 
    preprocess=(;rng)
)
println("random forest regression with tuning strategy accuracy: ", get_accuracy(modelset))

# ---------------------------------------------------------------------------- #
#                           time series classifier                             #
# ---------------------------------------------------------------------------- #
X, y = load_arff_dataset("NATOPS")
num_cols_to_sample, num_rows_to_sample, rng = 10, 50, Xoshiro(11)
chosen_cols = sample(rng, 1:size(X, 2), num_cols_to_sample; replace=false)
chosen_rows = sample(rng, 1:size(X, 1), num_rows_to_sample; replace=false)
X = X[chosen_rows, chosen_cols]
y = y[chosen_rows]










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

no_resample = symbolic_analysis(X, y, preprocess=(;rng = Xoshiro(1)))
resample = symbolic_analysis(X, y; resample=(type=CV,), preprocess=(;rng = Xoshiro(1)))
parametrized_resample = symbolic_analysis(X, y; resample=(type=StratifiedCV, params=(nfolds=100,)), preprocess=(;rng = Xoshiro(1)))
@test get_accuracy(no_resample) ≤ get_accuracy(resample) ≤ get_accuracy(parametrized_resample)

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
#                                decision tree                                 #
# ---------------------------------------------------------------------------- #
# decision tree
modelset = symbolic_analysis(
    X, y;
    model=(;type=:decisiontree, params=(max_depth=5, min_samples_leaf=2)),
    features=(catch9),
    win=(type=adaptivewindow, params=(nwindows=3, relative_overlap=0.1)),
    preprocess=(;rng)
)
println("decision tree accuracy: ", get_accuracy(modelset))

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
