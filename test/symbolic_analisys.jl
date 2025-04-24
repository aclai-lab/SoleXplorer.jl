using Test
using SoleXplorer
using DataFrames
using StatsBase: sample

# ---------------------------------------------------------------------------- #
#                       numeric dataset classification                         #
# ---------------------------------------------------------------------------- #
@info "numeric dataset classification"
using MLJBase

X, y = @load_iris
X = DataFrame(X)

function show_results(modelset)
    println("rules extracted:")
    println(modelset.rules)
    println("accuracy: ", get_accuracy(modelset))
end

# ---------------------------------------------------------------------------- #
#                           decision tree classifier                           #
# ---------------------------------------------------------------------------- #
@info "decision tree classifier, intrees rule extractor"
modelset = symbolic_analysis(
    X, y;
    model=(type=:decisiontree, params=(max_depth=2, min_samples_leaf=2)),
    preprocess=(;rng=Xoshiro(11)),
    extract_rules=true
)
show_results(modelset)

# decision tree with resampling cross validation
modelset = symbolic_analysis(
    X, y;
    model=(type=:decisiontree, params=(max_depth=5, min_samples_leaf=2)),
    resample=(;type=CV),
    preprocess=(;rng=Xoshiro(11)),
    extract_rules=true
)
show_results(modelset)

@info "decision tree classifier, tuning strategy"
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
    preprocess=(;rng=Xoshiro(11))
)
show_results(modelset)

@info "decision tree classifier, refne rule extractor"
modelset = symbolic_analysis(
    X, y;
    model=(type=:decisiontree, params=(max_depth=2, min_samples_leaf=2)),
    preprocess=(;rng=Xoshiro(11)),
    extract_rules=(type=:refne, params=(;L=5))
)
show_results(modelset)

# decision tree with resampling cross validation
modelset = symbolic_analysis(
    X, y;
    model=(type=:decisiontree, params=(max_depth=5, min_samples_leaf=2)),
    resample=(;type=StratifiedCV),
    preprocess=(;rng=Xoshiro(11)),
    extract_rules=(type=:refne, params=(;L=5))
)
show_results(modelset)

@info "decision tree classifier, trepan rule extractor"
modelset = symbolic_analysis(
    X, y;
    model=(type=:decisiontree, params=(max_depth=2, min_samples_leaf=2)),
    preprocess=(;rng=Xoshiro(11)),
    extract_rules=(;type=:trepan)
)
show_results(modelset)

modelset = symbolic_analysis(
    X, y;
    model=(type=:decisiontree, params=(max_depth=5, min_samples_leaf=2)),
    resample=(;type=Holdout),
    preprocess=(;rng=Xoshiro(11)),
    extract_rules=(;type=:trepan)
)
show_results(modelset)

@test_throws ArgumentError symbolic_analysis(
    X, y;
    model=(type=:decisiontree, params=(max_depth=2, min_samples_leaf=2)),
    preprocess=(;rng=Xoshiro(11)),
    extract_rules=(;type=:batrees)
)

@test_throws ArgumentError symbolic_analysis(
    X, y;
    model=(type=:decisiontree, params=(max_depth=2, min_samples_leaf=2)),
    preprocess=(;rng=Xoshiro(11)),
    extract_rules=(;type=:rulecosi)
)

@test_throws ArgumentError symbolic_analysis(
    X, y;
    model=(type=:decisiontree, params=(max_depth=2, min_samples_leaf=2)),
    preprocess=(;rng=Xoshiro(11)),
    extract_rules=(;type=:lumen)
)

# ---------------------------------------------------------------------------- #
#                           random forest classifier                           #
# ---------------------------------------------------------------------------- #
@info "random forest classifier, intrees rule extractor"
modelset = symbolic_analysis(
    X, y;
    model=(;type=:randomforest),
    preprocess=(;rng=Xoshiro(1)),
    extract_rules=(;type=:intrees)
)
show_results(modelset)

# random forest with resampling cross validation
modelset = symbolic_analysis(
    X, y;
    model=(type=:randomforest, params=(;max_depth=2)),
    resample=(type=CV,),
    preprocess=(;rng=Xoshiro(11)),
    extract_rules=(;type=:intrees)
)
show_results(modelset)

# random forest with tuning strategy
modelset = symbolic_analysis(
    X, y;
    model=(;type=:randomforest, params=(;max_depth=5)),
    # resample=(type=StratifiedCV, params=(nfolds=10,)),
    tuning=(
        method=(;type=latinhypercube), 
        params=(repeats=25, n=10),
        ranges=(
            SoleXplorer.range(:max_depth, lower=2, upper=10),
            SoleXplorer.range(:feature_importance, values=[:impurity, :split])
        )
    ), 
    preprocess=(;rng=Xoshiro(11)),
)
show_results(modelset)

@info "random forest classifier, refne rule extractor"
modelset = symbolic_analysis(
    X, y;
    model=(type=:randomforest, params=(;max_depth=2)),
    preprocess=(;rng=Xoshiro(1)),
    extract_rules=(;type=:refne)
)
show_results(modelset)

# random forest with resampling cross validation
modelset = symbolic_analysis(
    X, y;
    model=(type=:randomforest, params=(;max_depth=2)),
    resample=(type=CV,),
    preprocess=(;rng=Xoshiro(11)),
    extract_rules=(;type=:refne)
)
show_results(modelset)

@info "random forest classifier, trepan rule extractor"
modelset = symbolic_analysis(
    X, y;
    model=(type=:randomforest, params=(;max_depth=2)),
    preprocess=(;rng=Xoshiro(1)),
    extract_rules=(;type=:trepan)
)
show_results(modelset)

# random forest with resampling cross validation
modelset = symbolic_analysis(
    X, y;
    model=(type=:randomforest, params=(;max_depth=2)),
    resample=(type=CV,),
    preprocess=(;rng=Xoshiro(11)),
    extract_rules=(;type=:trepan)
)
show_results(modelset)

@info "random forest classifier, batrees rule extractor"
modelset = symbolic_analysis(
    X, y;
    model=(type=:randomforest, params=(;max_depth=2)),
    preprocess=(;rng=Xoshiro(1)),
    extract_rules=(;type=:batrees)
)
show_results(modelset)

# random forest with resampling cross validation
modelset = symbolic_analysis(
    X, y;
    model=(type=:randomforest, params=(;max_depth=2)),
    resample=(type=CV,),
    preprocess=(;rng=Xoshiro(11)),
    extract_rules=(;type=:batrees)
)
show_results(modelset)

@info "random forest classifier, rulecosi rule extractor"
modelset = symbolic_analysis(
    X, y;
    model=(type=:randomforest, params=(;max_depth=2)),
    preprocess=(;rng=Xoshiro(1)),
    extract_rules=(;type=:rulecosi)
)
show_results(modelset)

# random forest with resampling cross validation
modelset = symbolic_analysis(
    X, y;
    model=(type=:randomforest, params=(;max_depth=2)),
    resample=(type=CV,),
    preprocess=(;rng=Xoshiro(11)),
    extract_rules=(;type=:rulecosi)
)
show_results(modelset)

@info "random forest classifier, lumen rule extractor"
modelset = symbolic_analysis(
    X, y;
    model=(type=:randomforest, params=(;max_depth=2)),
    preprocess=(;rng=Xoshiro(1)),
    extract_rules=(;type=:lumen)
)
show_results(modelset)

# random forest with resampling cross validation
modelset = symbolic_analysis(
    X, y;
    model=(type=:randomforest, params=(;max_depth=2)),
    resample=(type=CV,),
    preprocess=(;rng=Xoshiro(11)),
    extract_rules=(;type=:lumen)
)
show_results(modelset)

# ---------------------------------------------------------------------------- #
#                             adaboost classifier                              #
# ---------------------------------------------------------------------------- #
@info "adaboost classifier"
modelset = symbolic_analysis(
    X, y;
    model=(type=:adaboost, params=(;n_iter=5)),
    preprocess=(;rng=Xoshiro(11))
)
show_results(modelset)

# adaboost with resampling cross validation
modelset = symbolic_analysis(
    X, y;
    model=(type=:adaboost, params=(;n_iter=5)),
    resample=(type=CV,),
    preprocess=(;rng=Xoshiro(11))
)
show_results(modelset)

# adaboost with tuning strategy
modelset = symbolic_analysis(
    X, y;
    model=(;type=:adaboost, params=(;n_iter=5)),
    tuning=(
        method=(;type=latinhypercube), 
        params=(repeats=25, n=10),
        ranges=(SoleXplorer.range(:n_iter, lower=2, upper=10),)
    ), 
    preprocess=(;rng=Xoshiro(11))
)
show_results(modelset)

# ---------------------------------------------------------------------------- #
#                             xgboost classifier                              #
# ---------------------------------------------------------------------------- #
@info "xgboost classifier"
modelset = symbolic_analysis(
    X, y;
    model=(type=:xgboost, 
        params=(
            num_round=25,
            max_depth=6,
            eta=0.3, 
            objective="multi:softprob",
        )),
    preprocess=(;rng=Xoshiro(11))
)
show_results(modelset)

# xgboost with resampling cross validation
modelset = symbolic_analysis(
    X, y;
    model=(type=:xgboost,
        params=(
            num_round=25,
            max_depth=6,
            eta=0.3, 
            objective="multi:softprob",
        )),
    resample=(type=CV, params=(nfolds=10,)),
    preprocess=(;rng=Xoshiro(11))
)
show_results(modelset)

# xgboost with tuning strategy
modelset = symbolic_analysis(
    X, y;
    model=(type=:xgboost,
        params=(
            max_depth=6,
            objective="multi:softprob",
        )),
    tuning=(
        method=(;type=latinhypercube), 
        params=(repeats=25, n=10),
        ranges=(
            SoleXplorer.range(:eta, lower=0.1, upper=0.5),
            SoleXplorer.range(:num_round, lower=10, upper=80),
        )
    ),
    preprocess=(;rng=Xoshiro(11))
)
show_results(modelset)

# ---------------------------------------------------------------------------- #
#                                   summary                                    #
# ---------------------------------------------------------------------------- #
model_check_1 = symbolic_analysis(X, y; model=(type=:decisiontree,), tuning=true, preprocess=(;rng=Xoshiro(11)))
model_check_2 = symbolic_analysis(X, y; model=(type=:randomforest,), tuning=true, preprocess=(;rng=Xoshiro(11)))
model_check_3 = symbolic_analysis(X, y; model=(type=:adaboost,), tuning=true, preprocess=(;rng=Xoshiro(11)))
model_check_4 = symbolic_analysis(X, y; model=(type=:modaldecisiontree,), tuning=true, preprocess=(;rng=Xoshiro(11)))
model_check_5 = symbolic_analysis(X, y; model=(type=:modalrandomforest,), tuning=true, preprocess=(;rng=Xoshiro(11)))
model_check_6 = symbolic_analysis(X, y; model=(type=:modaladaboost,), tuning=true, preprocess=(;rng=Xoshiro(11)))
model_check_7 = symbolic_analysis(X, y; model=(type=:xgboost,), tuning=true, preprocess=(;rng=Xoshiro(11)))

# ---------------------------------------------------------------------------- #
#                                 regression                                   #
# ---------------------------------------------------------------------------- #
@info "regression"
using RDatasets

table = RDatasets.dataset("datasets", "LifeCycleSavings")
y = table[:, :DDPI]
X = DataFrames.select(table, Not([:DDPI, :Country]));

# ---------------------------------------------------------------------------- #
#                           decision tree regressor                            #
# ---------------------------------------------------------------------------- #
@info "decision tree regression"
modelset = symbolic_analysis(
    X, y;
    model=(type=:decisiontree, params=params=(max_depth=3, min_samples_leaf=5, min_purity_increase=0.01)),
    preprocess=(;rng=Xoshiro(11))
)
show_results(modelset)

# decision tree regression with resampling cross validation
modelset = symbolic_analysis(
    X, y;
    model=(type=:decisiontree, params=params=(max_depth=3, min_samples_leaf=5, min_purity_increase=0.01)),
    resample=(type=CV, params=(nfolds=10,)),
    preprocess=(;rng=Xoshiro(11))
)
show_results(modelset)

# decision tree regression with tuning strategy
modelset = symbolic_analysis(
    X, y;
    model=(;type=:decisiontree, params=params=(max_depth=3, min_samples_leaf=5, min_purity_increase=0.01)),
    tuning=(
        method=(;type=latinhypercube), 
        params=(repeats=25, n=10),
        ranges=(
            SoleXplorer.range(:max_depth, lower=2, upper=10),
            SoleXplorer.range(:feature_importance, values=[:impurity, :split])
        )
    ), 
    preprocess=(;rng=Xoshiro(11))
)
show_results(modelset)

# ---------------------------------------------------------------------------- #
#                           random forest regressor                            #
# ---------------------------------------------------------------------------- #
@info "random forest regression"
modelset = symbolic_analysis(
    X, y;
    model=(type=:randomforest, params=(;max_depth=5)),
    preprocess=(;rng=Xoshiro(11))
)
show_results(modelset)

# random forest regression with resampling cross validation
modelset = symbolic_analysis(
    X, y;
    model=(type=:randomforest, params=(;max_depth=5)),
    resample=(type=StratifiedCV, params=(nfolds=10,)),
    preprocess=(;rng=Xoshiro(11))
)
show_results(modelset)

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
    preprocess=(;rng=Xoshiro(11))
)
show_results(modelset)

# ---------------------------------------------------------------------------- #
#                           time series classifier                             #
# ---------------------------------------------------------------------------- #
@info "time series classifier"
X, y = load_arff_dataset("NATOPS")

# ---------------------------------------------------------------------------- #
#                           decision tree classifier                           #
# ---------------------------------------------------------------------------- #
@info "decision tree classifier"
modelset = symbolic_analysis(
    X, y;
    model=(type=:decisiontree, params=(max_depth=5, min_samples_leaf=2)),
    features=(catch9),
    win=(type=adaptivewindow, params=(nwindows=3, relative_overlap=0.1)),
    preprocess=(;train_ratio=0.8, rng=Xoshiro(11)),
    # rules_extraction=true
)
show_results(modelset)

# decision tree with resampling cross validation
modelset = symbolic_analysis(
    X, y;
    model=(type=:decisiontree, params=(max_depth=5, min_samples_leaf=2)),
    resample=(type=CV, params=(nfolds=10,)),
    features=(catch9),
    win=(type=adaptivewindow, params=(nwindows=3, relative_overlap=0.1)),
    preprocess=(;train_ratio=0.8, rng=Xoshiro(11))
)
show_results(modelset)

# decision tree with tuning strategy
modelset = symbolic_analysis(
    X, y;
    model=(;type=:decisiontree, params=(max_depth=5, min_samples_leaf=2)),
    features=(catch9),
    win=(type=adaptivewindow, params=(nwindows=3, relative_overlap=0.1)),
    tuning=true,
    preprocess=(;train_ratio=0.8, rng=Xoshiro(11))
)
show_results(modelset)

# ---------------------------------------------------------------------------- #
#                           random forest classifier                           #
# ---------------------------------------------------------------------------- #
@info "random forest classifier"
modelset = symbolic_analysis(
    X, y;
    model=(type=:randomforest, params=(;max_depth=5)),
    features=(minimum, maximum, mean),
    win=(type=adaptivewindow, params=(nwindows=3, relative_overlap=0.1)),
    preprocess=(;rng=Xoshiro(11))
)
show_results(modelset)

# random forest with resampling cross validation
modelset = symbolic_analysis(
    X, y;
    model=(type=:randomforest, params=(;max_depth=5)),
    resample=(type=StratifiedCV, params=(nfolds=10,)),
    features=(minimum, maximum, mean),
    win=(type=adaptivewindow, params=(nwindows=3, relative_overlap=0.1)),
    preprocess=(;rng=Xoshiro(11))
)
show_results(modelset)

# random forest with tuning strategy
modelset = symbolic_analysis(
    X, y;
    model=(;type=:randomforest, params=(;max_depth=5)),
    resample=(type=StratifiedCV, params=(nfolds=10,)),
    features=(minimum, maximum, mean),
    win=(type=adaptivewindow, params=(nwindows=3, relative_overlap=0.1)),
    tuning=true,
    preprocess=(;rng=Xoshiro(11))
)
show_results(modelset)

# ---------------------------------------------------------------------------- #
#                             adaboost classifier                              #
# ---------------------------------------------------------------------------- #
@info "adaboost classifier"
modelset = symbolic_analysis(
    X, y;
    model=(type=:adaboost, params=(;n_iter=5)),
    features=(catch9),
    win=(type=adaptivewindow, params=(nwindows=5, relative_overlap=0.1)),
    preprocess=(;rng=Xoshiro(11)),
    extract_rules=true
)
show_results(modelset)

# adaboost with resampling cross validation
modelset = symbolic_analysis(
    X, y;
    model=(type=:adaboost, params=(;n_iter=5)),
    resample=(type=CV, params=(nfolds=10,)),
    features=(catch9),
    win=(type=adaptivewindow, params=(nwindows=5, relative_overlap=0.1)),
    preprocess=(;rng=Xoshiro(11))
)
show_results(modelset)

# adaboost with tuning strategy
modelset = symbolic_analysis(
    X, y;
    model=(;type=:adaboost, params=(;n_iter=5)),
    tuning=true,
    features=(catch9),
    win=(type=adaptivewindow, params=(nwindows=5, relative_overlap=0.1)),
    preprocess=(;rng=Xoshiro(11))
)
show_results(modelset)

# ---------------------------------------------------------------------------- #
#                        modal decision tree classifier                        #
# ---------------------------------------------------------------------------- #
@info "modal decision tree classifier"
modelset = symbolic_analysis(
    X, y;
    model=(type=:modaldecisiontree, params=(max_depth=5, min_samples_leaf=4)),
    features=(catch9),
    win=(type=adaptivewindow, params=(nwindows=3, relative_overlap=0.1)),
    preprocess=(;rng=Xoshiro(11)),
    reducefunc=median
)
show_results(modelset)

# modal decision tree with resampling cross validation
modelset = symbolic_analysis(
    X, y;
    model=(type=:modaldecisiontree, params=(max_depth=5, min_samples_leaf=2)),
    resample=(type=CV, params=(nfolds=10,)),
    features=(catch9),
    win=(type=adaptivewindow, params=(nwindows=3, relative_overlap=0.1)),
    preprocess=(;rng=Xoshiro(11))
)
show_results(modelset)

# modal decision tree with tuning strategy
modelset = symbolic_analysis(
    X, y;
    model=(;type=:modaldecisiontree, params=(max_depth=5, min_samples_leaf=2)),
    features=(catch9),
    win=(type=adaptivewindow, params=(nwindows=3, relative_overlap=0.1)),
    tuning=(
        method=(;type=latinhypercube), 
        params=(repeats=25, n=10),
        ranges=(
            SoleXplorer.range(:max_depth, lower=2, upper=10),
            SoleXplorer.range(:min_samples_leaf, lower=2, upper=6)
        )
    ), 
    preprocess=(;rng=Xoshiro(11)),
    reducefunc=median
)
show_results(modelset)

# ---------------------------------------------------------------------------- #
#                         modal random forest classifier                       #
# ---------------------------------------------------------------------------- #
@info "modal random forest classifier"
modelset = symbolic_analysis(
    X, y;
    model=(type=:modalrandomforest, params=(;max_depth=5)),
    features=(minimum, maximum, entropy_pairs),
    win=(type=adaptivewindow, params=(nwindows=3, relative_overlap=0.1)),
    preprocess=(;rng=Xoshiro(11))
)
show_results(modelset)

# modal random forest with resampling cross validation
modelset = symbolic_analysis(
    X, y;
    model=(type=:modalrandomforest, params=(;max_depth=5)),
    resample=(type=StratifiedCV, params=(nfolds=10,)),
    features=(minimum, maximum, entropy_pairs),
    win=(type=adaptivewindow, params=(nwindows=3, relative_overlap=0.1)),
    preprocess=(;rng=Xoshiro(11))
)
show_results(modelset)

# modal random forest with tuning strategy
modelset = symbolic_analysis(
    X, y;
    model=(;type=:modalrandomforest, params=(;max_depth=5)),
    resample=(type=StratifiedCV, params=(nfolds=10,)),
    features=(minimum, maximum, entropy_pairs),
    win=(type=adaptivewindow, params=(nwindows=3, relative_overlap=0.1)),
    tuning=(
        method=(;type=latinhypercube), 
        params=(repeats=25, n=10),
        ranges=(
            SoleXplorer.range(:max_depth, lower=2, upper=10),
            SoleXplorer.range(:feature_importance, values=[:impurity, :split])
        )
    ), 
    preprocess=(;rng=Xoshiro(11))
)
show_results(modelset)

# ---------------------------------------------------------------------------- #
#                          modal adaboost classifier                           #
# ---------------------------------------------------------------------------- #
@info "modal adaboost classifier"
modelset = symbolic_analysis(
    X, y;
    model=(type=:modaladaboost, params=(;n_iter=5)),
    preprocess=(;rng=Xoshiro(11))
)
show_results(modelset)

# modal adaboost with resampling cross validation
modelset = symbolic_analysis(
    X, y;
    model=(type=:modaladaboost, params=(;n_iter=5)),
    resample=(type=CV,),
    preprocess=(;rng=Xoshiro(11))
)
show_results(modelset)

# modal adaboost with tuning strategy
modelset = symbolic_analysis(
    X, y;
    model=(;type=:modaladaboost, params=(;n_iter=5)),
    tuning=(
        method=(;type=latinhypercube), 
        params=(repeats=25, n=10),
        ranges=(SoleXplorer.range(:n_iter, lower=2, upper=10),)
    ), 
    preprocess=(;rng=Xoshiro(11))
)
show_results(modelset)

# ---------------------------------------------------------------------------- #
#                             xgboost classifier                              #
# ---------------------------------------------------------------------------- #
@info "xgboost classifier"
modelset = symbolic_analysis(
    X, y;
    model=(type=:xgboost, 
        params=(
            num_round=20,
            max_depth=3,
            eta=0.3, 
            objective="multi:softprob",
            # early_stopping parameters
            early_stopping_rounds=10,
            watchlist=makewatchlist
        )),
    features=(catch9),
    win=(;type=adaptivewindow),
    # with early stopping a validation set is required
    preprocess=(;valid_ratio=0.8, rng=Xoshiro(11))
)
show_results(modelset)

# xgboost with resampling cross validation
modelset = symbolic_analysis(
    X, y;
    model=(type=:xgboost,
        params=(
            num_round=30,
            max_depth=6,
            eta=0.1, 
            objective="multi:softprob",
            early_stopping_rounds=10,
            watchlist=makewatchlist
        )),
    resample=(type=CV, params=(nfolds=10,)),
    features=(catch9),
    win=(;type=adaptivewindow),
    preprocess=(;valid_ratio=0.8, rng=Xoshiro(11))
)
show_results(modelset)

# xgboost with tuning strategy
modelset = symbolic_analysis(
    X, y;
    model=(type=:xgboost,
        params=(
            num_round=100,
            max_depth=6,
            objective="multi:softprob",
            early_stopping_rounds=10,
            watchlist=makewatchlist
        )),
    tuning=(
        method=(;type=latinhypercube), 
        params=(repeats=25, n=10),
        ranges=(SoleXplorer.range(:eta, lower=0.1, upper=0.5),)
    ),
    # features=(catch9),
    win=(;type=adaptivewindow),
    preprocess=(;valid_ratio=0.8, rng=Xoshiro(11))
)
show_results(modelset)

@info "end test."
