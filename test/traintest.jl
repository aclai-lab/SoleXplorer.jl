using Test
using Sole
import SoleXplorer as SX
using SoleXplorer
using Random, StatsBase, JLD2, DataFrames
using RDatasets

# ---------------------------------------------------------------------------- #
#                                CLASSIFICATION                                #
# ---------------------------------------------------------------------------- #
X, y       = SoleData.load_arff_dataset("NATOPS")
train_seed = 11
rng        = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

# downsize dataset
num_cols_to_sample = 10
num_rows_to_sample = 50
chosen_cols = StatsBase.sample(rng, 1:size(X, 2), num_cols_to_sample; replace=false)
chosen_rows = StatsBase.sample(rng, 1:size(X, 1), num_rows_to_sample; replace=false)

X = X[chosen_rows, chosen_cols]
y = y[chosen_rows]

# ---------------------------------------------------------------------------- #
#                  basic train/test classification analysis                    #
# ---------------------------------------------------------------------------- #
result = traintest(X, y; models=(type=:decisiontree_classifier, params=(; rng=rng)))
result = traintest(X, y; models=(type=:randomforest_classifier, params=(; rng=rng)))
result = traintest(X, y; models=(type=:adaboost_classifier,     params=(; rng=rng)))

result = traintest(X, y; models=(type=:modaldecisiontree,       params=(; rng=rng)))
result = traintest(X, y; models=(type=:modalrandomforest,       params=(; rng=rng)))

# ---------------------------------------------------------------------------- #
#                  tuning train/test classification analysis                   #
# ---------------------------------------------------------------------------- #
result = traintest(X, y; models=(type=:decisiontree_classifier, tuning=true))
result = traintest(X, y; models=(type=:randomforest_classifier, tuning=true))
result = traintest(X, y; models=(type=:adaboost_classifier, tuning=true))

result = traintest(X, y; models=(type=:modaldecisiontree, tuning=true))
result = traintest(X, y; models=(type=:modalrandomforest, tuning=true))

# ---------------------------------------------------------------------------- #
#                parametrized train/test classification analysis               #
# ---------------------------------------------------------------------------- #
result = traintest(X, y;
    models=(
            type=:decisiontree_classifier,
            params=(max_depth=3, min_samples_leaf=1),
            winparams=(type=movingwindow, window_size=6),
            features=[minimum, mean, cov, mode_5]
        ),
    global_params=(
        params=(;min_samples_split=2),
        winparams=(;type=adaptivewindow),
        features=[std]
    )
)

result = traintest(X, y;
    models=(
            type=:randomforest_classifier,
            params=(max_depth=3, min_samples_leaf=14),
            winparams=(type=movingwindow, window_size=6),
            features=[minimum, mean, cov, mode_5]
        ),
    global_params=(
        params=(min_samples_split=17,),
        winparams=(type=adaptivewindow,),
        features=[std]
    )
)

result = traintest(X, y;
    models=(
        type=:adaboost_classifier,
        winparams=(type=movingwindow, window_size=6),
        features=[minimum, mean, cov, mode_5]
    ),
    global_params=(
        params=(;n_iter=17),
        winparams=(;type=adaptivewindow),
        features=[std]
    )
)

result = traintest(X, y;
    models=(
            type=:modaldecisiontree,
            params=(max_depth=3, min_samples_leaf=1),
            features=[minimum, mean, cov, mode_5]
        ),
    global_params=(
        params=(min_samples_split=17,),
        winparams=(type=adaptivewindow,),
        features=[std]
    )
)

result = traintest(X, y;
    models=(
            type=:modalrandomforest,
            params=(max_depth=3, min_samples_leaf=5),
            features=[minimum, mean, cov, mode_5]
        ),
    global_params=(
        params=(min_samples_split=17,),
        winparams=(type=adaptivewindow,),
        features=[std]
    )
)

# ---------------------------------------------------------------------------- #
#                  multiple train/test classification analysis                 #
# ---------------------------------------------------------------------------- #
result = traintest(X, y;
    models=[(
            type=:decisiontree_classifier,
            params=(max_depth=3, min_samples_leaf=14),
            winparams=(type=movingwindow, window_size=6),
            features=[minimum, mean, cov, mode_5],
            tuning=(
                method=(type=latinhypercube, ntour=20,), 
                params=(repeats=11,), 
                ranges=[SX.range(:feature_importance; values=[:impurity, :split])]
            ),   
        ),
        (type=:randomforest_classifier, params=(min_samples_leaf=30, min_samples_split=1,)
    )],
)

result = traintest(X, y;
    models=[(
            type=:adaboost_classifier,
            features=[minimum, mean, cov, mode_5]
        ),
        (type=:modaldecisiontree, params=(min_samples_leaf=30, min_samples_split=-2,)),
        (
            type=:modalrandomforest,
            tuning=(
                method=(type=latinhypercube,), 
                params=(repeats=2,), 
                ranges=[
                    SX.range(:merge_purity_threshold; lower=0, upper=1),
                    SX.range(:feature_importance; values=[:impurity, :split])]),   
        )],
    global_params=(
        winparams=(type=adaptivewindow, relative_overlap=0.23),
        features=[std]
    )
)

# ---------------------------------------------------------------------------- #
#                                  REGRESSION                                  #
# ---------------------------------------------------------------------------- #
table = RDatasets.dataset("datasets", "LifeCycleSavings")
y = table[:, :DDPI]
X = select(table, Not([:DDPI, :Country]));
train_seed = 11
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

# ---------------------------------------------------------------------------- #
#                     basic train/test regression analysis                     #
# ---------------------------------------------------------------------------- #
result = traintest(X, y; models=(type=:decisiontree_regressor, params=(; rng=rng)))
result = traintest(X, y; models=(type=:randomforest_regressor, params=(; rng=rng)))

# ---------------------------------------------------------------------------- #
#                  tuning train/test classification analysis                   #
# ---------------------------------------------------------------------------- #
result = traintest(X, y; models=(type=:decisiontree_regressor, tuning=true))
result = traintest(X, y; models=(type=:randomforest_regressor, tuning=true))

# ---------------------------------------------------------------------------- #
#                parametrized train/test classification analysis               #
# ---------------------------------------------------------------------------- #
result = traintest(X, y;
    models=(
            type=:decisiontree_regressor,
            params=(max_depth=3, min_samples_leaf=14),
            winparams=(type=movingwindow, window_size=12),
            features=[minimum, mean, cov, mode_5]
        ),
    global_params=(
        params=(min_samples_split=17,),
        winparams=(type=adaptivewindow,),
        features=[std]
    )
)

result = traintest(X, y;
    models=(
            type=:randomforest_regressor,
            params=(max_depth=3, min_samples_leaf=14),
            winparams=(type=movingwindow, window_size=12),
            features=[minimum, mean, cov, mode_5]
        ),
    global_params=(
        params=(min_samples_split=17,),
        winparams=(type=adaptivewindow,),
        features=[std]
    )
)

# ---------------------------------------------------------------------------- #
#                  multiple train/test classification analysis                 #
# ---------------------------------------------------------------------------- #
result=traintest(X, y;
    models=[(
            type=:decisiontree_regressor,
            tuning=(
                method=(type=latinhypercube,), 
                params=(repeats=2,), 
                ranges=[
                    SX.range(:merge_purity_threshold; lower=0, upper=1),
                    SX.range(:feature_importance; values=[:impurity, :split])]),   
        ), (
            type=:randomforest_regressor,
            tuning=(
                method=(type=latinhypercube, ntour=20,), 
                params=(repeats=2,), 
            ),)
    ],
    global_params=(
        params=(min_samples_split=17,),
        winparams=(type=adaptivewindow, relative_overlap=0.23),
        features=[std]
    )
)
