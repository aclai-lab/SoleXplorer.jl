using Test
using Sole
import SoleXplorer as SX
using SoleXplorer
using Random, StatsBase, JLD2, DataFrames
using RDatasets

# ---------------------------------------------------------------------------- #
X, y = SoleData.load_arff_dataset("NATOPS")
train_seed = 11
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

# ---------------------------------------------------------------------------- #
#                           basic symbolic analysis                            #
# ---------------------------------------------------------------------------- #
result = traintest(X, y; models=(type=:decisiontree, params=(rng=rng,)))
result = traintest(X, y; models=(type=:randomforest,))
result = traintest(X, y; models=(type=:adaboost,))

result = traintest(X, y; models=(type=:decisiontree, tuning=true),)

result = traintest(X, y;
    models=(
            type=:decisiontree,
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
    models=[(
            type=:decisiontree,
            params=(max_depth=3, min_samples_leaf=14),
            winparams=(type=movingwindow, window_size=12),
            features=[minimum, mean, cov, mode_5],
            tuning=(
                method=(type=latinhypercube, ntour=20,), 
                params=(repeats=11,), 
                ranges=[SX.range(:feature_importance; values=[:impurity, :split])]
            ),   
        ),
        (type=:decisiontree, params=(min_samples_leaf=30, min_samples_split=1,)
    )],
)

result = traintest(X, y;
    models=[(
            type=:decisiontree,
            params=(max_depth=3, min_samples_leaf=14),
            winparams=(type=movingwindow, window_size=12),
            features=[minimum, mean, cov, mode_5]
        ),
        (type=:decisiontree, params=(min_samples_leaf=30, min_samples_split=-2,)
    )],
    global_params=(
        params=(min_samples_split=17,),
        winparams=(type=adaptivewindow, relative_overlap=0.23),
        features=[std]
    )
)

result=traintest(X, y;
    models=[(
            type=:decisiontree,
            tuning=(
                method=(type=latinhypercube,), 
                params=(repeats=2,), 
                ranges=[
                    SX.range(:merge_purity_threshold; lower=0, upper=1),
                    SX.range(:feature_importance; values=[:impurity, :split])
                ]
            ),   
        ),
        (
            type=:decisiontree,
            tuning=(
                method=(type=latinhypercube, ntour=20,), 
                params=(repeats=2,), 
            ), 
        )
    ],
)

