using Test
using Sole
import SoleXplorer as SX
using SoleXplorer
using Random, StatsBase, JLD2, DataFrames
using RDatasets

# ---------------------------------------------------------------------------- #
X, y = SoleData.load_arff_dataset("NATOPS")
train_seed = 11;

# ---------------------------------------------------------------------------- #
#                             basic decision tree                              #
# ---------------------------------------------------------------------------- #
@info "Test 1: Decision Tree"
model_name = :decisiontree
features = catch9
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = symbolic_analysis(X, y; model = (model = :decisiontree,))
model = symbolic_analysis(X, y; model = (model = :randomforest,))
model = symbolic_analysis(X, y; model = (model = :adaboost,))

model = symbolic_analysis(X, y;
    model = (model = :decisiontree, tuning = true),
)

model = symbolic_analysis(X, y;
    model = 
        (
            model = :decisiontree,
            params = (max_depth = 3, min_samples_leaf = 14),
            winparams = (type = movingwindow, window_size = 12),
            features = [minimum, mean, cov, mode_5]
        ),

    global_params = (
        params = (min_samples_split = 17,),
        winparams = (type = adaptivewindow,),
        features = [std]
    )
)

models = symbolic_analysis(X, y;
    models = [
        (
            model = :decisiontree,
            params = (max_depth = 3, min_samples_leaf = 14),
            winparams = (type = movingwindow, window_size = 12),
            features = [minimum, mean, cov, mode_5],
            tuning = (
                method=(type=latinhypercube, ntour=20,), 
                params=(repeats=11,), 
                ranges = [SX.range(:feature_importance; values=[:impurity, :split])]
            ),   
        ),
        (
            model = :decisiontree,
            params = (min_samples_leaf = 30, min_samples_split = 1,),
        )
    ],
)

models = symbolic_analysis(X, y;
    models = [
        (
            model = :decisiontree,
            params = (max_depth = 3, min_samples_leaf = 14),
            winparams = (type = movingwindow, window_size = 12),
            features = [minimum, mean, cov, mode_5]
        ),
        (
            model = :decisiontree,
            params = (min_samples_leaf = 30, min_samples_split = -2,)
        )
    ],
    global_params = (
        params = (min_samples_split = 17,),
        winparams = (type = adaptivewindow, relative_overlap = 0.23),
        features = [std]
    )
)

models = symbolic_analysis(X, y;
    models = [
        (
            model = :decisiontree,
            tuning = (
                method = (type=latinhypercube,), 
                params = (repeats=2,), 
                ranges = [
                    SX.range(:merge_purity_threshold; lower=0, upper=1),
                    SX.range(:feature_importance; values=[:impurity, :split])
                ]
            ),   
        ),
        (
            model = :decisiontree,
            tuning = (
                method = (type=latinhypercube, ntour=20,), 
                params = (repeats=2,), 
            ), 
        )
    ],
)

