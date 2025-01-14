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
model_name = :decision_tree
features = catch9
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

model = symbolic_analysis(X, y;
    model = 
        (
            model = :decision_tree,
            params = (max_depth = 3, min_samples_leaf = 14),
            winparams = (type = movingwindow, window_size = 1024),
            features = [minimum, mean, cov, mode_5]
        ),

    global_params = (
        params = (min_samples_split = 17,),
        winparams = (type = adaptivewindow,),
        features = [std]
    )
)

model = symbolic_analysis(X, y;
    model = (model = :decision_tree,),
)

models = symbolic_analysis(X, y;
    models = [
        (
            model = :decision_tree,
            params = (max_depth = 3, min_samples_leaf = 14),
            winparams = (type = movingwindow, window_size = 1024),
            features = [minimum, mean, cov, mode_5],
            tuneparams = true,
            ranges = [model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])]
        ),
        (
            model = :decision_tree,
            params = (min_samples_leaf = 30, min_samples_split = -2,)
        )
    ],
)

models = symbolic_analysis(X, y;
    models = [
        (
            model = :decision_tree,
            params = (max_depth = 3, min_samples_leaf = 14),
            winparams = (type = movingwindow, window_size = 1024),
            features = [minimum, mean, cov, mode_5]
        ),
        (
            model = :decision_tree,
            params = (min_samples_leaf = 30, min_samples_split = -2,)
        )
    ],
    global_params = (
        params = (min_samples_split = 17,),
        winparams = (type = adaptivewindow, relative_overlap = 0.23),
        features = [std]
    )
)
