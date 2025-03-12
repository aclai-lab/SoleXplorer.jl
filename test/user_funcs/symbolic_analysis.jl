using SoleXplorer
using Test
using Sole
using Random, StatsBase, DataFrames
using MLJTuning
using DecisionTree: load_data

# ---------------------------------------------------------------------------- #
#                                CLASSIFICATION                                #
# ---------------------------------------------------------------------------- #
X, y = load_data("iris")
X = DataFrame(Float64.(X), :auto)
y = string.(y)

# ---------------------------------------------------------------------------- #
#                           basic symbolic analysis                            #
# ---------------------------------------------------------------------------- #
result = symbolic_analysis(X, y; models=(type=:decisiontree_classifier, rules=(type=:plainrules,)))
result = symbolic_analysis(X, y; models=(type=:randomforest_classifier,))
result = symbolic_analysis(X, y; models=(type=:adaboost,))

result = symbolic_analysis(X, y; models=(type=:decisiontree, tuning=true),)

result = symbolic_analysis(X, y;
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

result = symbolic_analysis(X, y;
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

result = symbolic_analysis(X, y;
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

result=symbolic_analysis(X, y;
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

