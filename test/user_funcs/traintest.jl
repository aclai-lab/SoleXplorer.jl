using Test
using Sole
using SoleXplorer
using Random, StatsBase, JLD2, DataFrames
using MLJTuning
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
#                      basic train/test classification                         #
# ---------------------------------------------------------------------------- #
@testset "basic usage traintest function" begin
    @testset "decisiontree_classifier" begin
        result = traintest(X, y; models=(type=:decisiontree_classifier, params=(; rng=rng)))
        @test result isa SoleXplorer.ModelConfig
        @test result.classifier isa SoleXplorer.DecisionTreeClassifier
        @test result.model isa SoleXplorer.DecisionTree
    end

    @testset "randomforest_classifier" begin
        result = traintest(X, y; models=(type=:randomforest_classifier, params=(; rng=rng)))
        @test result isa SoleXplorer.ModelConfig
        @test result.classifier isa SoleXplorer.RandomForestClassifier # type piracy?
        @test result.model isa SoleXplorer.DecisionEnsemble
    end

    @testset "adaboost_classifier" begin
        result = traintest(X, y; models=(type=:adaboost_classifier, params=(; rng=rng)))
        @test result isa SoleXplorer.ModelConfig
        @test result.classifier isa SoleXplorer.AdaBoostStumpClassifier
        @test result.model isa SoleXplorer.DecisionEnsemble
    end

    @testset "modaldecisiontree" begin
        result = traintest(X, y; models=(type=:modaldecisiontree, params=(; rng=rng)))
        @test result isa SoleXplorer.ModelConfig
        @test result.classifier isa SoleXplorer.ModalDecisionTree
        @test result.model isa SoleXplorer.DecisionTree
    end

    @testset "modalrandomforest" begin
        result = traintest(X, y; models=(type=:modalrandomforest, params=(; rng=rng)))
        @test result isa SoleXplorer.ModelConfig
        @test result.classifier isa SoleXplorer.ModalRandomForest
        @test result.model isa SoleXplorer.DecisionEnsemble
    end

    @testset "xgboost_classifier" begin
        result = traintest(X, y; models=(; type=:xgboost_classifier))
        @test result isa SoleXplorer.ModelConfig
        @test result.classifier isa SoleXplorer.XGBoostClassifier
        @test result.model isa SoleXplorer.DecisionEnsemble
    end
end

# ---------------------------------------------------------------------------- #
#                       tuning train/test classification                       #
# ---------------------------------------------------------------------------- #
@testset "tuning usage traintest function" begin
    @testset "decisiontree_classifier" begin
        result = traintest(X, y; models=(type=:decisiontree_classifier, params=(; rng=rng), tuning=true))
        @test result isa SoleXplorer.ModelConfig
        @test result.classifier isa MLJTuning.ProbabilisticTunedModel{LatinHypercube, SoleXplorer.DecisionTreeClassifier}
        @test result.model isa SoleXplorer.DecisionTree
    end

    @testset "randomforest_classifier" begin
        result = traintest(X, y; models=(type=:randomforest_classifier, params=(; rng=rng), tuning=true))
        @test result isa SoleXplorer.ModelConfig
        @test result.classifier isa MLJTuning.ProbabilisticTunedModel{LatinHypercube, SoleXplorer.RandomForestClassifier}
        @test result.model isa SoleXplorer.DecisionEnsemble
    end

    @testset "adaboost_classifier" begin
        result = traintest(X, y; models=(type=:adaboost_classifier, params=(; rng=rng), tuning=true))
        @test result isa SoleXplorer.ModelConfig
        @test result.classifier isa MLJTuning.ProbabilisticTunedModel{LatinHypercube, SoleXplorer.AdaBoostStumpClassifier}
        @test result.model isa SoleXplorer.DecisionEnsemble
    end

    @testset "modaldecisiontree" begin
        result = traintest(X, y; models=(type=:modaldecisiontree, params=(; rng=rng), tuning=true))
        @test result isa SoleXplorer.ModelConfig
        @test result.classifier isa MLJTuning.ProbabilisticTunedModel{LatinHypercube, SoleXplorer.ModalDecisionTree}
        @test result.model isa SoleXplorer.DecisionTree
    end

    @testset "modalrandomforest" begin
        result = traintest(X, y; models=(type=:modalrandomforest, params=(; rng=rng), tuning=true))
        @test result isa SoleXplorer.ModelConfig
        @test result.classifier isa MLJTuning.ProbabilisticTunedModel{LatinHypercube, SoleXplorer.ModalRandomForest}
        @test result.model isa SoleXplorer.DecisionEnsemble
    end

    @testset "xgboost_classifier" begin
        result = traintest(X, y; models=(type=:xgboost_classifier, tuning=true))
        @test result isa SoleXplorer.ModelConfig
        @test result.classifier isa MLJTuning.ProbabilisticTunedModel{LatinHypercube, SoleXplorer.XGBoostClassifier}
        @test result.model isa SoleXplorer.DecisionEnsemble
    end
end

# ---------------------------------------------------------------------------- #
#                    classification pratical usage examples                    #
# ---------------------------------------------------------------------------- #
@testset "classification pratical usage examples" begin
    @testset "decisiontree_classifier" begin
        result = traintest(X, y;
            models=(
                # always declare the model you're going to use
                type=:decisiontree_classifier,
                # you can tweak every parameter of the model
                params=(; max_depth=5, min_samples_leaf=1),
                # optionally you can use different windowing strategies:
                # in this case, even if the model is propositional, and doesnt accept data vectors,
                # we mimic a modal behaviour splitting data vectors in 2 windows,
                # and then, apply choosen features on each window
                winparams=(; type=adaptivewindow, nwindows=2),
                # you can choose which features to use, mode_5 comes from Catch22 package
                features=catch9,
                # optionally you can turn on tuning default settings for every model,
                # using simply "tuning=true"
                tuning=true
            )
        )
        @test result isa SoleXplorer.ModelConfig
        @test result.classifier isa MLJTuning.ProbabilisticTunedModel{LatinHypercube, SoleXplorer.DecisionTreeClassifier}
        @test result.model isa SoleXplorer.DecisionTree
    end

    @testset "randomforest_classifier" begin
        result = traintest(X, y;
            models=(
                type=:randomforest_classifier,
                # params is a NamedTuple: in case you have only one parameter,
                # remember to place a ';' at the beginning, or a ',' at the end
                params=(; n_trees=25),
                features=[minimum, mean, std],
                # you can use a tuning strategy coming from MLJ library
                tuning=(
                    # you can choose the tuning method and adjust the parameters
                    # specific for the choosen method
                    method=(type=latinhypercube, rng=rng), 
                    # you can also tweak global tuning parameters
                    params=(repeats=10, n=5),
                    # every model has default ranges for tuning
                    # but it's highly recommended to choose which parameters to tune
                    ranges=[
                        SoleXplorer.range(:sampling_fraction, lower=0.3, upper=0.9),
                        SoleXplorer.range(:feature_importance, values=[:impurity, :split])
                    ]
                ),   
            )
        )
        @test result isa SoleXplorer.ModelConfig
        @test result.classifier isa MLJTuning.ProbabilisticTunedModel{LatinHypercube, SoleXplorer.RandomForestClassifier}
        @test result.model isa SoleXplorer.DecisionEnsemble
    end

    @testset "modaldecisiontree" begin
        result = traintest(X, y;
            models=(
                type=:modaldecisiontree,
                winparams=(; type=adaptivewindow, nwindows=20),
                features=[minimum, mean, std]
            )
        )
        @test result isa SoleXplorer.ModelConfig
        @test result.classifier isa SoleXplorer.ModalDecisionTree
        @test result.model isa SoleXplorer.DecisionTree
    end

    @testset "preprocess params" begin
        result = traintest(X, y;
            models=(
                type=:decisiontree_classifier,
                params=(; max_depth=5, min_samples_leaf=1),
                winparams=(; type=adaptivewindow, nwindows=2),
                features=[minimum, mean, cov, mode_5],
                tuning=true
            ),
            # you can also specify preprocessing parameters
            # to fine tuning train test split
            preprocess=(
                train_ratio = 0.7,
                stratified=true,
                nfolds=3,
                rng=rng
            )
        )
        @test result isa SoleXplorer.ModelConfig
        @test result.classifier isa MLJTuning.ProbabilisticTunedModel{LatinHypercube, SoleXplorer.DecisionTreeClassifier}
        @test result.model isa Vector{<:SoleXplorer.DecisionTree}
    end
end

# ---------------------------------------------------------------------------- #
#                       classification multiple models                         #
# ---------------------------------------------------------------------------- #
@testset "classification multiple models" begin
    results = traintest(X, y;
        # you can stack multiple models in a vector
        models=[(
                type=:decisiontree_classifier,
                params=(max_depth=3, min_samples_leaf=14),
                features=[minimum, mean, cov, mode_5]
            ),
            (
                type=:adaboost_classifier,
                winparams=(type=movingwindow, window_size=6),
                tuning=true
            ),
            (; type=:modaldecisiontree)],
        # you can also specify global parameters for all models
        # note that if you specify them also in model definitions,
        # they will be overwritten.
        # for example, this could be very useful if you want to pass rng parameter to all models
        globals=(
            params=(; rng=rng),
            features=[std],
            tuning=false
        )
    )
    @test results isa Vector{ModelConfig}
    
    @test results[1] isa SoleXplorer.ModelConfig
    @test results[1].classifier isa SoleXplorer.DecisionTreeClassifier
    @test results[1].model isa SoleXplorer.DecisionTree

    @test results[2] isa SoleXplorer.ModelConfig
    @test results[2].classifier isa MLJTuning.ProbabilisticTunedModel{LatinHypercube, SoleXplorer.AdaBoostStumpClassifier}
    @test results[2].model isa SoleXplorer.DecisionEnsemble

    @test results[3] isa SoleXplorer.ModelConfig
    @test results[3].classifier isa SoleXplorer.ModalDecisionTree
    @test results[3].model isa SoleXplorer.DecisionTree
end

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
#                          basic train/test regression                         #
# ---------------------------------------------------------------------------- #
@testset "basic usage traintest function" begin
    @testset "decisiontree_classifier" begin
        result = traintest(X, y; models=(type=:decisiontree_regressor, params=(; rng=rng)))
        @test result isa SoleXplorer.ModelConfig
        @test result.classifier isa SoleXplorer.DecisionTreeRegressor
        @test result.model isa SoleXplorer.DecisionTree
    end

    @testset "randomforest_classifier" begin
        result = traintest(X, y; models=(type=:randomforest_regressor, params=(; rng=rng)))
        @test result isa SoleXplorer.ModelConfig
        @test result.classifier isa SoleXplorer.RandomForestRegressor
        @test result.model isa SoleXplorer.DecisionEnsemble
    end
end

# ---------------------------------------------------------------------------- #
#                         tuning train/test regression                         #
# ---------------------------------------------------------------------------- #
@testset "tuning usage traintest function" begin
    @testset "decisiontree_classifier" begin
        result = traintest(X, y; models=(type=:decisiontree_regressor, tuning=true))
        @test result isa SoleXplorer.ModelConfig
        @test result.classifier isa MLJTuning.DeterministicTunedModel{LatinHypercube, SoleXplorer.DecisionTreeRegressor}
        @test result.model isa SoleXplorer.DecisionTree
    end

    @testset "randomforest_classifier" begin
        result = traintest(X, y; models=(type=:randomforest_regressor, tuning=true))
        @test result isa SoleXplorer.ModelConfig
        @test result.classifier isa MLJTuning.DeterministicTunedModel{LatinHypercube, SoleXplorer.RandomForestRegressor}
        @test result.model isa SoleXplorer.DecisionEnsemble
    end
end

# ---------------------------------------------------------------------------- #
#                                auto detection                                #
# ---------------------------------------------------------------------------- #
# if the model type (classifier or regressor) is not specified, it will be automatically detected
@testset "automatic detection" begin
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

    @testset "decisiontree" begin
        result = traintest(X, y; models=(; type=:decisiontree))
        @test result isa SoleXplorer.ModelConfig
        @test result.classifier isa SoleXplorer.DecisionTreeClassifier
        @test result.model isa SoleXplorer.DecisionTree
    end

    table = RDatasets.dataset("datasets", "LifeCycleSavings")
    y = table[:, :DDPI]
    X = select(table, Not([:DDPI, :Country]));
    train_seed = 11
    rng = Random.Xoshiro(train_seed)
    Random.seed!(train_seed)

    @testset "decisiontree" begin
        result = traintest(X, y; models=(; type=:decisiontree))
        @test result isa SoleXplorer.ModelConfig
        @test result.classifier isa SoleXplorer.DecisionTreeRegressor
        @test result.model isa SoleXplorer.DecisionTree
    end
end

# ---------------------------------------------------------------------------- #
#                              XGBoost early stop                              #
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

@testset "XGBoost early stop" begin
    @testset "xgboost_classifier" begin
        result = traintest(X, y; 
            models=(type=:xgboost_classifier,
                params=(
                    num_round=100,
                    max_depth=6,
                    eta=0.1, 
                    objective="multi:softprob",
                    # early_stopping parameters
                    early_stopping_rounds=20,
                    watchlist=makewatchlist)
            ),
            # with early stopping a validation set is required
            preprocess=(; valid_ratio = 0.7)
        )
        @test result isa SoleXplorer.ModelConfig
        @test result.classifier isa SoleXplorer.XGBoostClassifier
        @test result.model isa SoleXplorer.DecisionEnsemble

        @test_throws ArgumentError result = traintest(X, y; 
            models=(type=:xgboost_classifier,
                params=(
                    num_round=100,
                    max_depth=6,
                    eta=0.1, 
                    objective="multi:softprob",
                    # early_stopping parameters
                    early_stopping_rounds=20,
                    watchlist=makewatchlist)
            )
        )
    end

    @testset "xgboost_classifier with tuning" begin
        result = traintest(X, y; 
            models=(type=:xgboost_classifier,
                params=(
                    num_round=100,
                    max_depth=6,
                    eta=0.1, 
                    objective="multi:softprob",
                    # early_stopping parameters
                    early_stopping_rounds=20,
                    watchlist=makewatchlist),
                tuning=true
            ),
            preprocess=(; valid_ratio = 0.7)
        )
        @test result isa SoleXplorer.ModelConfig
        @test result.classifier isa MLJTuning.ProbabilisticTunedModel{LatinHypercube, SoleXplorer.XGBoostClassifier}
        @test result.model isa SoleXplorer.DecisionEnsemble
    end

    @testset "xgboost_early_stopping_rounds" begin
        X, y = Sole.load_arff_dataset("NATOPS")
        result = traintest(X, y; 
            models=(type=:xgboost_classifier,
                params=(
                    num_round=10000,
                    max_depth=6,
                    objective="multi:softprob",
                    early_stopping_rounds=20,
                    watchlist=makewatchlist,
                    seed=11),
                    winparams=(; type=adaptivewindow, nwindows=5),
                    features=catch9
                ),
            preprocess=(; valid_ratio = 0.8)
        )
    end
end