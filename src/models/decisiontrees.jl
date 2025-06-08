# ---------------------------------------------------------------------------- #
#                     models from DecisionTrees package                        #
# ---------------------------------------------------------------------------- #

# CLASSIFIER ----------------------------------------------------------------- #
function DecisionTreeClassifierModel()::ModelSetup{TypeDTC}
    type = MLJDecisionTreeInterface.DecisionTreeClassifier
    config  = (algo=:classification, type=DecisionTree, treatment=:aggregate, rawapply=DT.apply_tree)

    params = (;
        max_depth              = -1,
        min_samples_leaf       = 1,
        min_samples_split      = 2,
        min_purity_increase    = 0.0,
        n_subfeatures          = 0,
        post_prune             = false,
        merge_purity_threshold = 1.0,
        display_depth          = 5,
        feature_importance     = :impurity, # :impurity or :split
        rng                    = Random.TaskLocalRNG()
    )

    winparams = WinParams(wholewindow, NamedTuple())

    rawmodel = (
        mach -> MLJ.fitted_params(mach).tree,
        mach -> MLJ.fitted_params(mach).best_fitted_params.tree    
    )

    learn_method = (
        (mach, X, y) -> (solem = solemodel(MLJ.fitted_params(mach).tree); apply!(solem, X, y); solem),
        (mach, X, y) -> (solem = solemodel(MLJ.fitted_params(mach).best_fitted_params.tree); apply!(solem, X, y); solem)
    )

    tuning = SoleXplorer.TuningParams(
        SoleXplorer.TuningStrategy(latinhypercube, (ntour = 20,)),
        TUNING_PARAMS[:classification],
        (
            model -> MLJ.range(model, :merge_purity_threshold, lower=0., upper=2.0),
            model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
        )
    )

    rulesparams = RulesParams(:intrees, NamedTuple())

    return ModelSetup{TypeDTC}(
        type,
        config,
        params,
        DEFAULT_FEATS,
        nothing,
        winparams,
        rawmodel,
        learn_method,
        tuning,
        rulesparams,
        DEFAULT_PREPROC
    )
end

function RandomForestClassifierModel()::ModelSetup{TypeRFC}
    type   = MLJDecisionTreeInterface.RandomForestClassifier
    config = (algo=:classification, type=DecisionEnsemble, treatment=:aggregate, rawapply=DT.apply_forest)

    params = (;
        max_depth           = -1,
        min_samples_leaf    = 1,
        min_samples_split   = 2,
        min_purity_increase = 0.0,
        n_subfeatures       = -1,
        n_trees             = 10,
        sampling_fraction   = 0.7,
        feature_importance  = :impurity,
        rng                 = Random.TaskLocalRNG()
    )

    winparams = WinParams(wholewindow, NamedTuple())

    rawmodel = (
        mach -> MLJ.fitted_params(mach).forest,
        mach -> MLJ.fitted_params(mach).best_fitted_params.forest
    )

    learn_method = (
        (mach, X, y) -> begin
            classlabels  = (mach).fitresult[2][sortperm((mach).fitresult[3])]
            featurenames = MLJ.report(mach).features
            solem        = solemodel(MLJ.fitted_params(mach).forest; classlabels, featurenames)
            apply!(solem, X, y)
            return solem
        end,
        (mach, X, y) -> begin
            classlabels  = (mach).fitresult.fitresult[2][sortperm((mach).fitresult.fitresult[3])]
            featurenames = MLJ.report(mach).best_report.features
            solem        = solemodel(MLJ.fitted_params(mach).best_fitted_params.forest; classlabels, featurenames)
            apply!(solem, X, y)
            return solem
        end
    )

    tuning = SoleXplorer.TuningParams(
        SoleXplorer.TuningStrategy(latinhypercube, (ntour = 20,)),
        TUNING_PARAMS[:classification],
        (
            model -> MLJ.range(model, :sampling_fraction, lower=0.3, upper=0.9),
            model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
        )
    )

    rulesparams = RulesParams(:intrees, NamedTuple())

    return ModelSetup{TypeRFC}(
        type,
        config,
        params,
        DEFAULT_FEATS,
        nothing,
        winparams,
        rawmodel,
        learn_method,
        tuning,
        rulesparams,
        DEFAULT_PREPROC
    )
end

function AdaBoostClassifierModel()::ModelSetup{TypeABC}
    type   = MLJDecisionTreeInterface.AdaBoostStumpClassifier
    config = (algo=:classification, type=DecisionEnsemble, treatment=:aggregate, rawapply=DT.apply_adaboost_stumps)

    params = (;
        n_iter             = 10,
        feature_importance = :impurity,
        rng                = Random.TaskLocalRNG()
    )

    winparams = WinParams(wholewindow, NamedTuple())

    rawmodel = (
        mach -> MLJ.fitted_params(mach).stumps,
        mach -> MLJ.fitted_params(mach).best_fitted_params.stumps
    )

    learn_method = (
        (mach, X, y) -> begin
            weights      = mach.fitresult[2]
            classlabels  = sort(mach.fitresult[3])
            featurenames = MLJ.report(mach).features
            solem        = solemodel(MLJ.fitted_params(mach).stumps; weights, classlabels, featurenames)
            apply!(solem, X, y)
            return solem
        end,
        (mach, X, y) -> begin
            weights      = mach.fitresult.fitresult[2]
            classlabels  = sort(mach.fitresult.fitresult[3])
            featurenames = MLJ.report(mach).best_report.features
            solem        = solemodel(MLJ.fitted_params(mach).best_fitted_params.stumps; weights, classlabels, featurenames)
            apply!(solem, X, y)
            return solem
        end
    )

    tuning = SoleXplorer.TuningParams(
        SoleXplorer.TuningStrategy(latinhypercube, (ntour = 20,)),
        TUNING_PARAMS[:classification],
        (
            model -> MLJ.range(model, :n_iter, lower=5, upper=50),
            model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
        )
    )

    rulesparams = RulesParams(:intrees, NamedTuple())

    return ModelSetup{TypeABC}(
        type,
        config,
        params,
        DEFAULT_FEATS,
        nothing,
        winparams,
        rawmodel,
        learn_method,
        tuning,
        rulesparams,
        DEFAULT_PREPROC
    )
end

# REGRESSOR ------------------------------------------------------------------ #
function DecisionTreeRegressorModel()::ModelSetup{TypeDTR}
    type = MLJDecisionTreeInterface.DecisionTreeRegressor
    config  = (algo=:regression, type=DecisionTree, treatment=:aggregate, rawapply=DT.apply_tree)

    params = (;
        max_depth              = -1,
        min_samples_leaf       = 5,
        min_samples_split      = 2,
        min_purity_increase    = 0.0,
        n_subfeatures          = 0,
        post_prune             = false,
        merge_purity_threshold = 1.0,
        feature_importance     = :impurity, # :impurity or :split
        rng                    = Random.TaskLocalRNG()
    )

    winparams = WinParams(wholewindow, NamedTuple())

    rawmodel = (
        mach -> MLJ.fitted_params(mach).tree,
        mach -> MLJ.fitted_params(mach).best_fitted_params.tree
    )

    learn_method = (
        (mach, X, y) -> (solem = solemodel(MLJ.fitted_params(mach).tree); apply!(solem, X, y); solem),
        (mach, X, y) -> (solem = solemodel(MLJ.fitted_params(mach).best_fitted_params.tree); apply!(solem, X, y); solem)
    )

    tuning = SoleXplorer.TuningParams(
        SoleXplorer.TuningStrategy(latinhypercube, (ntour = 20,)),
        TUNING_PARAMS[:regression],
        (
            model -> MLJ.range(model, :merge_purity_threshold, lower=0., upper=2.0),
            model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
        )
    )

    rulesparams = RulesParams(:intrees, NamedTuple())

    return ModelSetup{TypeDTR}(
        type,
        config,
        params,
        DEFAULT_FEATS,
        nothing,
        winparams,
        rawmodel,
        learn_method,
        tuning,
        rulesparams,
        DEFAULT_PREPROC
    )
end

function RandomForestRegressorModel()::ModelSetup{TypeRFR}
    type   = MLJDecisionTreeInterface.RandomForestRegressor
    config = (algo=:regression, type=DecisionEnsemble, treatment=:aggregate, rawapply=DT.apply_forest)

    params = (;
        max_depth           = -1,
        min_samples_leaf    = 1,
        min_samples_split   = 2,
        min_purity_increase = 0.0,
        n_subfeatures       = -1,
        n_trees             = 100,
        sampling_fraction   = 0.7,
        feature_importance  = :impurity,
        rng                 = Random.TaskLocalRNG()
    )

    winparams = WinParams(wholewindow, NamedTuple())

    rawmodel = (
        mach -> MLJ.fitted_params(mach).forest,
        mach -> MLJ.fitted_params(mach).best_fitted_params.forest
    )

    learn_method = (
        (mach, X, y) -> begin
            featurenames = MLJ.report(mach).features
            solem        = solemodel(MLJ.fitted_params(mach).forest; featurenames)
            apply!(solem, X, y)
            return solem
        end,
        (mach, X, y) -> begin
            featurenames = MLJ.report(mach).best_report.features
            solem        = solemodel(MLJ.fitted_params(mach).best_fitted_params.forest; featurenames)
            apply!(solem, X, y)
            return solem
        end
    )

    tuning = SoleXplorer.TuningParams(
        SoleXplorer.TuningStrategy(latinhypercube, (ntour = 20,)),
        TUNING_PARAMS[:regression],
        (
            model -> MLJ.range(model, :sampling_fraction, lower=0.3, upper=0.9),
            model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
        )
    )

    rulesparams = RulesParams(:intrees, NamedTuple())

    return ModelSetup{TypeRFR}(
        type,
        config,
        params,
        DEFAULT_FEATS,
        nothing,
        winparams,
        rawmodel,
        learn_method,
        tuning,
        rulesparams,
        DEFAULT_PREPROC
    )
end

