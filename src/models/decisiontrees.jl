# ---------------------------------------------------------------------------- #
#                     models from DecisionTrees package                        #
# ---------------------------------------------------------------------------- #

# CLASSIFIER ----------------------------------------------------------------- #
function DecisionTreeClassifierModel()::ModelSetup{AbstractClassification}
    type = MLJDecisionTreeInterface.DecisionTreeClassifier
    config  = (type=DecisionTree, treatment=:aggregate, rawapply=DT.apply_tree)

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

    tuning = SoleXplorer.TuningParams(
        SoleXplorer.TuningStrategy(latinhypercube, (ntour = 20,)),
        TUNING_PARAMS[AbstractClassification],
        (
            model -> MLJ.range(model, :merge_purity_threshold, lower=0., upper=2.0),
            model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
        )
    )

    resultsparams = (m) -> (m.info.supporting_labels, m.info.supporting_predictions)

    rulesparams = RulesParams(:intrees, NamedTuple())

    return ModelSetup{AbstractClassification}(
        type,
        config,
        params,
        DEFAULT_FEATS,
        nothing,
        winparams,
        rawmodel,
        tuning,
        resultsparams,
        rulesparams,
        DEFAULT_PREPROC,
        DEFAULT_MEAS,
    )
end

function RandomForestClassifierModel()::ModelSetup{AbstractClassification}
    type   = MLJDecisionTreeInterface.RandomForestClassifier
    config = (type=DecisionEnsemble, treatment=:aggregate, rawapply=DT.apply_forest)

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

    tuning = SoleXplorer.TuningParams(
        SoleXplorer.TuningStrategy(latinhypercube, (ntour = 20,)),
        TUNING_PARAMS[AbstractClassification],
        (
            model -> MLJ.range(model, :sampling_fraction, lower=0.3, upper=0.9),
            model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
        )
    )

    resultsparams = (m) -> (m.info.supporting_labels, m.info.supporting_predictions)

    rulesparams = RulesParams(:intrees, NamedTuple())

    return ModelSetup{AbstractClassification}(
        type,
        config,
        params,
        DEFAULT_FEATS,
        nothing,
        winparams,
        rawmodel,
        tuning,
        resultsparams,
        rulesparams,
        DEFAULT_PREPROC,
        DEFAULT_MEAS,
    )
end

function AdaBoostClassifierModel()::ModelSetup{AbstractClassification}
    type   = MLJDecisionTreeInterface.AdaBoostStumpClassifier
    config = (type=DecisionEnsemble, treatment=:aggregate, rawapply=DT.apply_adaboost_stumps)

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

    tuning = SoleXplorer.TuningParams(
        SoleXplorer.TuningStrategy(latinhypercube, (ntour = 20,)),
        TUNING_PARAMS[AbstractClassification],
        (
            model -> MLJ.range(model, :n_iter, lower=5, upper=50),
            model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
        )
    )

    resultsparams = (m) -> (m.info.supporting_labels, m.info.supporting_predictions)

    rulesparams = RulesParams(:intrees, NamedTuple())

    return ModelSetup{AbstractClassification}(
        type,
        config,
        params,
        DEFAULT_FEATS,
        nothing,
        winparams,
        rawmodel,
        tuning,
        resultsparams,
        rulesparams,
        DEFAULT_PREPROC,
        DEFAULT_MEAS,
    )
end

# REGRESSOR ------------------------------------------------------------------ #
function DecisionTreeRegressorModel()::ModelSetup{AbstractRegression}
    type = MLJDecisionTreeInterface.DecisionTreeRegressor
    config  = (type=DecisionTree, treatment=:aggregate, rawapply=DT.apply_tree)

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

    # learn_method = (
    #     (mach, X, y) -> (solem = solemodel(MLJ.fitted_params(mach).tree); apply!(solem, X, y); solem),
    #     (mach, X, y) -> (solem = solemodel(MLJ.fitted_params(mach).best_fitted_params.tree); apply!(solem, X, y); solem)
    # )

    tuning = SoleXplorer.TuningParams(
        SoleXplorer.TuningStrategy(latinhypercube, (ntour = 20,)),
        TUNING_PARAMS[AbstractRegression],
        (
            model -> MLJ.range(model, :merge_purity_threshold, lower=0., upper=2.0),
            model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
        )
    )

    resultsparams = (m) -> (m.info.supporting_labels, m.info.supporting_predictions)

    rulesparams = RulesParams(:intrees, NamedTuple())

    return ModelSetup{AbstractRegression}(
        type,
        config,
        params,
        DEFAULT_FEATS,
        nothing,
        winparams,
        rawmodel,
        # learn_method,
        tuning,
        resultsparams,
        rulesparams,
        DEFAULT_PREPROC,
        DEFAULT_MEAS,
    )
end

function RandomForestRegressorModel()::ModelSetup{AbstractRegression}
    type   = MLJDecisionTreeInterface.RandomForestRegressor
    config = (type=DecisionEnsemble, treatment=:aggregate, rawapply=DT.apply_forest)

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

    # learn_method = (
    #     (mach, X, y) -> begin
    #         featurenames = MLJ.report(mach).features
    #         solem        = solemodel(MLJ.fitted_params(mach).forest; featurenames)
    #         apply!(solem, X, y)
    #         return solem
    #     end,
    #     (mach, X, y) -> begin
    #         featurenames = MLJ.report(mach).best_report.features
    #         solem        = solemodel(MLJ.fitted_params(mach).best_fitted_params.forest; featurenames)
    #         apply!(solem, X, y)
    #         return solem
    #     end
    # )

    tuning = SoleXplorer.TuningParams(
        SoleXplorer.TuningStrategy(latinhypercube, (ntour = 20,)),
        TUNING_PARAMS[AbstractRegression],
        (
            model -> MLJ.range(model, :sampling_fraction, lower=0.3, upper=0.9),
            model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
        )
    )

    resultsparams = (m) -> (m.info.supporting_labels, m.info.supporting_predictions)

    rulesparams = RulesParams(:intrees, NamedTuple())

    return ModelSetup{AbstractRegression}(
        type,
        config,
        params,
        DEFAULT_FEATS,
        nothing,
        winparams,
        rawmodel,
        # learn_method,
        tuning,
        resultsparams,
        rulesparams,
        DEFAULT_PREPROC,
        DEFAULT_MEAS,
    )
end

