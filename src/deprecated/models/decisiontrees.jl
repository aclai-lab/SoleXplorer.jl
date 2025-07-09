# ---------------------------------------------------------------------------- #
#                     models from DecisionTrees package                        #
# ---------------------------------------------------------------------------- #

# CLASSIFIER ----------------------------------------------------------------- #
function DecisionTreeClassifierModel()::ModelSetup{AbstractClassification}
    type = MLJDecisionTreeInterface.DecisionTreeClassifier
    config  = (type=DecisionTree, treatment=:aggregate, rawapply=DT.apply_tree)

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
        nothing,
        winparams,
        rawmodel,
        tuning,
        resultsparams,
        rulesparams,
        DEFAULT_PREPROC,
        nothing,
    )
end

function RandomForestClassifierModel()::ModelSetup{AbstractClassification}
    type   = MLJDecisionTreeInterface.RandomForestClassifier
    config = (type=DecisionEnsemble, treatment=:aggregate, rawapply=DT.apply_forest)

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
        nothing,
        winparams,
        rawmodel,
        tuning,
        resultsparams,
        rulesparams,
        DEFAULT_PREPROC,
        nothing,
    )
end

function AdaBoostClassifierModel()::ModelSetup{AbstractClassification}
    type   = MLJDecisionTreeInterface.AdaBoostStumpClassifier
    config = (type=DecisionEnsemble, treatment=:aggregate, rawapply=DT.apply_adaboost_stumps)

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
        nothing,
        winparams,
        rawmodel,
        tuning,
        resultsparams,
        rulesparams,
        DEFAULT_PREPROC,
        nothing,
    )
end

# REGRESSOR ------------------------------------------------------------------ #
function DecisionTreeRegressorModel()::ModelSetup{AbstractRegression}
    type = MLJDecisionTreeInterface.DecisionTreeRegressor
    config  = (type=DecisionTree, treatment=:aggregate, rawapply=DT.apply_tree)

    winparams = WinParams(wholewindow, NamedTuple())

    rawmodel = (
        mach -> MLJ.fitted_params(mach).tree,
        mach -> MLJ.fitted_params(mach).best_fitted_params.tree
    )

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
        nothing,
        winparams,
        rawmodel,
        tuning,
        resultsparams,
        rulesparams,
        DEFAULT_PREPROC,
        nothing,
    )
end

function RandomForestRegressorModel()::ModelSetup{AbstractRegression}
    type   = MLJDecisionTreeInterface.RandomForestRegressor
    config = (type=DecisionEnsemble, treatment=:aggregate, rawapply=DT.apply_forest)

    winparams = WinParams(wholewindow, NamedTuple())

    rawmodel = (
        mach -> MLJ.fitted_params(mach).forest,
        mach -> MLJ.fitted_params(mach).best_fitted_params.forest
    )

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
        nothing,
        winparams,
        rawmodel,
        tuning,
        resultsparams,
        rulesparams,
        DEFAULT_PREPROC,
        nothing,
    )
end

