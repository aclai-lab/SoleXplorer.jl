# ---------------------------------------------------------------------------- #
#                     models from DecisionTrees package                        #
# ---------------------------------------------------------------------------- #

# CLASSIFIER ----------------------------------------------------------------- #
function DecisionTreeClassifierModel()
    type = MLJDecisionTreeInterface.DecisionTreeClassifier
    config  = (algo=:classification, type=DecisionTree, treatment=:aggregate)

    params = (;
        max_depth              = -1,
        min_samples_leaf       = 1,
        min_samples_split      = 2,
        min_purity_increase    = 0.0,
        n_subfeatures          = 0,
        post_prune             = false,
        merge_purity_threshold = 1.0,
        display_depth          = 5,
        feature_importance     = :impurity,
        rng                    = Random.TaskLocalRNG()
    )

    winparams = SoleFeatures.WinParams(SoleBase.wholewindow, NamedTuple())

    learn_method = (
        (mach, X, y) -> (dt = solemodel(MLJ.fitted_params(mach).tree); apply!(dt, X, y); dt),
        (mach, X, y) -> (dt = solemodel(MLJ.fitted_params(mach).best_fitted_params.tree); apply!(dt, X, y); dt)
    )

    tuning = SoleXplorer.TuningParams(
        SoleXplorer.TuningStrategy(latinhypercube, (ntour = 20,)),
        TUNING_PARAMS[:classification],
        (
            model -> MLJ.range(model, :merge_purity_threshold, lower=0., upper=2.0),
            model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
        )
    )

    rulesparams = RulesParams(PlainRuleExtractor(), NamedTuple())

    return ModelSetup(
        type,
        config,
        params,
        DEFAULT_FEATS,
        nothing,
        winparams,
        learn_method,
        tuning,
        rulesparams,
        DEFAULT_PREPROC
    )
end

function RandomForestClassifierModel()
    type   = MLJDecisionTreeInterface.RandomForestClassifier
    config = (algo=:classification, type=DecisionEnsemble, treatment=:aggregate)

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

    winparams = SoleFeatures.WinParams(SoleBase.wholewindow, NamedTuple())

    learn_method = (
        (mach, X, y) -> begin
            classlabels  = (mach).fitresult[2][sortperm((mach).fitresult[3])]
            featurenames = MLJ.report(mach).features
            dt           = solemodel(MLJ.fitted_params(mach).forest; classlabels, featurenames)
            apply!(dt, X, y)
            return dt
        end,
        (mach, X, y) -> begin
            classlabels  = (mach).fitresult.fitresult[2][sortperm((mach).fitresult.fitresult[3])]
            featurenames = MLJ.report(mach).best_report.features
            dt           = solemodel(MLJ.fitted_params(mach).best_fitted_params.forest; classlabels, featurenames)
            apply!(dt, X, y)
            return dt
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

    rulesparams = RulesParams(InTreesRuleExtractor(), NamedTuple())

    return ModelSetup(
        type,
        config,
        params,
        DEFAULT_FEATS,
        nothing,
        winparams,
        learn_method,
        tuning,
        rulesparams,
        DEFAULT_PREPROC
    )
end

function AdaBoostClassifierModel()
    type   = MLJDecisionTreeInterface.AdaBoostStumpClassifier
    config = (algo=:classification, type=DecisionEnsemble, treatment=:aggregate)

    params = (;
        n_iter             = 10,
        feature_importance = :impurity,
        rng                = Random.TaskLocalRNG()
    )

    winparams = SoleFeatures.WinParams(SoleBase.wholewindow, NamedTuple())

    learn_method = (
        (mach, X, y) -> begin
            weights      = mach.fitresult[2]
            classlabels  = sort(mach.fitresult[3])
            featurenames = MLJ.report(mach).features
            dt           = solemodel(MLJ.fitted_params(mach).stumps; weights, classlabels, featurenames)
            apply!(dt, X, y)
            return dt
        end,
        (mach, X, y) -> begin
            weights      = mach.fitresult.fitresult[2]
            classlabels  = sort(mach.fitresult.fitresult[3])
            featurenames = MLJ.report(mach).best_report.features
            dt           = solemodel(MLJ.fitted_params(mach).best_fitted_params.stumps; weights, classlabels, featurenames)
            apply!(dt, X, y)
            return dt
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

    rulesparams = RulesParams(InTreesRuleExtractor(), NamedTuple())

    return ModelSetup(
        type,
        config,
        params,
        DEFAULT_FEATS,
        nothing,
        winparams,
        learn_method,
        tuning,
        rulesparams,
        DEFAULT_PREPROC
    )
end

# REGRESSOR ------------------------------------------------------------------ #
function DecisionTreeRegressorModel()
    type = MLJDecisionTreeInterface.DecisionTreeRegressor
    config  = (algo=:regression, type=DecisionTree, treatment=:aggregate)

    params = (;
        max_depth              = -1,
        min_samples_leaf       = 5,
        min_samples_split      = 2,
        min_purity_increase    = 0.0,
        n_subfeatures          = 0,
        post_prune             = false,
        merge_purity_threshold = 1.0,
        feature_importance     = :impurity,
        rng                    = Random.TaskLocalRNG()
    )

    winparams = SoleFeatures.WinParams(SoleBase.wholewindow, NamedTuple())

    learn_method = (
        (mach, X, y) -> (dt = solemodel(MLJ.fitted_params(mach).tree); apply!(dt, X, y); dt),
        (mach, X, y) -> (dt = solemodel(MLJ.fitted_params(mach).best_fitted_params.tree); apply!(dt, X, y); dt)
    )

    tuning = SoleXplorer.TuningParams(
        SoleXplorer.TuningStrategy(latinhypercube, (ntour = 20,)),
        TUNING_PARAMS[:regression],
        (
            model -> MLJ.range(model, :merge_purity_threshold, lower=0., upper=2.0),
            model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
        )
    )

    rulesparams = RulesParams(PlainRuleExtractor(), NamedTuple())

    return ModelSetup(
        type,
        config,
        params,
        DEFAULT_FEATS,
        nothing,
        winparams,
        learn_method,
        tuning,
        rulesparams,
        DEFAULT_PREPROC
    )
end

function RandomForestRegressorModel()
    type   = MLJDecisionTreeInterface.RandomForestRegressor
    config = (algo=:regression, type=DecisionEnsemble, treatment=:aggregate)

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

    winparams = SoleFeatures.WinParams(SoleBase.wholewindow, NamedTuple())

    learn_method = (
        (mach, X, y) -> begin
            featurenames = MLJ.report(mach).features
            dt           = solemodel(MLJ.fitted_params(mach).forest; featurenames)
            apply!(dt, X, y)
            return dt
        end,
        (mach, X, y) -> begin
            featurenames = MLJ.report(mach).best_report.features
            dt           = solemodel(MLJ.fitted_params(mach).best_fitted_params.forest; featurenames)
            apply!(dt, X, y)
            return dt
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

    rulesparams = RulesParams(InTreesRuleExtractor(), NamedTuple())

    return ModelSetup(
        type,
        config,
        params,
        DEFAULT_FEATS,
        nothing,
        winparams,
        learn_method,
        tuning,
        rulesparams,
        DEFAULT_PREPROC
    )
end

