# ---------------------------------------------------------------------------- #
#                     models from DecisionTrees package                        #
# ---------------------------------------------------------------------------- #
function DecisionTreeModel()
    type = MLJDecisionTreeInterface.DecisionTreeClassifier
    config  = (; algo=:classification, type=DecisionTree, treatment=:aggregate)

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

    winparams = (type=SoleBase.wholewindow,)

    learn_method = (
        (mach, X, y) -> (dt = solemodel(MLJ.fitted_params(mach).tree); apply!(dt, X, y); dt),
        (mach, X, y) -> (dt = solemodel(MLJ.fitted_params(mach).best_fitted_params.tree); apply!(dt, X, y); dt)
    )

    tuning = (
        tuning        = false,
        method        = (type = latinhypercube, ntour = 20),
        params        = TUNING_PARAMS,
        ranges        = [
            model -> MLJ.range(model, :merge_purity_threshold, lower=0, upper=1),
            model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
        ]
    )

    rules_method = SoleModels.PlainRuleExtractor()

    return SymbolicModelSet(
        type,
        config,
        params,
        DEFAULT_FEATS,
        winparams,
        learn_method,
        tuning,
        rules_method,
        DEFAULT_PREPROC
    )
end

function RandomForestModel()
    type   = MLJDecisionTreeInterface.RandomForestClassifier
    config = (; algo=:classification, type=DecisionEnsemble, treatment=:aggregate)

    params = (;
        max_depth              = -1,
        min_samples_leaf       = 1, 
        min_samples_split      = 2, 
        min_purity_increase    = 0.0, 
        n_subfeatures          = -1,
        n_trees                = 100,
        sampling_fraction      = 0.7,
        feature_importance     = :impurity,
        rng                    = Random.TaskLocalRNG()
    )

    winparams = (type=SoleBase.wholewindow,)

    learn_method = (
        (mach, X, y) -> begin
            classlabels = (mach).fitresult[2][sortperm((mach).fitresult[3])]
            featurenames = MLJ.report(mach).features
            dt = solemodel(MLJ.fitted_params(mach).forest; classlabels, featurenames)
            apply!(dt, X, y)
            return dt
        end,
        (mach, X, y) -> begin
            classlabels = (mach).fitresult.fitresult[2][sortperm((mach).fitresult.fitresult[3])]
            featurenames = MLJ.report(mach).best_report.features
            dt = solemodel(MLJ.fitted_params(mach).best_fitted_params.forest; classlabels, featurenames)
            apply!(dt, X, y)
            return dt
        end
    )

    tuning = (
        tuning        = false,
        method        = (type = latinhypercube, ntour = 20),
        params        = TUNING_PARAMS,
        ranges        = [
            model -> MLJ.range(model, :merge_purity_threshold, lower=0, upper=1),
            model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
        ]
    )

    rules_method = SolePostHoc.LumenRuleExtractor()

    return SymbolicModelSet(
        type,
        config,
        params,
        DEFAULT_FEATS,
        winparams,
        learn_method,
        tuning,
        rules_method,
        DEFAULT_PREPROC
    )
end

function AdaBoostModel()
    type   = MLJDecisionTreeInterface.AdaBoostStumpClassifier
    config = (; algo=:classification, type=DecisionEnsemble, treatment=:aggregate)

    params = (;
        n_iter             = 10,
        feature_importance = :impurity,
        rng                = Random.TaskLocalRNG()
    )

    winparams = (type=SoleBase.wholewindow,)

    learn_method = (
        (mach, X, y) -> begin
            weights = mach.fitresult[2]
            classlabels = sort(mach.fitresult[3])
            featurenames = MLJ.report(mach).features
            dt = solemodel(MLJ.fitted_params(mach).stumps; weights, classlabels, featurenames)
            apply!(dt, X, y)
            return dt
        end,
        (mach, X, y) -> begin
            weights = mach.fitresult.fitresult[2]
            classlabels = sort(mach.fitresult.fitresult[3])
            featurenames = MLJ.report(mach).best_report.features
            dt = solemodel(MLJ.fitted_params(mach).best_fitted_params.stumps; weights, classlabels, featurenames)
            apply!(dt, X, y)
            return dt
        end
    )

    tuning = (
        tuning        = false,
        method        = (type = latinhypercube, ntour = 20),
        params        = TUNING_PARAMS,
        ranges        = [
            model -> MLJ.range(model, :merge_purity_threshold, lower=0, upper=1),
            model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
        ]
    )

    rules_method = SolePostHoc.LumenRuleExtractor()

    return SymbolicModelSet(
        type,
        config,
        params,
        DEFAULT_FEATS,
        winparams,
        learn_method,
        tuning,
        rules_method,
        DEFAULT_PREPROC
    )
end
