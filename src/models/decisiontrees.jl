# ---------------------------------------------------------------------------- #
#                     models from DecisionTrees package                        #
# ---------------------------------------------------------------------------- #
function DecisionTreeModel()
    model = MLJDecisionTreeInterface.DecisionTreeClassifier
    type  = (; algo=:classification, type=DecisionTree, treatment=:aggregate)

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

    features  = DEFAULT_FEATS
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

    return ModelSet(
        model,
        type,
        params,
        features,
        winparams,
        learn_method,
        tuning,
        rules_method
    )
end



