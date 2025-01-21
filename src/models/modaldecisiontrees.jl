# ---------------------------------------------------------------------------- #
#                   models from ModalDecisionTrees package                     #
# ---------------------------------------------------------------------------- #
function ModalDecisionTreeModel()
    type = ModalDecisionTrees.ModalDecisionTree
    config  = (; algo=:classification, type=DecisionTree, treatment=:reducesize)

    params = (;
        max_depth              = nothing, 
        min_samples_leaf       = 4, 
        min_purity_increase    = 0.002, 
        max_purity_at_leaf     = Inf, 
        max_modal_depth        = nothing, 
        relations              = :IA7, 
        features               = DEFAULT_FEATS, 
        conditions             = nothing, 
        featvaltype            = Float64, 
        initconditions         = nothing, 
        downsize               = true, 
        force_i_variables      = true, 
        fixcallablenans        = true, 
        print_progress         = false, 
        rng                    = Random.TaskLocalRNG(), 
        display_depth          = nothing, 
        min_samples_split      = nothing, 
        n_subfeatures          = identity, 
        post_prune             = false, 
        merge_purity_threshold = nothing, 
        feature_importance     = :split,
    )

    winparams = (type=SoleBase.adaptivewindow, nwindows=20)

    learn_method = (
        (mach, X, y) -> ((_, dt) = MLJ.report(mach).sprinkle(X, y); dt),
        (mach, X, y) -> ((_, dt) = MLJ.report(mach).best_report.sprinkle(X, y); dt)
    )

    tuning = (
        tuning        = false,
        method        = (type = latinhypercube, ntour = 20),
        params        = TUNING_PARAMS,
        ranges        = [
            model -> MLJ.range(:n_iter; lower=5, upper=15),
            model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
        ]
    )

    rules_method = SoleModels.PlainRuleExtractor()

    return SymbolicModelSet(
        type,
        config,
        params,
        nothing,
        winparams,
        learn_method,
        tuning,
        rules_method,
        DEFAULT_PREPROC
    )
end

function ModalRandomForestModel()
    type   = ModalDecisionTrees.ModalRandomForest
    config = (; algo=:classification, type=DecisionForest, treatment=:reducesize)

    params = (;
        sampling_fraction      = 0.7, 
        ntrees                 = 10, 
        max_depth              = nothing, 
        min_samples_leaf       = 1, 
        min_purity_increase    = -Inf, 
        max_purity_at_leaf     = Inf, 
        max_modal_depth        = nothing, 
        relations              = :IA7, 
        features               = DEFAULT_FEATS, 
        conditions             = nothing, 
        featvaltype            = Float64, 
        initconditions         = nothing, 
        downsize               = true, 
        force_i_variables      = true, 
        fixcallablenans        = true, 
        print_progress         = false, 
        rng                    = Random.TaskLocalRNG(), 
        display_depth          = nothing, 
        min_samples_split      = nothing, 
        n_subfeatures          = ModalDecisionTrees.MLJInterface.sqrt_f, 
        post_prune             = false, 
        merge_purity_threshold = nothing, 
        feature_importance     = :split
    )

    winparams = (type=SoleBase.adaptivewindow, nwindows=20)

    learn_method = (
        (mach, X, y) -> ((_, dt) = MLJ.report(mach).sprinkle(X, y); dt),
        (mach, X, y) -> ((_, dt) = MLJ.report(mach).best_report.sprinkle(X, y); dt)
    )

    tuning = (
        tuning        = false,
        method        = (type = latinhypercube, ntour = 20),
        params        = TUNING_PARAMS,
        ranges        = [
            model -> MLJ.range(:n_iter; lower=5, upper=15),
            model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
        ]
    )

    rules_method = SolePostHoc.LumenRuleExtractor()
    # rules_method = SoleModels.PlainRuleExtractor() ### Lumen does not currently support symbolic feature names.

    return SymbolicModelSet(
        type,
        config,
        params,
        nothing,
        winparams,
        learn_method,
        tuning,
        rules_method,
        DEFAULT_PREPROC
    )
end

function ModalAdaBoostModel()
    type   = ModalDecisionTrees.ModalAdaBoost
    config = (; algo=:classification, type=DecisionEnsemble, treatment=:reducesize)

    params = (;
        max_depth              = 1, 
        min_samples_leaf       = 1, 
        min_purity_increase    = 0.0, 
        max_purity_at_leaf     = Inf, 
        max_modal_depth        = nothing, 
        relations              = :IA7, 
        features               = DEFAULT_FEATS, 
        conditions             = nothing, 
        featvaltype            = Float64, 
        initconditions         = nothing, 
        downsize               = true, 
        force_i_variables      = true, 
        fixcallablenans        = true, 
        print_progress         = false, 
        rng                    = Random.TaskLocalRNG(), 
        display_depth          = nothing, 
        min_samples_split      = nothing, 
        n_subfeatures          = identity, 
        post_prune             = false, 
        merge_purity_threshold = nothing, 
        feature_importance     = :split, 
        n_iter                 = 10
    )

    winparams = (type=SoleBase.adaptivewindow, nwindows=20)

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
            model -> MLJ.range(:n_iter; lower=5, upper=15),
            model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
        ]
    )

    # rules_method = SolePostHoc.LumenRuleExtractor()
    rules_method = SoleModels.PlainRuleExtractor() ### Lumen does not currently support symbolic feature names.

    return SymbolicModelSet(
        type,
        config,
        params,
        nothing,
        winparams,
        learn_method,
        tuning,
        rules_method,
        DEFAULT_PREPROC
    )
end
