# ---------------------------------------------------------------------------- #
#                   models from ModalDecisionTrees package                     #
# ---------------------------------------------------------------------------- #

function XGBoostModel()
    type   = MLJXGBoostInterface.XGBoostClassifier
    config = (; algo=:classification, type=DecisionEnsemble, treatment=:aggregate)

    params = (;
        test                        = 1, 
        num_round                   = 100, 
        booster                     = "gbtree", 
        disable_default_eval_metric = 0, 
        eta                         = 0.3, 
        num_parallel_tree           = 1, 
        gamma                       = 0.0, 
        max_depth                   = 6, 
        min_child_weight            = 1.0, 
        max_delta_step              = 0.0, 
        subsample                   = 1.0, 
        colsample_bytree            = 1.0, 
        colsample_bylevel           = 1.0, 
        colsample_bynode            = 1.0, 
        lambda                      = 1.0, 
        alpha                       = 0.0, 
        tree_method                 = "auto", 
        sketch_eps                  = 0.03, 
        scale_pos_weight            = 1.0, 
        updater                     = nothing, 
        refresh_leaf                = 1, 
        process_type                = "default", 
        grow_policy                 = "depthwise", 
        max_leaves                  = 0, 
        max_bin                     = 256, 
        predictor                   = "cpu_predictor", 
        sample_type                 = "uniform", 
        normalize_type              = "tree", 
        rate_drop                   = 0.0, 
        one_drop                    = 0, 
        skip_drop                   = 0.0, 
        feature_selector            = "cyclic", 
        top_k                       = 0, 
        tweedie_variance_power      = 1.5, 
        objective                   = "automatic", 
        base_score                  = 0.5, 
        early_stopping_rounds       = 0, 
        watchlist                   = nothing, 
        nthread                     = 1, 
        importance_type             = "gain", 
        seed                        = nothing, 
        validate_parameters         = false, 
        eval_metric                 = String[]
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
            model -> MLJ.range(model, :max_depth, lower=3, upper=6),
            model -> MLJ.range(model, :sample_type, values=["uniform", "weighted"])
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
    