# References
# early stopping: 
# https://xgboost.readthedocs.io/en/latest/tutorials/model.html#early-stopping
# https://mljar.com/blog/xgboost-early-stopping/

# ---------------------------------------------------------------------------- #
#                   models from ModalDecisionTrees package                     #
# ---------------------------------------------------------------------------- #
get_encoding(classes_seen) = Dict(MLJ.int(c) => c for c in MLJ.classes(classes_seen))
get_classlabels(encoding)  = [string(encoding[i]) for i in sort(keys(encoding) |> collect)]

function makewatchlist(ds::Dataset)
    isempty(ds.tt[1].valid) && throw(ArgumentError("No validation data provided, use preprocess valid_ratio parameter"))

    # _Xtrain, _Xvalid, _ytrain, _yvalid = ds.Xtrain isa AbstractVector ? 
    #         (ds.Xtrain[1], ds.Xvalid[1], ds.ytrain[1], ds.yvalid[1]) :
    #         (ds.Xtrain, ds.Xvalid, ds.ytrain, ds.yvalid)
            
    y_coded_train = @. MLJ.levelcode(ds.y[ds.tt[1].train]) - 1 # convert to 0-based indexing
    y_coded_valid = @. MLJ.levelcode(ds.y[ds.tt[1].valid]) - 1 # convert to 0-based indexing
    dtrain        = XGB.DMatrix((ds.X[ds.tt[1].train, :], y_coded_train); feature_names=ds.info.vnames)
    dvalid        = XGB.DMatrix((ds.X[ds.tt[1].valid, :], y_coded_valid); feature_names=ds.info.vnames)

    XGB.OrderedDict(["train" => dtrain, "eval" => dvalid])
end

function XGBoostClassifierModel()::ModelSetup{AbstractClassification}
    type   = MLJXGBoostInterface.XGBoostClassifier
    config = (type=DecisionEnsemble, treatment=:aggregate, rawapply=XGB.predict)

    params = (;
        test                        = 1,
        num_round                   = 10,
        booster                     = "gbtree",
        disable_default_eval_metric = 0,
        eta                         = 0.3,     # alias: learning_rate
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

    winparams = WinParams(wholewindow, NamedTuple())

    rawmodel = (
        mach -> XGB.trees(mach.fitresult[1]),
        mach -> XGB.trees(mach.fitresult.fitresult[1])
    )

    tuning = SoleXplorer.TuningParams(
        SoleXplorer.TuningStrategy(latinhypercube, (ntour = 20,)),
        TUNING_PARAMS[AbstractClassification],
        (
            model -> MLJ.range(model, :eta, lower=0.1, upper=0.9),
            model -> MLJ.range(model, :gamma, lower=0.0, upper=1.0),
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

function XGBoostRegressorModel()::ModelSetup{AbstractRegression}
    type   = MLJXGBoostInterface.XGBoostRegressor
    config = (type=DecisionEnsemble, treatment=:aggregate, rawapply=XGB.predict)

    params = (;
        test                        = 1,
        num_round                   = 10,
        booster                     = "gbtree",
        disable_default_eval_metric = 0,
        eta                         = 0.3,     # alias: learning_rate
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
        objective                   = "reg:squarederror",
        base_score                  = -Inf,
        early_stopping_rounds       = 0,
        watchlist                   = nothing,
        nthread                     = 1,
        importance_type             = "gain",
        seed                        = nothing,
        validate_parameters         = false,
        eval_metric                 = String[]
    )

    winparams = WinParams(wholewindow, NamedTuple())

    rawmodel = (
        mach -> XGB.trees(mach.fitresult[1]),
        mach -> XGB.trees(mach.fitresult.fitresult[1])
    )

    tuning = SoleXplorer.TuningParams(
        SoleXplorer.TuningStrategy(latinhypercube, (ntour = 20,)),
        TUNING_PARAMS[AbstractRegression],
        (
            model -> MLJ.range(model, :eta, lower=0.1, upper=0.9),
            model -> MLJ.range(model, :gamma, lower=0.0, upper=1.0),
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
        tuning,
        resultsparams,
        rulesparams,
        DEFAULT_PREPROC,
        DEFAULT_MEAS,
    )
end
    