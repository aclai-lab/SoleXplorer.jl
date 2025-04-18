# References
# early stopping: 
# https://xgboost.readthedocs.io/en/latest/tutorials/model.html#early-stopping
# https://mljar.com/blog/xgboost-early-stopping/

# ---------------------------------------------------------------------------- #
#                   models from ModalDecisionTrees package                     #
# ---------------------------------------------------------------------------- #
get_encoding(classes_seen) = Dict(MMI.int(c) => c for c in MMI.classes(classes_seen))
get_classlabels(encoding)  = [string(encoding[i]) for i in sort(keys(encoding) |> collect)]

function makewatchlist(ds::Dataset)
    isempty(ds.Xvalid) && throw(ArgumentError("No validation data provided, use preprocess valid_ratio parameter"))

    _Xtrain, _Xvalid, _ytrain, _yvalid = ds.Xtrain isa AbstractVector ? 
            (ds.Xtrain[1], ds.Xvalid[1], ds.ytrain[1], ds.yvalid[1]) :
            (ds.Xtrain, ds.Xvalid, ds.ytrain, ds.yvalid)
            
    y_coded_train = @. CategoricalArrays.levelcode(_ytrain) - 1 # convert to 0-based indexing
    y_coded_valid = @. CategoricalArrays.levelcode(_yvalid) - 1 # convert to 0-based indexing
    dtrain        = XGB.DMatrix((_Xtrain, y_coded_train); feature_names=ds.info.vnames)
    dvalid        = XGB.DMatrix((_Xvalid, y_coded_valid); feature_names=ds.info.vnames)

    OrderedDict(["train" => dtrain, "eval" => dvalid])
end

function XGBoostClassifierModel()
    type   = MLJXGBoostInterface.XGBoostClassifier
    config = (; algo=:classification, type=DecisionEnsemble, treatment=:aggregate)

    params = (;
        test                        = 1, 
        num_round                   = 10, 
        booster                     = "gbtree", 
        disable_default_eval_metric = 0, 
        eta                         = 0.3,      # alias: learning_rate
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

    winparams = SoleFeatures.WinParams(SoleBase.wholewindow, NamedTuple())

    learn_method = (
        (mach, X, y) -> begin
            trees        = XGB.trees(mach.fitresult[1])
            encoding     = get_encoding(mach.fitresult[2])
            classlabels  = get_classlabels(encoding)
            featurenames = mach.report.vals[1].features
            solem        = solemodel(trees, @views(Matrix(X)), @views(y); classlabels, featurenames)
            apply!(solem, mapcols(col -> Float32.(col), X), @views(y))
            return solem
        end,
        (mach, X, y) -> begin
            trees        = XGB.trees(mach.fitresult.fitresult[1])
            encoding     = get_encoding(mach.fitresult.fitresult[2])
            classlabels  = get_classlabels(encoding)
            featurenames = mach.fitresult.report.vals[1].features
            solem        = solemodel(trees, @views(Matrix(X)), @views(y); classlabels, featurenames)
            apply!(solem, mapcols(col -> Float32.(col), X), @views(y))
            return solem
        end
    )

    tuning = SoleXplorer.TuningParams(
        SoleXplorer.TuningStrategy(latinhypercube, (ntour = 20,)),
        TUNING_PARAMS[:classification],
        (
            model -> MLJ.range(model, :eta, lower=0.1, upper=0.9),
            model -> MLJ.range(model, :gamma, lower=0.0, upper=1.0),
        )
    )

    rulesparams = RulesParams(InTreesRuleExtractor(), NamedTuple())

    return ModelSetup{TypeXGC}(
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
    