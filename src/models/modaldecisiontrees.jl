# ---------------------------------------------------------------------------- #
#                   models from ModalDecisionTrees package                     #
# ---------------------------------------------------------------------------- #

# CLASSIFIER ----------------------------------------------------------------- #
function ModalDecisionTreeModel()
    type = ModalDecisionTrees.ModalDecisionTree
    config  = (algo=:classification, type=DecisionTree, treatment=:reducesize)

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

    winparams = SoleFeatures.WinParams(SoleBase.adaptivewindow, NamedTuple())

    learn_method = (
        (mach, X, y) -> ((_, dt) = MLJ.report(mach).sprinkle(X, y); dt),
        (mach, X, y) -> ((_, dt) = MLJ.report(mach).best_report.sprinkle(X, y); dt)
    )

    tuning = (
        tuning = false,
        method = (; type = latinhypercube, ntour = 20),
        params = TUNING_PARAMS[:classification],
        ranges = [
            model -> MLJ.range(model, :merge_purity_threshold, lower=0, upper=1),
            model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
        ]
    )

    rulesparams = RulesParams(PlainRuleExtractor(), NamedTuple())

    return ModelSetup(
        type,
        config,
        params,
        DEFAULT_FEATS,
        winparams,
        learn_method,
        tuning,
        rulesparams,
        DEFAULT_PREPROC
    )
end

function ModalRandomForestModel()
    type   = ModalDecisionTrees.ModalRandomForest
    config = (algo=:classification, type=DecisionForest, treatment=:reducesize)

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

    winparams = SoleFeatures.WinParams(SoleBase.adaptivewindow, NamedTuple())

    learn_method = (
        (mach, X, y) -> ((_, dt) = MLJ.report(mach).sprinkle(X, y); dt),
        (mach, X, y) -> ((_, dt) = MLJ.report(mach).best_report.sprinkle(X, y); dt)
    )

    tuning = (
        tuning = false,
        method = (; type = latinhypercube, ntour = 20),
        params = TUNING_PARAMS[:classification],
        ranges = [
            model -> MLJ.range(model, :sampling_fraction, lower=0.3, upper=0.9),
            model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
        ]
    )

    rulesparams = RulesParams(InTreesRuleExtractor(), NamedTuple())

    return ModelSetup(
        type,
        config,
        params,
        DEFAULT_FEATS,
        winparams,
        learn_method,
        tuning,
        rulesparams,
        DEFAULT_PREPROC
    )
end

function ModalAdaBoostModel()
    type   = ModalDecisionTrees.ModalAdaBoost
    config = (algo=:classification, type=DecisionEnsemble, treatment=:reducesize)

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
        min_samples_split      = 2, 
        n_subfeatures          = ModalDecisionTrees.MLJInterface.sqrt_f,
        post_prune             = false, 
        merge_purity_threshold = nothing, 
        feature_importance     = :split, 
        n_iter                 = 10
    )

    winparams = SoleFeatures.WinParams(SoleBase.adaptivewindow, NamedTuple())

    learn_method = (
        (mach, X, y) -> ((_, dt) = MLJ.report(mach).sprinkle(X, y); dt),
        (mach, X, y) -> ((_, dt) = MLJ.report(mach).best_report.sprinkle(X, y); dt)
    )

    tuning = (
        tuning = false,
        method = (; type = latinhypercube, ntour = 20),
        params = TUNING_PARAMS[:classification],
        ranges = [
            model -> MLJ.range(:n_iter; lower=5, upper=15),
            model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
        ]
    )

    rulesparams = RulesParams(InTreesRuleExtractor(), NamedTuple())

    return ModelSetup(
        type,
        config,
        params,
        DEFAULT_FEATS,
        winparams,
        learn_method,
        tuning,
        rulesparams,
        DEFAULT_PREPROC
    )
end
