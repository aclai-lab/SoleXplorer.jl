# ---------------------------------------------------------------------------- #
#                   models from MDT package                     #
# ---------------------------------------------------------------------------- #

# CLASSIFIER ----------------------------------------------------------------- #
function ModalDecisionTreeModel()
    type = MDT.ModalDecisionTree
    config  = (algo=:classification, type=DecisionTree, treatment=:reducesize, reducefunc=MLJ.mean, rawapply=MDT.apply)

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

    winparams = WinParams(adaptivewindow, NamedTuple())

    rawmodel = (
        mach -> MLJ.report(mach).rawmodel,
        mach -> MLJ.report(mach).best_report.rawmodel
    )

    learn_method = (
        (mach, X, y) -> ((_, solem) = MLJ.report(mach).sprinkle(X, y); solem),
        (mach, X, y) -> ((_, solem) = MLJ.report(mach).best_report.sprinkle(X, y); solem)
    )

    tuning = SoleXplorer.TuningParams(
        SoleXplorer.TuningStrategy(latinhypercube, (ntour = 20,)),
        TUNING_PARAMS[:classification],
        (
            model -> MLJ.range(model, :min_samples_leaf, lower=2, upper=6),
            model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
        )
    )

    rulesparams = RulesParams(:intrees, NamedTuple())

    return ModelSetup{TypeMDT}(
        type,
        config,
        params,
        DEFAULT_FEATS,
        nothing,
        winparams,
        rawmodel,
        learn_method,
        tuning,
        rulesparams,
        DEFAULT_PREPROC
    )
end

function ModalRandomForestModel()
    type   = MDT.ModalRandomForest
    config = (algo=:classification, type=MDT.DecisionForest, treatment=:reducesize, reducefunc=MLJ.mean, rawapply=MDT.apply)

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
        n_subfeatures          = MDT.MLJInterface.sqrt_f, 
        post_prune             = false, 
        merge_purity_threshold = nothing, 
        feature_importance     = :split
    )

    winparams = WinParams(adaptivewindow, NamedTuple())

    rawmodel = (
        mach -> MLJ.report(mach).rawmodel,
        mach -> MLJ.report(mach).best_report.rawmodel
    )

    learn_method = (
        (mach, X, y) -> ((_, solem) = MLJ.report(mach).sprinkle(X, y); solem),
        (mach, X, y) -> ((_, solem) = MLJ.report(mach).best_report.sprinkle(X, y); solem)
    )

    tuning = SoleXplorer.TuningParams(
        SoleXplorer.TuningStrategy(latinhypercube, (ntour = 20,)),
        TUNING_PARAMS[:classification],
        (
            model -> MLJ.range(model, :sampling_fraction, lower=0.3, upper=0.9),
            model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
        )
    )

    rulesparams = RulesParams(:intrees, NamedTuple())

    return ModelSetup{TypeMRF}(
        type,
        config,
        params,
        DEFAULT_FEATS,
        nothing,
        winparams,
        rawmodel,
        learn_method,
        tuning,
        rulesparams,
        DEFAULT_PREPROC
    )
end

function ModalAdaBoostModel()
    type   = MDT.ModalAdaBoost
    config = (algo=:classification, type=DecisionEnsemble, treatment=:reducesize, reducefunc=MLJ.mean, rawapply=MDT.apply)

    params = (;
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
        n_subfeatures          = MDT.MLJInterface.sqrt_f,
        post_prune             = false, 
        merge_purity_threshold = nothing, 
        feature_importance     = :split, 
        n_iter                 = 10
    )

    winparams = WinParams(adaptivewindow, NamedTuple())

    rawmodel = (
        mach -> MLJ.report(mach).rawmodel,
        mach -> MLJ.report(mach).best_report.rawmodel
    )

    learn_method = (
        (mach, X, y) -> ((_, solem) = MLJ.report(mach).sprinkle(X, y); solem),
        (mach, X, y) -> ((_, solem) = MLJ.report(mach).best_report.sprinkle(X, y); solem)
    )

    tuning = SoleXplorer.TuningParams(
        SoleXplorer.TuningStrategy(latinhypercube, (ntour = 20,)),
        TUNING_PARAMS[:classification],
        (
            model -> MLJ.range(model, :min_samples_leaf, lower=1, upper=3),
            model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
        )
    )

    rulesparams = RulesParams(:intrees, NamedTuple())

    return ModelSetup{TypeMAB}(
        type,
        config,
        params,
        DEFAULT_FEATS,
        nothing,
        winparams,
        rawmodel,
        learn_method,
        tuning,
        rulesparams,
        DEFAULT_PREPROC
    )
end
