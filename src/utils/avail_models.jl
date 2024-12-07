# ---------------------------------------------------------------------------- #
#                              available models                                #
# ---------------------------------------------------------------------------- #
const AVAIL_MODELS = Dict(
    :decision_tree => (
        method = MLJDecisionTreeInterface.DecisionTreeClassifier,

        model_params = (;
            max_depth=-1, 
            min_samples_leaf=1, 
            min_samples_split=2, 
            min_purity_increase=0.0, 
            n_subfeatures=0, 
            post_prune=false, 
            merge_purity_threshold=1.0, 
            display_depth=5, 
            feature_importance=:impurity, 
            rng=Random.TaskLocalRNG()
        ),

        model_type = DecisionTree,
        learn_method = (mach, X, y) -> (dt = solemodel(MLJ.fitted_params(mach).tree); apply!(dt, X, y); dt),
        tune_learn_method = (mach, X, y) -> (dt = solemodel(MLJ.fitted_params(mach).best_fitted_params.tree); apply!(dt, X, y); dt),


        data_treatment = :aggregate,
        default_features = [maximum, minimum, mean], # magari catch9
        default_treatment = wholewindow,
        treatment_params = (;),

        ranges = [
            model -> MLJ.range(model, :merge_purity_threshold, lower=0, upper=1),
            model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
        ],
    ),

    :modal_decision_tree => (
        method = ModalDecisionTree,

        model_params = (;
            max_depth=nothing, 
            min_samples_leaf=4, 
            min_purity_increase=0.002, 
            max_purity_at_leaf=Inf, 
            max_modal_depth=nothing, 
            relations=nothing, 
            features=nothing, 
            conditions=nothing, 
            featvaltype=Float64, 
            initconditions=nothing, 
            # downsize=SoleData.var"#downsize#482"(), 
            print_progress=false, 
            display_depth=nothing, 
            min_samples_split=nothing, 
            n_subfeatures=identity, 
            post_prune=false, 
            merge_purity_threshold=nothing, 
            feature_importance=:split,
            rng=Random.TaskLocalRNG()
        ),

        model_type = DecisionTree,
        learn_method = (mach, X, y) -> ((_, dt) = MLJ.report(mach).sprinkle(X, y); dt),
        tune_learn_method = (mach, X, y) -> ((_, dt) = MLJ.report(mach).best_report.sprinkle(X, y); dt),

        data_treatment = :reducesize,
        default_features = [mean],
        default_treatment = adaptivewindow,
        treatment_params = (nwindows=10, relative_overlap=0.3),

        ranges = [
            model -> MLJ.range(model, :merge_purity_threshold, lower=0, upper=1),
            model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
        ]
    ),

    :modal_decision_list => (
        method = ModalDecisionLists.MLJInterface.ExtendedSequentialCovering,

        model_params = (;
            searchmethod = BeamSearch(conjuncts_search_method=AtomSearch(), beam_width=3),
            loss_function=ModalDecisionLists.LossFunctions.entropy, 
            discretizedomain=false, 
            max_infogain_ratio=1.0, 
            significance_alpha=0.0, 
            min_rule_coverage=1, 
            max_rulebase_length=nothing, 
            suppress_parity_warning=false,
        ),

        model_type = DecisionList,
        learn_method = (mach, X, y) -> (dt = solemodel(MLJ.fitted_params(mach).tree); apply!(dt, X, y); dt),
        tune_learn_method = (mach, X, y) -> (dt = solemodel(MLJ.fitted_params(mach).best_fitted_params.tree); apply!(dt, X, y); dt),
        
        data_treatment = :aggregate,
        default_features = [maximum, minimum, mean], # magari catch9
        default_treatment = wholewindow,
        treatment_params = (;),

        ranges = [
            model -> MLJ.range(model, :beam_width, lower=3, upper=25),
            model -> MLJ.range(model, :discretizedomain, values=[:false, :true])
        ]
    ),
)
