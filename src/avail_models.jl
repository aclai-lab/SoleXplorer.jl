# ---------------------------------------------------------------------------- #
#                              available models                                #
# ---------------------------------------------------------------------------- #
# const AVAIL_MODELS = Dict(
AVAIL_MODELS = Dict(
    # ------------------------------------------------------------------------ #
    #                              decision tree                               #
    # ------------------------------------------------------------------------ #
    :decision_tree => (
        method = MLJDecisionTreeInterface.DecisionTreeClassifier,

        model_params = (;
            max_depth = -1,
            min_samples_leaf = 1, 
            min_samples_split = 2, 
            min_purity_increase = 0.0, 
            n_subfeatures = 0,
            post_prune = false,
            merge_purity_threshold = 1.0,
            display_depth = 5,
            feature_importance = :impurity, 
            rng=Random.TaskLocalRNG()
        ),

        model = (; algo = :classification, type = DecisionTree),
        learn_method = (mach, X, y) -> (dt = solemodel(MLJ.fitted_params(mach).tree); apply!(dt, X, y); dt),
        tune_learn_method = (mach, X, y) -> (dt = solemodel(MLJ.fitted_params(mach).best_fitted_params.tree); apply!(dt, X, y); dt),

        data_treatment = :aggregate,
        nested_features = [maximum, minimum, mean], # magari catch9
        nested_treatment = (mode=SoleBase.wholewindow, params=(;)),

        ranges = [
            model -> MLJ.range(model, :merge_purity_threshold, lower=0, upper=1),
            model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
        ],

        rules_method = Sole.listrules
    ),

    # ------------------------------------------------------------------------ #
    #                           modal decision tree                            #
    # ------------------------------------------------------------------------ #
    :modal_decision_tree => (
        method = ModalDecisionTrees.ModalDecisionTree,

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
            print_progress=false, 
            display_depth=nothing, 
            min_samples_split=nothing, 
            n_subfeatures=identity, 
            post_prune=false, 
            merge_purity_threshold=nothing, 
            feature_importance=:split,
            rng=Random.TaskLocalRNG()
        ),

        model = (; algo = :classification, type = DecisionTree),
        learn_method = (mach, X, y) -> ((_, dt) = MLJ.report(mach).sprinkle(X, y); dt),
        tune_learn_method = (mach, X, y) -> ((_, dt) = MLJ.report(mach).best_report.sprinkle(X, y); dt),

        data_treatment = :reducesize,
        nested_features = [mean],
        nested_treatment = (mode=SoleBase.adaptivewindow, params=(nwindows=10, relative_overlap=0.3)),

        ranges = [
            model -> MLJ.range(model, :merge_purity_threshold, lower=0, upper=1),
            model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
        ],

        rules_method = Sole.listrules
    ),

    # ------------------------------------------------------------------------ #
    #                             decision forest                              #
    # ------------------------------------------------------------------------ #
    :decision_forest => (
        method = MLJDecisionTreeInterface.RandomForestClassifier,

        model_params = (;
            max_depth = -1,
            min_samples_leaf = 1, 
            min_samples_split = 2, 
            min_purity_increase = 0.0, 
            n_subfeatures = -1, 
            n_trees = 100, 
            sampling_fraction = 0.7, 
            feature_importance = :impurity, 
            rng=Random.TaskLocalRNG()
        ),

        model = (; algo = :classification, type = DecisionEnsemble),
        learn_method = (mach, X, y) -> begin
                classlabels = (mach).fitresult[2][sortperm((mach).fitresult[3])]
                featurenames = MLJ.report(mach).features
                dt = solemodel(MLJ.fitted_params(mach).forest; classlabels, featurenames)
                apply!(dt, X, y)
                return dt
            end,
        tune_learn_method = (mach, X, y) -> begin
                classlabels = (mach).fitresult.fitresult[2][sortperm((mach).fitresult.fitresult[3])]
                featurenames = MLJ.report(mach).best_report.features
                dt = solemodel(MLJ.fitted_params(mach).best_fitted_params.forest; classlabels, featurenames)
                apply!(dt, X, y)
                return dt
            end,

        data_treatment = :aggregate,
        nested_features = [maximum, minimum, mean],
        nested_treatment = (mode=SoleBase.wholewindow, params=(;)),

        ranges = [
            model -> MLJ.range(:sampling_fraction; lower=0.5, upper=0.8),
            model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
        ],

        rules_method = Sole.listrules
    ),

    # ------------------------------------------------------------------------ #
    #                           adaboost classifier                            #
    # ------------------------------------------------------------------------ #
    :adaboost => (
        method = MLJDecisionTreeInterface.AdaBoostStumpClassifier,

        model_params = (;
            n_iter=10, 
            feature_importance=:impurity, 
            rng=Random.TaskLocalRNG(),
        ),

        model = (; algo = :classification, type = DecisionEnsemble),
        learn_method = (mach, X, y) -> begin
                weights = mach.fitresult[2]
                classlabels = sort(mach.fitresult[3])
                featurenames = MLJ.report(mach).features
                dt = solemodel(MLJ.fitted_params(mach).stumps; weights, classlabels, featurenames)
                apply!(dt, X, y)
                return dt
            end,
        tune_learn_method = (mach, X, y) -> begin
                weights = mach.fitresult.fitresult[2]
                classlabels = sort(mach.fitresult.fitresult[3])
                featurenames = MLJ.report(mach).best_report.features
                dt = solemodel(MLJ.fitted_params(mach).best_fitted_params.stumps; weights, classlabels, featurenames)
                apply!(dt, X, y)
                return dt
            end,

        data_treatment = :aggregate,
        nested_features = [maximum, minimum, mean],
        nested_treatment = (mode=SoleBase.wholewindow, params=(;)),

        ranges = [
            model -> MLJ.range(:n_iter; lower=5, upper=15),
            model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
        ],

        rules_method = Sole.listrules
    ),

    # ------------------------------------------------------------------------ #
    #                         adaboost modal classifier                        #
    # ------------------------------------------------------------------------ #
    :modal_adaboost => (
        method = ModalDecisionTrees.ModalAdaBoost,

        model_params = (;
            max_purity_at_leaf=Inf, 
            max_modal_depth=nothing, 
            relations=nothing, 
            features=nothing, 
            conditions=nothing, 
            featvaltype=Float64, 
            initconditions=nothing, 
            print_progress=false, 
            display_depth=nothing, 
            n_subfeatures=identity, 
            post_prune=false, 
            merge_purity_threshold=nothing, 
            feature_importance=:split,
            rng=Random.TaskLocalRNG(),
            n_iter=10, 
        ),

        model = (; algo = :classification, type = DecisionEnsemble),
        learn_method = (mach, X, y) -> ((_, dt) = MLJ.report(mach).sprinkle(X, y); dt),
        tune_learn_method = (mach, X, y) -> ((_, dt) = MLJ.report(mach).best_report.sprinkle(X, y); dt),

        data_treatment = :reducesize,
        nested_features = [mean],
        nested_treatment = (mode=SoleBase.adaptivewindow, params=(nwindows=10, relative_overlap=0.3)),

        ranges = [
            model -> MLJ.range(:n_iter; lower=5, upper=15),
            model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
        ],

        rules_method = Sole.listrules
    ),

    # ------------------------------------------------------------------------ #
    #                             regression tree                              #
    # ------------------------------------------------------------------------ #
    :regression_tree => (
        method = MLJDecisionTreeInterface.DecisionTreeRegressor,

        model_params = (;
        max_depth=-1, 
        min_samples_leaf=5, 
        min_samples_split=2, 
        min_purity_increase=0.0, 
        n_subfeatures=0, 
        post_prune=false, 
        merge_purity_threshold=1.0, 
        feature_importance=:impurity, 
        rng=Random.TaskLocalRNG()
        ),

        model = (; algo = :regression, type = DecisionTree),
        learn_method = (mach, X, y) -> (dt = solemodel(MLJ.fitted_params(mach).tree); apply!(dt, X, y); dt),
        tune_learn_method = (mach, X, y) -> (dt = solemodel(MLJ.fitted_params(mach).best_fitted_params.tree); apply!(dt, X, y); dt),

        data_treatment = :aggregate,
        nested_features = [maximum, minimum, mean], # magari catch9
        nested_treatment = (mode=SoleBase.wholewindow, params=(;)),

        ranges = [
            model -> MLJ.range(model, :merge_purity_threshold, lower=0, upper=1),
            model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
        ],

        rules_method = Sole.listrules
    ),
    # ------------------------------------------------------------------------ #
    #                            regression forest                             #
    # ------------------------------------------------------------------------ #
    :regression_forest => (
        method = MLJDecisionTreeInterface.RandomForestRegressor,

        model_params = (;
            max_depth = -1,
            min_samples_leaf = 1, 
            min_samples_split = 2, 
            min_purity_increase = 0.0, 
            n_subfeatures = -1, 
            n_trees = 100, 
            sampling_fraction = 0.7, 
            feature_importance = :impurity, 
            rng=Random.TaskLocalRNG()
        ),

        model = (; algo = :regression, type = DecisionEnsemble),
        learn_method = (mach, X, y) -> (dt = solemodel(MLJ.fitted_params(mach).tree); apply!(dt, X, y); dt),
        tune_learn_method = (mach, X, y) -> (dt = solemodel(MLJ.fitted_params(mach).best_fitted_params.tree); apply!(dt, X, y); dt),

        data_treatment = :aggregate,
        nested_features = [maximum, minimum, mean], # magari catch9
        nested_treatment = (mode=SoleBase.wholewindow, params=(;)),

        ranges = [
            model -> MLJ.range(model, :merge_purity_threshold, lower=0, upper=1),
            model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
        ],

        rules_method = Sole.listrules
    ),

    # ------------------------------------------------------------------------ #
    #                           modal decision list                            #
    # ------------------------------------------------------------------------ #
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

        model = (; algo = :classification, type = DecisionList),
        learn_method = (mach, X, y) -> begin
            MLJ.fitted_params(mach).fitresult.model
            #TODO: not working yet
        end,
        tune_learn_method = (mach, X, y) -> (dt = solemodel(MLJ.fitted_params(mach).best_fitted_params.tree); apply!(dt, X, y); dt),
        
        data_treatment = :aggregate,
        nested_features = [maximum, minimum, mean], # magari catch9
        nested_treatment = (mode=SoleBase.wholewindow, params=(;)),

        ranges = [
            model -> MLJ.range(model, :beam_width, lower=3, upper=25),
            model -> MLJ.range(model, :discretizedomain, values=[:false, :true])
        ],

        rules_method = Sole.listrules
    ),

    # ------------------------------------------------------------------------ #
    #                                  xgboost                                 #
    # ------------------------------------------------------------------------ #
    # :xgboost => (
    #     method = MLJXGBoostInterface.XGBoostClassifier,

    #     model_params = (;
    #         booster="gbtree",
    #         num_round=100,
    #         max_depth=6,
    #         # eval_metric="mlogloss",
    #         eta=0.3,
    #         alpha=0,
    #         gamma=0,
    #         lambda=1,
    #     ),

    #     model = (; algo = :classification, type = Vector{DecisionTree}),
    #     learn_method = (mach, X, y) -> begin
    #             dt = MLJXGBoostInterface.solemodel(MLJ.fitted_params(mach)...)
    #             # for d in dt
    #             #     apply!(d, X, y)
    #             # end
    #             dt
    #         end,
    #     # TODO
    #     tune_learn_method = (mach, X, y) -> (dt = solemodel(MLJ.fitted_params(mach).best_fitted_params.tree); apply!(dt, X, y); dt),

    #     data_treatment = :aggregate,
    #     nested_features = [maximum, minimum, mean], # magari catch9
    #     nested_treatment = (mode=SoleBase.wholewindow, params=(;)),

    #     ranges = [
    #         model -> MLJ.range(model, :merge_purity_threshold, lower=0, upper=1),
    #         model -> MLJ.range(model, :feature_importance, values=[:impurity, :split])
    #     ],

    #     rules_method = Sole.listrules
    # ),
)
