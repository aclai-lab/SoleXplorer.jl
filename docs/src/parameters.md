prepare_dataset(
    X:: dataset, format accepted: DataFrame.
    y:: vector of targets, could be for classification tasks: Symbols, Strings, CategoricalValues or, for regression tasks: Numbers.
    
    optional parameter # 1
    model
    define the model (type) and relative (params)

    if is model is unspecified, then it will be used default decision_tree model
    model:: (
        type:: Symbol, model are going to be used, could be:
            :decisiontree, :randomforest, :adaboost,
            :modaldecisiontree, :modalrandomforest, :modaladaboost,
            :xgboost
            Sole can automatically recognize if a dataset if for classification or regression,
            but in case, we can force the specific task expliciting the model type:
                :decisiontree_classifier => DecisionTreeClassifierModel,
                :randomforest_classifier => RandomForestClassifierModel,
                :adaboost_classifier     => AdaBoostClassifierModel,

                :decisiontree_regressor  => DecisionTreeRegressorModel,
                :randomforest_regressor  => RandomForestRegressorModel,

                :modaldecisiontree       => ModalDecisionTreeModel,
                :modalrandomforest       => ModalRandomForestModel,
                :modaladaboost           => ModalAdaBoostModel,

                :xgboost_classifier      => XGBoostClassifierModel,
                :xgboost_regressor       => XGBoostRegressorModel,

        params:: model specific parameters
        
        DecisionTreeClassifierModel()
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

        RandomForestClassifierModel()
            params = (;
                max_depth           = -1,
                min_samples_leaf    = 1,
                min_samples_split   = 2,
                min_purity_increase = 0.0,
                n_subfeatures       = -1,
                n_trees             = 100,
                sampling_fraction   = 0.7,
                feature_importance  = :impurity,
                rng                 = Random.TaskLocalRNG()
            )

        AdaBoostClassifierModel()
            params = (;
                n_iter             = 10,
                feature_importance = :impurity,
                rng                = Random.TaskLocalRNG()
            )

        DecisionTreeRegressorModel()
            params = (;
                max_depth              = -1,
                min_samples_leaf       = 5,
                min_samples_split      = 2,
                min_purity_increase    = 0.0,
                n_subfeatures          = 0,
                post_prune             = false,
                merge_purity_threshold = 1.0,
                feature_importance     = :impurity,
                rng                    = Random.TaskLocalRNG()
            )

        RandomForestRegressorModel()
            params = (;
                max_depth           = -1,
                min_samples_leaf    = 1,
                min_samples_split   = 2,
                min_purity_increase = 0.0,
                n_subfeatures       = -1,
                n_trees             = 100,
                sampling_fraction   = 0.7,
                feature_importance  = :impurity,
                rng                 = Random.TaskLocalRNG()
            )

        ModalDecisionTreeModel()
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

        ModalRandomForestModel()
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

        ModalAdaBoostModel()
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

        XGBoostClassifierModel()
            params = (;
                test                        = 1, 
                num_round                   = 100, 
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
    )

    optional parameter # 2
    resample

    define the resample strategy (type) and relative (params)
    for more details check MLJ resample

    resample = (
        type:: Function, resample strategy are going to be used, could be:
        CV, Holdout, StratifiedCV, TimeSeriesCV

        params:: resample strategy specific parameters

        CV()
            params = (nfolds = 6, shuffle = true, rng = TaskLocalRNG())

        Holdout()
            params = (fraction_train = 0.7, shuffle = true, rng = TaskLocalRNG())

        StratifiedCV()
            Params = (nfolds = 6, shuffle = true, rng = TaskLocalRNG())

        TimeSeriesCV()
            params = (nfolds = 4,)
    )

    optional parameter # 3
    win

    this parameter is only for multi dimensional dataset, ie: time series
    as now, it only supports dataset composed of time series (vectors) elements.
    performs a windowing of elements to reduce size.

    win = (
        type:: Function, windowing strategy, could be:
        movingwindow, wholewindow, splitwindow, adaptivewindow

        params:: 

        movingwindow()
            params = (window_size = 1024, window_step = 512)

        wholewindow()
            params = ()

        splitwindow()
            Params = (nwindows = 5,)

        adaptivewindow()
            params = (nwindows = 5, relative_overlap = 0.1)
    )

    optional parameter # 4
    features

    function features to be used in train/test

    features=(
        can be every kind of function like:
        maximum, minimum, mean, median, std, StatsBase.cov
        
        Sole uses also catch22 functions:
        emode_5, mode_10, embedding_dist, acf_timescale, acf_first_min, ami2, trev, outlier_timing_pos
        outlier_timing_neg, whiten_timescale, forecast_error, ami_timescale, high_fluctuation, stretch_decreasing
        stretch_high, entropy_pairs, rs_range, dfa, low_freq_power, centroid_freq, transition_variance, periodicity

        and there's also available boundle of functions:
        base_set, catch9, catch22_set, complete_set
    )
)