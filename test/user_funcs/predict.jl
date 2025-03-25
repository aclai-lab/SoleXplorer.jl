using Test
using Sole
using SoleXplorer
using Random, StatsBase, JLD2, DataFrames
using MLJ, MLJTuning
using RDatasets
using StatisticalMeasures

# ---------------------------------------------------------------------------- #
#                                CLASSIFICATION                                #
# ---------------------------------------------------------------------------- #
X, y       = SoleData.load_arff_dataset("NATOPS")
train_seed = 11
rng        = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

# @testset "basic usage traintest function" begin
#     @testset "decisiontree_classifier" begin
        model = traintest(X, y; models=(type=:decisiontree_classifier, params=(; rng=rng)))

        preds = MLJ.predict(model.mach, model.ds.Xtest)
        yhat = MLJ.mode.(preds)
        
        # MultiClassification
        MLJ.accuracy(yhat, model.ds.ytest)                              # Accuracy
        MLJ.balanced_accuracy(yhat, model.ds.ytest)                     # BalancedAccuracy
        MLJ.confusion_matrix(yhat, model.ds.ytest)                      # ConfusionMatrix
        MLJ.kappa(yhat, model.ds.ytest)                                 # Kappa
        MLJ.macro_f1score(yhat, model.ds.ytest)                         # MulticlassFScore
        MLJ.matthews_correlation(yhat, model.ds.ytest)                  # MatthewsCorrelation
        MLJ.micro_f1score(yhat, model.ds.ytest)                         # MulticlassFScore
        MLJ.misclassification_rate(yhat, model.ds.ytest)                # MisclassificationRate
        MLJ.multiclass_false_discovery_rate(yhat, model.ds.ytest)       # MulticlassFalseDiscoveryRate
        MLJ.multiclass_false_negative_rate(yhat, model.ds.ytest)        # MulticlassFalseNegativeRate
        MLJ.multiclass_false_negative(yhat, model.ds.ytest)             # MulticlassFalseNegative
        MLJ.multiclass_false_positive_rate(yhat, model.ds.ytest)        # MulticlassFalsePositiveRate
        MLJ.multiclass_false_positive(yhat, model.ds.ytest)             # MulticlassFalsePositive
        MLJ.multiclass_hit_rate(yhat, model.ds.ytest)                   # MulticlassTruePositiveRate
        MLJ.multiclass_miss_rate(yhat, model.ds.ytest)                  # MulticlassFalseNegativeRate
        MLJ.multiclass_negative_predictive_value(yhat, model.ds.ytest)  # MulticlassNegativePredictiveValue
        MLJ.multiclass_precision(yhat, model.ds.ytest)                  # MulticlassPositivePredictiveValue
        MLJ.multiclass_selectivity(yhat, model.ds.ytest)                # MulticlassTrueNegativeRate
        MLJ.multiclass_sensitivity(yhat, model.ds.ytest)                # MulticlassTruePositiveRate
        MLJ.multiclass_specificity(yhat, model.ds.ytest)                # MulticlassTrueNegativeRate
        MLJ.multiclass_true_negative(yhat, model.ds.ytest)              # MulticlassTrueNegative
        MLJ.multiclass_true_positive(yhat, model.ds.ytest)              # MulticlassTruePositive

filename = "respiratory_Pneumonia.jld2"
filepath = joinpath(@__DIR__, filename)
df = jldopen(filepath)
X, y = df["X"], df["y"]
rng = Random.Xoshiro(1)
train_seed = 11;

# @testset "basic usage traintest function" begin
#     @testset "decisiontree_classifier" begin
        model = traintest(X, y; models=(type=:decisiontree_classifier, params=(; rng=rng)))

        preds = MLJ.predict(model.mach, model.ds.Xtest)
        yhat = MLJ.mode.(preds)

        # BinaryClassification
        MLJ.accuracy(yhat, model.ds.ytest) #	Accuracy
        MLJ.balanced_accuracy(yhat, model.ds.ytest) #	BalancedAccuracy
        MLJ.confusion_matrix(yhat, model.ds.ytest) #	ConfusionMatrix
        MLJ.f1score(yhat, model.ds.ytest) #	FScore
        MLJ.fallout(yhat, model.ds.ytest) #	FalsePositiveRate
        MLJ.false_discovery_rate(yhat, model.ds.ytest) #	FalseDiscoveryRate
        MLJ.false_negative_rate(yhat, model.ds.ytest) #	FalseNegativeRate
        MLJ.false_negative(yhat, model.ds.ytest) #	FalseNegative
        MLJ.false_positive_rate(yhat, model.ds.ytest) #	FalsePositiveRate
        MLJ.false_positive(yhat, model.ds.ytest) #	FalsePositive
        MLJ.hit_rate(yhat, model.ds.ytest) #	TruePositiveRate
        MLJ.kappa(yhat, model.ds.ytest) #	Kappa
        MLJ.macro_f1score(yhat, model.ds.ytest) #	MulticlassFScore
        MLJ.matthews_correlation(yhat, model.ds.ytest) #	MatthewsCorrelation
        MLJ.mcc(yhat, model.ds.ytest) #	MatthewsCorrelation
        MLJ.mcr(yhat, model.ds.ytest) #	MisclassificationRate
        MLJ.micro_f1score(yhat, model.ds.ytest) #	MulticlassFScore
        MLJ.misclassification_rate(yhat, model.ds.ytest) #	MisclassificationRate
        MLJ.miss_rate(yhat, model.ds.ytest) #	FalseNegativeRate
        MLJ.multiclass_f1score(yhat, model.ds.ytest) #	MulticlassFScore
        MLJ.multiclass_fallout(yhat, model.ds.ytest) #	MulticlassFalsePositiveRate
        MLJ.multiclass_false_discovery_rate(yhat, model.ds.ytest) #	MulticlassFalseDiscoveryRate
        MLJ.multiclass_false_negative_rate(yhat, model.ds.ytest) #	MulticlassFalseNegativeRate
        MLJ.multiclass_false_negative(yhat, model.ds.ytest) #	MulticlassFalseNegative
        MLJ.multiclass_false_positive_rate(yhat, model.ds.ytest) #	MulticlassFalsePositiveRate
        MLJ.multiclass_false_positive(yhat, model.ds.ytest) #	MulticlassFalsePositive
        MLJ.multiclass_hit_rate(yhat, model.ds.ytest) #	MulticlassTruePositiveRate
        MLJ.multiclass_miss_rate(yhat, model.ds.ytest) #	MulticlassFalseNegativeRate
        MLJ.multiclass_negative_predictive_value(yhat, model.ds.ytest) #	MulticlassNegativePredictiveValue
        MLJ.multiclass_negativepredictive_value(yhat, model.ds.ytest) #	MulticlassNegativePredictiveValue
        MLJ.multiclass_npv(yhat, model.ds.ytest) #	MulticlassNegativePredictiveValue
        MLJ.multiclass_positive_predictive_value(yhat, model.ds.ytest) #	MulticlassPositivePredictiveValue
        MLJ.multiclass_positivepredictive_value(yhat, model.ds.ytest) #	MulticlassPositivePredictiveValue
        MLJ.multiclass_ppv(yhat, model.ds.ytest) #	MulticlassPositivePredictiveValue
        MLJ.multiclass_precision(yhat, model.ds.ytest) #	MulticlassPositivePredictiveValue
        MLJ.multiclass_recall(yhat, model.ds.ytest) #	MulticlassTruePositiveRate
        MLJ.multiclass_selectivity(yhat, model.ds.ytest) #	MulticlassTrueNegativeRate
        MLJ.multiclass_sensitivity(yhat, model.ds.ytest) #	MulticlassTruePositiveRate
        MLJ.multiclass_specificity(yhat, model.ds.ytest) #	MulticlassTrueNegativeRate
        MLJ.multiclass_tnr(yhat, model.ds.ytest) #	MulticlassTrueNegativeRate
        MLJ.multiclass_tpr(yhat, model.ds.ytest) #	MulticlassTruePositiveRate
        MLJ.multiclass_true_negative_rate(yhat, model.ds.ytest) #	MulticlassTrueNegativeRate
        MLJ.multiclass_true_negative(yhat, model.ds.ytest) #	MulticlassTrueNegative
        MLJ.multiclass_true_positive_rate(yhat, model.ds.ytest) #	MulticlassTruePositiveRate
        MLJ.multiclass_true_positive(yhat, model.ds.ytest) #	MulticlassTruePositive
        MLJ.multiclass_truenegative_rate(yhat, model.ds.ytest) #	MulticlassTrueNegativeRate
        MLJ.multiclass_truenegative(yhat, model.ds.ytest) #	MulticlassTrueNegative
        MLJ.multiclass_truepositive_rate(yhat, model.ds.ytest) #	MulticlassTruePositiveRate
        MLJ.multiclass_truepositive(yhat, model.ds.ytest) #	MulticlassTruePositive

        MLJ.negative_predictive_value(yhat, model.ds.ytest) #	NegativePredictiveValue
        MLJ.negativepredictive_value(yhat, model.ds.ytest) #	NegativePredictiveValue
        MLJ.npv(yhat, model.ds.ytest) #	NegativePredictiveValue
        MLJ.positive_predictive_value(yhat, model.ds.ytest) #	PositivePredictiveValue
        MLJ.positivepredictive_value(yhat, model.ds.ytest) #	PositivePredictiveValue
        MLJ.ppv(yhat, model.ds.ytest) #	PositivePredictiveValue
        MLJ.precision(yhat, model.ds.ytest) #	PositivePredictiveValue
        MLJ.recall(yhat, model.ds.ytest) #	TruePositiveRate
        MLJ.selectivity(yhat, model.ds.ytest) #	TrueNegativeRate
        MLJ.sensitivity(yhat, model.ds.ytest) #	TruePositiveRate
        MLJ.specificity(yhat, model.ds.ytest) #	TrueNegativeRate
        MLJ.tnr(yhat, model.ds.ytest) #	TrueNegativeRate
        MLJ.tpr(yhat, model.ds.ytest) #	TruePositiveRate
        MLJ.true_negative_rate(yhat, model.ds.ytest) #	TrueNegativeRate
        MLJ.true_negative(yhat, model.ds.ytest) #	TrueNegative
        MLJ.true_positive_rate(yhat, model.ds.ytest) #	TruePositiveRate
        MLJ.true_positive(yhat, model.ds.ytest) #	TruePositive
        MLJ.truenegative_rate(yhat, model.ds.ytest) #	TrueNegativeRate
        MLJ.truenegative(yhat, model.ds.ytest) #	TrueNegative
        MLJ.truepositive_rate(yhat, model.ds.ytest) #	TruePositiveRate
        MLJ.truepositive(yhat, model.ds.ytest) # TruePositive

table = RDatasets.dataset("datasets", "LifeCycleSavings")
y = table[:, :DDPI]
X = select(table, Not([:DDPI, :Country]));
train_seed = 11
rng = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

# @testset "basic usage traintest function" begin
#     @testset "decisiontree_classifier" begin
        model = traintest(X, y; models=(type=:decisiontree_regressor, params=(; rng=rng)))

        preds = MLJ.predict(model.mach, model.ds.Xtest)
        yhat = MLJ.mode.(preds)


        # regression?
        # MLJ.accuracy(yhat, model.ds.ytest) #	Accuracy
        # MLJ.area_under_curve(yhat, model.ds.ytest) #	AreaUnderCurve
        # MLJ.auc(yhat, model.ds.ytest) #	AreaUnderCurve
        # MLJ.bac(yhat, model.ds.ytest) #	BalancedAccuracy
        # MLJ.bacc(yhat, model.ds.ytest) #	BalancedAccuracy
        # MLJ.balanced_accuracy(yhat, model.ds.ytest) #	BalancedAccuracy
        # MLJ.brier_loss(yhat, model.ds.ytest) #	BrierLoss
        # MLJ.brier_score(yhat, model.ds.ytest) #	BrierScore
        # MLJ.confmat(yhat, model.ds.ytest) #	ConfusionMatrix
        # MLJ.confusion_matrix(yhat, model.ds.ytest) #	ConfusionMatrix
        # MLJ.cross_entropy(yhat, model.ds.ytest) #	LogLoss
        # MLJ.cross_entropy(yhat, model.ds.ytest) #	BrierLoss
        # MLJ.f1score(yhat, model.ds.ytest) #	FScore
        # MLJ.fallout(yhat, model.ds.ytest) #	FalsePositiveRate
        # MLJ.false_discovery_rate(yhat, model.ds.ytest) #	FalseDiscoveryRate
        # MLJ.false_negative_rate(yhat, model.ds.ytest) #	FalseNegativeRate
        # MLJ.false_negative(yhat, model.ds.ytest) #	FalseNegative
        # MLJ.false_positive_rate(yhat, model.ds.ytest) #	FalsePositiveRate
        # MLJ.false_positive(yhat, model.ds.ytest) #	FalsePositive
        # MLJ.falsediscovery_rate(yhat, model.ds.ytest) #	FalseDiscoveryRate
        # MLJ.falsenegative_rate(yhat, model.ds.ytest) #	FalseNegativeRate
        # MLJ.falsenegative(yhat, model.ds.ytest) #	FalseNegative
        # MLJ.falsepositive_rate(yhat, model.ds.ytest) #	FalsePositiveRate
        # MLJ.falsepositive(yhat, model.ds.ytest) #	FalsePositive
        # MLJ.fdr(yhat, model.ds.ytest) #	FalseDiscoveryRate
        # MLJ.fnr(yhat, model.ds.ytest) #	FalseNegativeRate
        # MLJ.fpr(yhat, model.ds.ytest) #	FalsePositiveRate
        # MLJ.hit_rate(yhat, model.ds.ytest) #	TruePositiveRate
        # MLJ.kappa(yhat, model.ds.ytest) #	Kappa
        l1_sum(yhat, model.ds.ytest) #	LPSumLoss
        MLJ.l1(yhat, model.ds.ytest) #	LPLoss
        MLJ.l2(yhat, model.ds.ytest) #_sum	LPSumLoss
        MLJ.l2(yhat, model.ds.ytest) #	LPLoss
        MLJ.log_cosh_loss(yhat, model.ds.ytest) #	LogCoshLoss
        MLJ.log_cosh(yhat, model.ds.ytest) #	LogCoshLoss
        # MLJ.log_loss(yhat, model.ds.ytest) #	LogLoss
        # MLJ.log_score(yhat, model.ds.ytest) #	LogScore
        # MLJ.macro_f1score(yhat, model.ds.ytest) #	MulticlassFScore
        MLJ.mae(yhat, model.ds.ytest) #	LPLoss
        MLJ.mape(yhat, model.ds.ytest) #	MeanAbsoluteProportionalError
        # MLJ.matthews_correlation(yhat, model.ds.ytest) #	MatthewsCorrelation
        MLJ.mav(yhat, model.ds.ytest) #	LPLoss
        # MLJ.mcc(yhat, model.ds.ytest) #	MatthewsCorrelation
        # MLJ.mcr(yhat, model.ds.ytest) #	MisclassificationRate
        MLJ.mean_absolute_error(yhat, model.ds.ytest) #	LPLoss
        MLJ.mean_absolute_value(yhat, model.ds.ytest) #	LPLoss
        # MLJ.micro_f1score(yhat, model.ds.ytest) #	MulticlassFScore
        # MLJ.misclassification_rate(yhat, model.ds.ytest) #	MisclassificationRate
        # MLJ.miss_rate(yhat, model.ds.ytest) #	FalseNegativeRate
        # MLJ.multiclass_f1score(yhat, model.ds.ytest) #	MulticlassFScore
        # MLJ.multiclass_fallout(yhat, model.ds.ytest) #	MulticlassFalsePositiveRate
        # MLJ.multiclass_false_discovery_rate(yhat, model.ds.ytest) #	MulticlassFalseDiscoveryRate
        # MLJ.multiclass_false_negative_rate(yhat, model.ds.ytest) #	MulticlassFalseNegativeRate
        # MLJ.multiclass_false_negative(yhat, model.ds.ytest) #	MulticlassFalseNegative
        # MLJ.multiclass_false_positive_rate(yhat, model.ds.ytest) #	MulticlassFalsePositiveRate
        # MLJ.multiclass_false_positive(yhat, model.ds.ytest) #	MulticlassFalsePositive
        # MLJ.multiclass_falsediscovery_rate(yhat, model.ds.ytest) #	MulticlassFalseDiscoveryRate
        # MLJ.multiclass_falsenegative_rate(yhat, model.ds.ytest) #	MulticlassFalseNegativeRate
        # MLJ.multiclass_falsenegative(yhat, model.ds.ytest) #	MulticlassFalseNegative
        # MLJ.multiclass_falsepositive_rate(yhat, model.ds.ytest) #	MulticlassFalsePositiveRate
        # MLJ.multiclass_falsepositive(yhat, model.ds.ytest) #	MulticlassFalsePositive
        # MLJ.multiclass_fdr(yhat, model.ds.ytest) #	MulticlassFalseDiscoveryRate
        # MLJ.multiclass_fnr(yhat, model.ds.ytest) #	MulticlassFalseNegativeRate
        # MLJ.multiclass_fpr(yhat, model.ds.ytest) #	MulticlassFalsePositiveRate
        # MLJ.multiclass_hit_rate(yhat, model.ds.ytest) #	MulticlassTruePositiveRate
        # MLJ.multiclass_miss_rate(yhat, model.ds.ytest) #	MulticlassFalseNegativeRate
        # MLJ.multiclass_negative_predictive_value(yhat, model.ds.ytest) #	MulticlassNegativePredictiveValue
        # MLJ.multiclass_negativepredictive_value(yhat, model.ds.ytest) #	MulticlassNegativePredictiveValue
        # MLJ.multiclass_npv(yhat, model.ds.ytest) #	MulticlassNegativePredictiveValue
        # MLJ.multiclass_positive_predictive_value(yhat, model.ds.ytest) #	MulticlassPositivePredictiveValue
        # MLJ.multiclass_positivepredictive_value(yhat, model.ds.ytest) #	MulticlassPositivePredictiveValue
        # MLJ.multiclass_ppv(yhat, model.ds.ytest) #	MulticlassPositivePredictiveValue
        # MLJ.multiclass_precision(yhat, model.ds.ytest) #	MulticlassPositivePredictiveValue
        # MLJ.multiclass_recall(yhat, model.ds.ytest) #	MulticlassTruePositiveRate
        # MLJ.multiclass_selectivity(yhat, model.ds.ytest) #	MulticlassTrueNegativeRate
        # MLJ.multiclass_sensitivity(yhat, model.ds.ytest) #	MulticlassTruePositiveRate
        # MLJ.multiclass_specificity(yhat, model.ds.ytest) #	MulticlassTrueNegativeRate
        # MLJ.multiclass_tnr(yhat, model.ds.ytest) #	MulticlassTrueNegativeRate
        # MLJ.multiclass_tpr(yhat, model.ds.ytest) #	MulticlassTruePositiveRate
        # MLJ.multiclass_true_negative_rate(yhat, model.ds.ytest) #	MulticlassTrueNegativeRate
        # MLJ.multiclass_true_negative(yhat, model.ds.ytest) #	MulticlassTrueNegative
        # MLJ.multiclass_true_positive_rate(yhat, model.ds.ytest) #	MulticlassTruePositiveRate
        # MLJ.multiclass_true_positive(yhat, model.ds.ytest) #	MulticlassTruePositive
        # MLJ.multiclass_truenegative_rate(yhat, model.ds.ytest) #	MulticlassTrueNegativeRate
        # MLJ.multiclass_truenegative(yhat, model.ds.ytest) #	MulticlassTrueNegative
        # MLJ.multiclass_truepositive_rate(yhat, model.ds.ytest) #	MulticlassTruePositiveRate
        # MLJ.multiclass_truepositive(yhat, model.ds.ytest) #	MulticlassTruePositive
        # multitarget_accuracy(yhat, model.ds.ytest) #	MultitargetAccuracy
        multitarget_l1_sum(yhat, model.ds.ytest) #	MultitargetLPSumLoss
        multitarget_l1(yhat, model.ds.ytest) #	MultitargetLPLoss
        multitarget_l2_sum(yhat, model.ds.ytest) #	MultitargetLPSumLoss
        multitarget_l2(yhat, model.ds.ytest) #	MultitargetLPLoss
        multitarget_mae(yhat, model.ds.ytest) #	MultitargetLPLoss
        multitarget_mape(yhat, model.ds.ytest) #	MultitargetMeanAbsoluteProportionalError
        multitarget_mape(yhat, model.ds.ytest) #	MultitargetLogCoshLoss
        multitarget_mav(yhat, model.ds.ytest) #	MultitargetLPLoss
        multitarget_mcr(yhat, model.ds.ytest) #	MultitargetMisclassificationRate
        multitarget_mean_absolute_error(yhat, model.ds.ytest) #	MultitargetLPLoss
        multitarget_mean_absolute_value(yhat, model.ds.ytest) #	MultitargetLPLoss
        # multitarget_misclassification_rate(yhat, model.ds.ytest) #	MultitargetMisclassificationRate
        multitarget_rms(yhat, model.ds.ytest) #	MultitargetRootMeanSquaredError
        multitarget_rmse(yhat, model.ds.ytest) #	MultitargetRootMeanSquaredError
        multitarget_rmsl(yhat, model.ds.ytest) #	MultitargetRootMeanSquaredLogError
        multitarget_rmsle(yhat, model.ds.ytest) #	MultitargetRootMeanSquaredLogError
        multitarget_rmslp1(yhat, model.ds.ytest) #	MultitargetRootMeanSquaredLogProportionalError
        multitarget_rmsp(yhat, model.ds.ytest) #	MultitargetRootMeanSquaredProportionalError
        multitarget_root_mean_squared_error(yhat, model.ds.ytest) #	MultitargetRootMeanSquaredError
        multitarget_root_mean_squared_log_error(yhat, model.ds.ytest) #	MultitargetRootMeanSquaredLogError
        # MLJ.negative_predictive_value(yhat, model.ds.ytest) #	NegativePredictiveValue
        # MLJ.negativepredictive_value(yhat, model.ds.ytest) #	NegativePredictiveValue
        # MLJ.npv(yhat, model.ds.ytest) #	NegativePredictiveValue
        # MLJ.positive_predictive_value(yhat, model.ds.ytest) #	PositivePredictiveValue
        # MLJ.positivepredictive_value(yhat, model.ds.ytest) #	PositivePredictiveValue
        # MLJ.ppv(yhat, model.ds.ytest) #	PositivePredictiveValue
        # MLJ.precision(yhat, model.ds.ytest) #	PositivePredictiveValue
        # MLJ.probability_of_correct_classification(yhat, model.ds.ytest) #	BalancedAccuracy
        # quadratic_loss(yhat, model.ds.ytest) #	BrierLoss
        # quadratic_score(yhat, model.ds.ytest) #	BrierScore
        # MLJ.recall(yhat, model.ds.ytest) #	TruePositiveRate
        MLJ.rms(yhat, model.ds.ytest) #	RootMeanSquaredError
        MLJ.rmse(yhat, model.ds.ytest) #	RootMeanSquaredError
        MLJ.rmsl(yhat, model.ds.ytest) #	RootMeanSquaredLogError
        MLJ.rmsle(yhat, model.ds.ytest) #	RootMeanSquaredLogError
        MLJ.rmslp1(yhat, model.ds.ytest) #	RootMeanSquaredLogProportionalError
        MLJ.rmsp(yhat, model.ds.ytest) #	RootMeanSquaredProportionalError
        MLJ.root_mean_squared_error(yhat, model.ds.ytest) #	RootMeanSquaredError
        MLJ.root_mean_squared_log_error(yhat, model.ds.ytest) #	RootMeanSquaredLogError
        MLJ.rsq(yhat, model.ds.ytest) #	RSquared
        MLJ.rsquared(yhat, model.ds.ytest) #	RSquared
        # MLJ.selectivity(yhat, model.ds.ytest) #	TrueNegativeRate
        # MLJ.sensitivity(yhat, model.ds.ytest) #	TruePositiveRate
        # MLJ.specificity(yhat, model.ds.ytest) #	TrueNegativeRate
        # MLJ.spherical_score(yhat, model.ds.ytest) #	SphericalScore
        # MLJ.tnr(yhat, model.ds.ytest) #	TrueNegativeRate
        # MLJ.tpr(yhat, model.ds.ytest) #	TruePositiveRate
        # MLJ.true_negative_rate(yhat, model.ds.ytest) #	TrueNegativeRate
        # MLJ.true_negative(yhat, model.ds.ytest) #	TrueNegative
        # MLJ.true_positive_rate(yhat, model.ds.ytest) #	TruePositiveRate
        # MLJ.true_positive(yhat, model.ds.ytest) #	TruePositive
        # MLJ.truenegative_rate(yhat, model.ds.ytest) #	TrueNegativeRate
        # MLJ.truenegative(yhat, model.ds.ytest) #	TrueNegative
        # MLJ.truepositive_rate(yhat, model.ds.ytest) #	TruePositiveRate
        # MLJ.truepositive(yhat, model.ds.ytest) # TruePositive

#     end

#     @testset "randomforest_classifier" begin
#         result = traintest(X, y; models=(type=:randomforest_classifier, params=(; rng=rng)))
#         @test result isa SoleXplorer.Modelset
#         @test result.classifier isa SoleXplorer.RandomForestClassifier # type piracy?
#         @test result.model isa SoleXplorer.DecisionEnsemble
#     end

#     @testset "adaboost_classifier" begin
#         result = traintest(X, y; models=(type=:adaboost_classifier, params=(; rng=rng)))
#         @test result isa SoleXplorer.Modelset
#         @test result.classifier isa SoleXplorer.AdaBoostStumpClassifier
#         @test result.model isa SoleXplorer.DecisionEnsemble
#     end

#     @testset "modaldecisiontree" begin
#         result = traintest(X, y; models=(type=:modaldecisiontree, params=(; rng=rng)))
#         @test result isa SoleXplorer.Modelset
#         @test result.classifier isa SoleXplorer.ModalDecisionTree
#         @test result.model isa SoleXplorer.DecisionTree
#     end

#     @testset "modalrandomforest" begin
#         result = traintest(X, y; models=(type=:modalrandomforest, params=(; rng=rng)))
#         @test result isa SoleXplorer.Modelset
#         @test result.classifier isa SoleXplorer.ModalRandomForest
#         @test result.model isa SoleXplorer.DecisionEnsemble
#     end
# end

# # ---------------------------------------------------------------------------- #
# #                       tuning train/test classification                       #
# # ---------------------------------------------------------------------------- #
# @testset "tuning usage traintest function" begin
#     @testset "decisiontree_classifier" begin
#         result = traintest(X, y; models=(type=:decisiontree_classifier, params=(; rng=rng), tuning=true))
#         @test result isa SoleXplorer.Modelset
#         @test result.classifier isa MLJTuning.ProbabilisticTunedModel{LatinHypercube, SoleXplorer.DecisionTreeClassifier}
#         @test result.model isa SoleXplorer.DecisionTree
#     end

#     @testset "randomforest_classifier" begin
#         result = traintest(X, y; models=(type=:randomforest_classifier, params=(; rng=rng), tuning=true))
#         @test result isa SoleXplorer.Modelset
#         @test result.classifier isa MLJTuning.ProbabilisticTunedModel{LatinHypercube, SoleXplorer.RandomForestClassifier}
#         @test result.model isa SoleXplorer.DecisionEnsemble
#     end

#     @testset "adaboost_classifier" begin
#         result = traintest(X, y; models=(type=:adaboost_classifier, params=(; rng=rng), tuning=true))
#         @test result isa SoleXplorer.Modelset
#         @test result.classifier isa MLJTuning.ProbabilisticTunedModel{LatinHypercube, SoleXplorer.AdaBoostStumpClassifier}
#         @test result.model isa SoleXplorer.DecisionEnsemble
#     end

#     @testset "modaldecisiontree" begin
#         result = traintest(X, y; models=(type=:modaldecisiontree, params=(; rng=rng), tuning=true))
#         @test result isa SoleXplorer.Modelset
#         @test result.classifier isa MLJTuning.ProbabilisticTunedModel{LatinHypercube, SoleXplorer.ModalDecisionTree}
#         @test result.model isa SoleXplorer.DecisionTree
#     end

#     @testset "modalrandomforest" begin
#         result = traintest(X, y; models=(type=:modalrandomforest, params=(; rng=rng), tuning=true))
#         @test result isa SoleXplorer.Modelset
#         @test result.classifier isa MLJTuning.ProbabilisticTunedModel{LatinHypercube, SoleXplorer.ModalRandomForest}
#         @test result.model isa SoleXplorer.DecisionEnsemble
#     end
# end

# # ---------------------------------------------------------------------------- #
# #                    classification pratical usage examples                    #
# # ---------------------------------------------------------------------------- #
# @testset "classification pratical usage examples" begin
#     @testset "decisiontree_classifier" begin
#         result = traintest(X, y;
#             models=(
#                 # always declare the model you're going to use
#                 type=:decisiontree_classifier,
#                 # you can tweak every parameter of the model
#                 params=(; max_depth=5, min_samples_leaf=1),
#                 # optionally you can use different windowing strategies:
#                 # in this case, even if the model is propositional, and doesnt accept data vectors,
#                 # we mimic a modal behaviour splitting data vectors in 2 windows,
#                 # and then, apply choosen features on each window
#                 winparams=(; type=adaptivewindow, nwindows=2),
#                 # you can choose which features to use, mode_5 comes from Catch22 package
#                 features=[minimum, mean, cov, mode_5],
#                 # optionally you can turn on tuning default settings for every model,
#                 # using simply "tuning=true"
#                 tuning=true
#             )
#         )
#         @test result isa SoleXplorer.Modelset
#         @test result.classifier isa MLJTuning.ProbabilisticTunedModel{LatinHypercube, SoleXplorer.DecisionTreeClassifier}
#         @test result.model isa SoleXplorer.DecisionTree
#     end

#     @testset "randomforest_classifier" begin
#         result = traintest(X, y;
#             models=(
#                 type=:randomforest_classifier,
#                 # params is a NamedTuple: in case you have only one parameter,
#                 # remember to place a ';' at the beginning, or a ',' at the end
#                 params=(; n_trees=25),
#                 features=[minimum, mean, std],
#                 # you can use a tuning strategy coming from MLJ library
#                 tuning=(
#                     # you can choose the tuning method and adjust the parameters
#                     # specific for the choosen method
#                     method=(type=latinhypercube, rng=rng), 
#                     # you can also tweak global tuning parameters
#                     params=(repeats=10, n=5),
#                     # every model has default ranges for tuning
#                     # but it's highly recommended to choose which parameters to tune
#                     ranges=[
#                         SoleXplorer.range(:sampling_fraction, lower=0.3, upper=0.9),
#                         SoleXplorer.range(:feature_importance, values=[:impurity, :split])
#                     ]
#                 ),   
#             )
#         )
#         @test result isa SoleXplorer.Modelset
#         @test result.classifier isa MLJTuning.ProbabilisticTunedModel{LatinHypercube, SoleXplorer.RandomForestClassifier}
#         @test result.model isa SoleXplorer.DecisionEnsemble
#     end

#     @testset "modaldecisiontree" begin
#         result = traintest(X, y;
#             models=(
#                 type=:modaldecisiontree,
#                 winparams=(; type=adaptivewindow, nwindows=20),
#                 features=[minimum, mean, std]
#             )
#         )
#         @test result isa SoleXplorer.Modelset
#         @test result.classifier isa SoleXplorer.ModalDecisionTree
#         @test result.model isa SoleXplorer.DecisionTree
#     end

#     @testset "preprocess params" begin
#         result = traintest(X, y;
#             models=(
#                 type=:decisiontree_classifier,
#                 params=(; max_depth=5, min_samples_leaf=1),
#                 winparams=(; type=adaptivewindow, nwindows=2),
#                 features=[minimum, mean, cov, mode_5],
#                 tuning=true
#             ),
#             # you can also specify preprocessing parameters
#             # to fine tuning train test split
#             preprocess=(
#                 train_ratio = 0.7,
#                 stratified=true,
#                 nfolds=3,
#                 rng=rng
#             )
#         )
#         @test result isa SoleXplorer.Modelset
#         @test result.classifier isa MLJTuning.ProbabilisticTunedModel{LatinHypercube, SoleXplorer.DecisionTreeClassifier}
#         @test result.model isa Vector{<:SoleXplorer.DecisionTree}
#     end
# end

# # ---------------------------------------------------------------------------- #
# #                       classification multiple models                         #
# # ---------------------------------------------------------------------------- #
# @testset "classification multiple models" begin
#     results = traintest(X, y;
#         # you can stack multiple models in a vector
#         models=[(
#                 type=:decisiontree_classifier,
#                 params=(max_depth=3, min_samples_leaf=14),
#                 features=[minimum, mean, cov, mode_5]
#             ),
#             (
#                 type=:adaboost_classifier,
#                 winparams=(type=movingwindow, window_size=6),
#                 tuning=true
#             ),
#             (; type=:modaldecisiontree)],
#         # you can also specify global parameters for all models
#         # note that if you specify them also in model definitions,
#         # they will be overwritten.
#         # for example, this could be very useful if you want to pass rng parameter to all models
#         globals=(
#             params=(; rng=rng),
#             features=[std],
#             tuning=false
#         )
#     )
#     @test results isa Vector{Modelset}
    
#     @test results[1] isa SoleXplorer.Modelset
#     @test results[1].classifier isa SoleXplorer.DecisionTreeClassifier
#     @test results[1].model isa SoleXplorer.DecisionTree

#     @test results[2] isa SoleXplorer.Modelset
#     @test results[2].classifier isa MLJTuning.ProbabilisticTunedModel{LatinHypercube, SoleXplorer.AdaBoostStumpClassifier}
#     @test results[2].model isa SoleXplorer.DecisionEnsemble

#     @test results[3] isa SoleXplorer.Modelset
#     @test results[3].classifier isa SoleXplorer.ModalDecisionTree
#     @test results[3].model isa SoleXplorer.DecisionTree
# end

# # ---------------------------------------------------------------------------- #
# #                                  REGRESSION                                  #
# # ---------------------------------------------------------------------------- #
# table = RDatasets.dataset("datasets", "LifeCycleSavings")
# y = table[:, :DDPI]
# X = select(table, Not([:DDPI, :Country]));
# train_seed = 11
# rng = Random.Xoshiro(train_seed)
# Random.seed!(train_seed)

# # ---------------------------------------------------------------------------- #
# #                          basic train/test regression                         #
# # ---------------------------------------------------------------------------- #
# @testset "basic usage traintest function" begin
#     @testset "decisiontree_classifier" begin
#         result = traintest(X, y; models=(type=:decisiontree_regressor, params=(; rng=rng)))
#         @test result isa SoleXplorer.Modelset
#         @test result.classifier isa SoleXplorer.DecisionTreeRegressor
#         @test result.model isa SoleXplorer.DecisionTree
#     end

#     @testset "randomforest_classifier" begin
#         result = traintest(X, y; models=(type=:randomforest_regressor, params=(; rng=rng)))
#         @test result isa SoleXplorer.Modelset
#         @test result.classifier isa SoleXplorer.RandomForestRegressor
#         @test result.model isa SoleXplorer.DecisionEnsemble
#     end
# end

# # ---------------------------------------------------------------------------- #
# #                         tuning train/test regression                         #
# # ---------------------------------------------------------------------------- #
# @testset "tuning usage traintest function" begin
#     @testset "decisiontree_classifier" begin
#         result = traintest(X, y; models=(type=:decisiontree_regressor, tuning=true))
#         @test result isa SoleXplorer.Modelset
#         @test result.classifier isa MLJTuning.DeterministicTunedModel{LatinHypercube, SoleXplorer.DecisionTreeRegressor}
#         @test result.model isa SoleXplorer.DecisionTree
#     end

#     @testset "randomforest_classifier" begin
#         result = traintest(X, y; models=(type=:randomforest_regressor, tuning=true))
#         @test result isa SoleXplorer.Modelset
#         @test result.classifier isa MLJTuning.DeterministicTunedModel{LatinHypercube, SoleXplorer.RandomForestRegressor}
#         @test result.model isa SoleXplorer.DecisionEnsemble
#     end
# end
