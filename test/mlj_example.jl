using MLJ, SoleXplorer, MLJBase
using MLJDecisionTreeInterface
using MLJModelInterface
using DataFrames, Random
using Plots

using NearestNeighborModels         # For KNN models
using MLJMultivariateStatsInterface # For PCA models

Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

# ---------------------------------------------------------------------------- #
#                              solexplorer setup                               #
# ---------------------------------------------------------------------------- #
dsc = prepare_dataset(
    Xc, yc;
    model=(;type=:decisiontree),
    preprocess=(;train_ratio=0.9, rng=Xoshiro(1)),
    measures=(log_loss, accuracy),
)

@btime begin
    dsc = prepare_dataset(
        Xc, yc;
        model=(;type=:decisiontree),
        preprocess=(;rng=Xoshiro(1)),
    )
end

dsc = prepare_dataset(
    Xc, yc;
    model=(;type=:decisiontree),
    resample=(;type=CV),
    preprocess=(;rng=Xoshiro(1))
)

dsr = prepare_dataset(
    Xr, yr;
    model=(;type=:decisiontree),
    preprocess=(;rng=Xoshiro(1))
)

# ---------------------------------------------------------------------------- #
#                                search models                                 #
# ---------------------------------------------------------------------------- #
# Listing the models compatible with the present data
amc = models(matching(Xc, yc))
amr = models(matching(Xr, yr))

# A more refined search
models() do model
    matching(model, Xc, yc) &&
    model.prediction_type == :deterministic &&
    model.is_pure_julia
end

# Searching for an unsupervised model
uamc = models(matching(Xc))

# ---------------------------------------------------------------------------- #
#                               evaluate method                                #
# ---------------------------------------------------------------------------- #
# Decision Tree Classifier
Tree = @load DecisionTreeClassifier pkg=DecisionTree
tree = Tree()

e1t = evaluate(
    tree, Xc, yc;
    resampling=CV(shuffle=false),
    measures=[log_loss, accuracy],
    per_observation=true,
    verbosity=0
)

e1f = evaluate(
    tree, Xc, yc;
    resampling=CV(shuffle=false),
    measures=[log_loss, accuracy],
    per_observation=false,
    verbosity=0
)

mset1 = train_test(dsc)

# ---------------------------------------------------------------------------- #
#                           fit and predict method                             #
# ---------------------------------------------------------------------------- #
Tree = @load DecisionTreeClassifier pkg=DecisionTree
tree = Tree()

mach = machine(tree, Xc, yc)
fit!(mach, rows=collect(dsc.ds.Xtrain.indices)[1])
yhat = predict(mach, MLJ.table(dsc.ds.Xtest))
l1 = log_loss(yhat, dsc.ds.ytest)

# SoleXplorer like
mach = machine(tree, MLJ.table(dsc.ds.Xtrain), dsc.ds.ytrain)
fit!(mach)
yhat = predict(mach, MLJ.table(dsc.ds.Xtest))
l2 = log_loss(yhat, dsc.ds.ytest)

l1 == l2 # true

fitted_params(mach)
report(mach)

# Notice that yhat is a vector of Distribution objects, 
# because DecisionTreeClassifier makes probabilistic predictions.
# The methods of the Distributions.jl package can be applied to such distributions:
broadcast(pdf, yhat, "virginica") # predicted probabilities of virginica
broadcast(pdf, yhat, dsc.ds.ytest) # predicted probability of observed class

# predict
pry = mode.(yhat)
prx = predict_mode(mach, DataFrame(dsc.ds.Xtest, :auto))

pry == prx # true

L = levels(yc)
pdf(yhat, L)

# machine(model::Unsupervised, X)
# machine(model::Supervised, X, y)

# ---------------------------------------------------------------------------- #
#                           DataFrame vs MLJ.table                             #
# ---------------------------------------------------------------------------- #
Tree = @load DecisionTreeClassifier pkg=DecisionTree
tree = Tree()

@btime machine(tree, DataFrame(dsc.ds.Xtrain, :auto), dsc.ds.ytrain)
# 13.934 μs (93 allocations: 7.92 KiB)

@btime machine(tree, MLJ.table(dsc.ds.Xtrain), dsc.ds.ytrain)
# 11.312 μs (81 allocations: 7.52 KiB)

# ---------------------------------------------------------------------------- #
#                                  MLJ workflow                                #
# ---------------------------------------------------------------------------- #
# Instantiating a model
Tree = @load DecisionTreeClassifier pkg=DecisionTree
tree = Tree(min_samples_split=5, max_depth=4)

# ---------------------------------------------------------------------------- #
# Evaluating a model
KNN = @load KNNRegressor
knn = KNN()
evaluate(
    knn, Xr, yr;
    resampling=CV(nfolds=5),
    measure=[RootMeanSquaredError(), LPLoss(1)]
)

# let see what's inside the model
Tree = @load DecisionTreeClassifier pkg=DecisionTree
tree = Tree()
m = evaluate(
    tree, Xc, yc;
    measures=[log_loss, accuracy],
    verbosity=0
)

# julia> m.
# fitted_params_per_fold  measure                 measurement
# model                   operation               per_fold
# per_observation         repeats                 report_per_fold
# resampling              train_test_rows

# ---------------------------------------------------------------------------- #
# More performance evaluation example
@btime evaluate(
    tree, Xc, yc;
    measures=[log_loss, accuracy],
    verbosity=0
)
# 1.421 ms (10366 allocations: 626.98 KiB)

@btime evaluate(
    tree, Xc, yc;
    measure=[LogLoss(), Accuracy()],
    verbosity=0
)
# 1.402 ms (10366 allocations: 626.98 KiB)

@btime evaluate(
    tree, Xc, yc,
    resampling=Holdout(fraction_train=0.7, shuffle=true, rng=1234),
    measure=[LogLoss(), Accuracy()]
)
# 338.442 μs (2516 allocations: 171.18 KiB)

@btime evaluate(
    tree, Xc, yc,
    resampling=Holdout(fraction_train=0.7, shuffle=true, rng=1234),
    measure=[LogLoss(), Accuracy()],
    verbosity=0
)
# 323.373 μs (2498 allocations: 170.44 KiB)

# ---------------------------------------------------------------------------- #
# Basic fit/transform for unsupervised models
PCA = @load PCA
pca = PCA(maxoutdim=2)
mach = machine(pca, Xc)
fit!(mach, rows=collect(dsc.ds.Xtrain.indices)[1])

MLJ.transform(mach, rows=collect(dsc.ds.Xtest.indices)[1])

# ---------------------------------------------------------------------------- #
# Nested hyperparameter tuning
Tree = @load DecisionTreeClassifier pkg=DecisionTree
tree = Tree()
forest = EnsembleModel(model=tree, n=300)

# ProbabilisticEnsembleModel(
#   model = DecisionTreeClassifier(
#         max_depth = -1, 
#         min_samples_leaf = 1, 
#         min_samples_split = 2, 
#         min_purity_increase = 0.0, 
#         n_subfeatures = 0, 
#         post_prune = false, 
#         merge_purity_threshold = 1.0, 
#         display_depth = 5, 
#         feature_importance = :impurity, 
#         rng = Random.TaskLocalRNG()), 
#   atomic_weights = Float64[], 
#   bagging_fraction = 0.8, 
#   rng = Random.TaskLocalRNG(), 
#   n = 300, 
#   acceleration = CPU1{Nothing}(nothing), 
#   out_of_bag_measure = Any[])

r1 = MLJ.range(forest, :bagging_fraction, lower=0.5, upper=1.0, scale=:log10)
r2 = MLJ.range(forest, :(model.n_subfeatures), lower=1, upper=4) # nested

tuned_forest = TunedModel(
    model=forest,
    tuning=Grid(resolution=12),
    resampling=CV(nfolds=6),
    ranges=[r1, r2],
    measure=BrierLoss()
)

tuned_forest = TunedModel(
    model=forest,
    tuning=Grid(resolution=12),
    resampling=CV(nfolds=6),
    ranges=[r1, r2],
    measure=[LogLoss(), Accuracy()]
)

mach = machine(tuned_forest, Xc, yc)
fit!(mach)
F = fitted_params(mach)
F.best_model

r = report(mach)
keys(r)
r.history[[1,end]]

plot(mach)

yhat = predict(mach, MLJ.table(dsc.ds.Xtest))

# ---------------------------------------------------------------------------- #
# Constructing linear pipelines

KNN = @load KNNRegressor
knn_with_target = TransformedTargetModel(model=KNN(K=3), transformer=Standardizer())

pipe = (Xr -> coerce(Xr, :age=>Continuous)) |> OneHotEncoder() |> knn_with_target

mach = machine(pipe, Xr, yr) |> fit!
F = fitted_params(mach)
F.transformed_target_model_deterministic.model

Tree = @load DecisionTreeClassifier pkg=DecisionTree
tree = Tree()
forest = EnsembleModel(model=tree, bagging_fraction=0.8, n=300)
mach = machine(forest, Xc, yc)
evaluate!(mach, measure=Accuracy())

# ---------------------------------------------------------------------------- #
#                                   Machines                                   #
# ---------------------------------------------------------------------------- #
Tree = @load DecisionTreeClassifier pkg=DecisionTree
tree = Tree()

mach = machine(tree, Xc, yc)
fit!(mach, rows=collect(dsc.ds.Xtrain.indices)[1])
training_losses(mach)
feature_importances(mach)

@btime begin
    mach = machine(tree, MLJ.table(dsc.ds.Xtrain), dsc.ds.ytrain)
    fit!(mach)
end
# 128.869 μs (578 allocations: 41.36 KiB)

# Specify cache=false to prioritize memory management over speed
@btime begin
    mach = machine(tree, MLJ.table(dsc.ds.Xtrain), dsc.ds.ytrain; cache=false)
    fit!(mach)
end
# 114.104 μs (579 allocations: 41.39 KiB)

# ---------------------------------------------------------------------------- #
#                                Partitioning                                  #
# ---------------------------------------------------------------------------- #
(Xtrain, Xvalid, Xtest), (ytrain, yvalid, ytest) = partition((Xc, yc), 0.7, 0.2, rng=Xoshiro(1), multi=true) # for 70:20:10 ratio

(Xtrain, Xtest), (ytrain, ytest) = partition((Xc, yc), 0.8, rng=Xoshiro(1), multi=true) # for 70:20:10 ratio

Matrix(Xtrain) == dsc.ds.Xtrain # true
Matrix(Xtest) == dsc.ds.Xtest   # true
ytrain == dsc.ds.ytrain         # true
ytest == dsc.ds.ytest           # true

# ---------------------------------------------------------------------------- #
#                                Experimental                                  #
# ---------------------------------------------------------------------------- #
# isnothing vs == nothing vs === nothing
test = 1
@btime test == nothing
# 16.087 ns (0 allocations: 0 bytes)
@btime test === nothing
# 2.813 ns (0 allocations: 0 bytes)
@btime isnothing(test)
# 3.209 ns (0 allocations: 0 bytes)

test = nothing
@btime test == nothing
# 19.441 ns (0 allocations: 0 bytes)
@btime test === nothing
# 2.813 ns (0 allocations: 0 bytes)
@btime isnothing(test)
# 3.209 ns (0 allocations: 0 bytes)

test = Xc
@btime test == nothing
# 19.901 ns (0 allocations: 0 bytes)
@btime test === nothing
# 2.813 ns (0 allocations: 0 bytes)
@btime isnothing(test)
# 3.209 ns (0 allocations: 0 bytes)

@btime !(test === nothing)
# 2.814 ns (0 allocations: 0 bytes)
@btime !isnothing(test)
# 3.209 ns (0 allocations: 0 bytes)

@btime symbolic_analysis(
    Xc, yc;
    model=(;type=:decisiontree),
    preprocess=(;rng=Xoshiro(1))
)

# === nothing -> 318.183 μs (1974 allocations: 145.19 KiB)

@btime a=[t.test for t in dsc[2].tt]
# 599.452 ns (5 allocations: 288 bytes)
@btime a=collect(t.test for t in dsc[2].tt)
# 554.075 ns (5 allocations: 288 bytes)

# ---------------------------------------------------------------------------- #
# mlj evaluate vs solexplorer
Tree = @load DecisionTreeClassifier pkg=DecisionTree
tree = Tree()

@btime begin
    mk1 = machine(tree, MLJ.table(dsc.ds.Xtrain), dsc.ds.ytrain)
    exp1 = evaluate!(
        mach;
        resampling=CV(shuffle=false),
        measures=[log_loss, accuracy],
        verbosity=0
    )
end
# 1.347 ms (9771 allocations: 581.53 KiB)

@btime begin
    mk2 = machine(tree, MLJ.table(dsc.ds.Xtrain), dsc.ds.ytrain)
    fit!(mk2)
    exp2 = evaluate!(
        mk2;
        resampling=CV(shuffle=false),
        measures=[log_loss, accuracy],
        verbosity=0
    )
end
# 1.594 ms (10268 allocations: 615.39 KiB)

@btime begin
    Tree = @load DecisionTreeClassifier pkg=DecisionTree
    tree = Tree()

    e1 = evaluate(
        tree, Xc, yc;
        resampling=Holdout(fraction_train=0.8, shuffle=false, rng=Xoshiro(1)),
        verbosity=0
    )
end
# 467.807 μs (1664 allocations: 110.12 KiB)

@btime begin
    dsc = symbolic_analysis(
        Xc, yc;
        model=(;type=:decisiontree),
        preprocess=(;rng=Xoshiro(1))
    )
end
# con la struttura ds esterna a Modelset
# 402.266 μs (2211 allocations: 156.42 KiB)
# con la struttura ds in Modelset
# 407.785 μs (2211 allocations: 156.39 KiB)
# julia> typeof(dsc.model)
# SoleModels.DecisionTree{String}

@btime xt = dsc.ds.X[dsc.ds.tt[1].train, :]
# 1.305 μs (13 allocations: 4.65 KiB)

@btime xt = view(dsc.ds.X, dsc.ds.tt[1].train, :)
# 1.111 μs (19 allocations: 528 bytes)

@btime xt = view.(Ref(dsc.ds.X), getfield(dsc.ds.tt[1], :train), Ref(:))
# 1.360 μs (9 allocations: 5.62 KiB)

@btime xt = @views dsc.ds.X[dsc.ds.tt[1].train, :]
# 926.065 ns (19 allocations: 528 bytes)

@btime begin
    measures = [LogLoss, Accuracy]
    a = measures[1]
    b = measures[2]
end
# 15.265 ns (1 allocation: 48 bytes)

@btime begin
    measures = (LogLoss, Accuracy)
    a = measures[1]
    b = measures[2]
end
# 1.609 ns (0 allocations: 0 bytes)


# ---------------------------------------------------------------------------- #
# try to understand how resamplig is treated
Tree = @load DecisionTreeClassifier pkg=DecisionTree
tree = Tree()

# in evaluate vengono passati: il modello, il dataset e cache=true
mach = machine(tree, Xc, yc; cache=true)

e1 = evaluate!(
    mach;
    resampling=CV(shuffle=true),
    verbosity=0
)

# decostruzione di evaluate!
# in resampling.jl riga: 1161

# default in funzione evaluate!
# mach::Machine;
# resampling=CV(),
resampling=CV(shuffle=true)
# measures=nothing
measures=[log_loss, accuracy]
measure=measures
weights=nothing
class_weights=nothing
operations=nothing
operation=operations
acceleration=default_resource()
rows=nothing
repeats=1
force=false
check_measure=true
per_observation_flag=true
# verbosity=1,
verbosity=0
logger=default_logger()
compact=false

# evaluate!
_measures = MLJBase._actual_measures(measure, mach.model)
# 2-element Vector{StatisticalMeasuresBase.RobustMeasure}:
#  LogLoss(tol = 2.22045e-16)
#  Accuracy()

_operations = MLJBase._actual_operations(operation, _measures, mach.model, verbosity)
# 2-element Vector{Function}:
#  predict (generic function with 43 methods)
#  predict_mode (generic function with 11 methods)

# si passa all'evaluate! a riga 1590
train_args = Tuple(a() for a in mach.args)
y = train_args[2]
# julia> mach.args[1]()
# Tables.MatrixTable{SubArray{Float64, 2, Matrix{Float64}, Tuple{Vector{Int64}, Base.Slice{Base.OneTo{Int64}}}, false}} with 120 rows, 4 columns, and schema:
#  :x1  Float64
#  :x2  Float64
#  :x3  Float64
#  :x4  Float64

# julia> mach.args[2]()
# 120-element view(::CategoricalArrays.CategoricalVector{String, UInt32, String, CategoricalArrays.CategoricalValue{String, UInt32}, Union{}}, [133, 3, 43, 51, 147, 40, 112, 129, 132, 124  …  22, 95, 120, 11, 88, 128, 8, 12, 99, 79]) with eltype CategoricalArrays.CategoricalValue{String, UInt32}:
#  "virginica"
#  "setosa"
#  "setosa"
#  ⋮
#  "setosa"
#  "versicolor"

_rows = MLJBase.actual_rows(rows, nrows(y), verbosity)

_resampling =
    vcat(
        [MLJBase.train_test_pairs(resampling, _rows, train_args...) for i in 1:repeats]...
    )

# 6-element Vector{Tuple{Vector{Int64}, Vector{Int64}}}:
#  ([94, 102, 73, 10, 32, 21, 26, 76, 8, 110  …  84, 47, 78, 75, 59, 103, 56, 49, 81, 97], [107, 30, 106, 50, 3, 92, 96, 64, 98, 72, 52, 66, 6, 5, 99, 34, 36, 115, 69, 111])
#  ([107, 30, 106, 50, 3, 92, 96, 64, 98, 72  …  84, 47, 78, 75, 59, 103, 56, 49, 81, 97], [94, 102, 73, 10, 32, 21, 26, 76, 8, 110, 101, 71, 109, 95, 62, 7, 93, 118, 58, 20])
#  ([107, 30, 106, 50, 3, 92, 96, 64, 98, 72  …  84, 47, 78, 75, 59, 103, 56, 49, 81, 97], [17, 83, 15, 42, 70, 44, 68, 119, 46, 116, 85, 11, 86, 4, 67, 38, 51, 79, 1, 113])
#  ([107, 30, 106, 50, 3, 92, 96, 64, 98, 72  …  84, 47, 78, 75, 59, 103, 56, 49, 81, 97], [60, 39, 88, 28, 29, 24, 117, 77, 9, 41, 45, 65, 12, 40, 91, 48, 80, 104, 22, 37])
#  ([107, 30, 106, 50, 3, 92, 96, 64, 98, 72  …  84, 47, 78, 75, 59, 103, 56, 49, 81, 97], [35, 61, 14, 53, 2, 90, 43, 89, 112, 25, 108, 16, 13, 54, 100, 23, 33, 82, 57, 105])
#  ([107, 30, 106, 50, 3, 92, 96, 64, 98, 72  …  108, 16, 13, 54, 100, 23, 33, 82, 57, 105], [114, 120, 74, 31, 27, 63, 87, 55, 18, 19, 84, 47, 78, 75, 59, 103, 56, 49, 81, 97])

# si passa all'evaluate! a riga 1413
X = mach.args[1]()
y = mach.args[2]()
_nrows = MLJBase.nrows(y)


nfolds = MLJ.length(_resampling)
test_fold_sizes = map(_resampling) do train_test_pair
    test = last(train_test_pair)
    test isa Colon && (return _nrows)
    length(test)
end

nmeasures = length(measures) # [log_loss, accuracy]

function fit_and_extract_on_fold(mach, k)
    train, test = _resampling[k]
    fit!(mach; rows=train, verbosity=verbosity - 1, force=force)
    # build a dictionary of predictions keyed on the operations
    # that appear (`predict`, `predict_mode`, etc):
    yhat_given_operation =
        Dict(op=>op(mach, rows=test) for op in unique(_operations))

    ytest = selectrows(y, test)
    if per_observation_flag
        measurements =  map(measures, _operations) do m, op
            MLJBase.StatisticalMeasuresBase.measurements(
                m,
                yhat_given_operation[op],
                ytest,
                MLJBase._view(weights, test),
                class_weights,
            )
        end
    else
        measurements =  map(measures, _operations) do m, op
            m(
                yhat_given_operation[op],
                ytest,
                _view(weights, test),
                class_weights,
            )
        end
    end

    fp = fitted_params(mach)
    r = report(mach)
    return (measurements, fp, r)
end

function _evaluate!(func, mach, nfolds)
    ret = mapreduce(vcat, 1:nfolds) do k
        r = func(mach, k)
        return [r, ]
    end

    return zip(ret...) |> collect

end

measurements_vector_of_vectors, fitted_params_per_fold, report_per_fold  =
    _evaluate!(
        fit_and_extract_on_fold,
        mach,
        nfolds
    )

# prova con k=1
k=1

train, test = _resampling[k]
fit!(mach; rows=train, verbosity=verbosity - 1, force=force)
yhat_given_operation = Dict(op=>op(mach, rows=test) for op in unique(_operations))

ytest = selectrows(y, test)

# per_observation_flag = true
@btime begin
    measurements =  map((LogLoss(), Accuracy()), _operations) do m, op
        MLJBase.StatisticalMeasuresBase.measurements(
            m,
            yhat_given_operation[op],
            ytest,
            MLJBase._view(weights, test),
            class_weights,
        )
    end
end
# 20.539 μs (193 allocations: 9.61 KiB)

# per_observation_flag = false
@btime begin
    measurements =  map((LogLoss(), Accuracy()), _operations) do m, op
        m(
            yhat_given_operation[op],
            ytest,
            MLJBase._view(weights, test),
            class_weights,
        )
    end
end
# 20.567 μs (187 allocations: 8.75 KiB)

@btime begin
    measurements =  map((log_loss, accuracy), _operations) do m, op
        m(
            yhat_given_operation[op],
            ytest,
            MLJBase._view(weights, test),
            class_weights,
        )
    end
end
# 20.678 μs (187 allocations: 8.75 KiB)

measurements =  map((LogLoss(), Accuracy()), _operations) do m, op
    MLJBase.StatisticalMeasuresBase.measurements(
        m,
        yhat_given_operation[op],
        ytest,
        MLJBase._view(weights, test),
        class_weights,
    )
end

fp = fitted_params(mach)
r = report(mach)

####

ret = mapreduce(vcat, 1:nfolds) do k
    r = fit_and_extract_on_fold(mach, k)
    return [r, ]
end

zreat = zip(ret...) |> collect

# ---------------------------------------------------------------------------- #
#             SoleXplorer eval_measures inspired by MLJ evaluate!              #
# ---------------------------------------------------------------------------- #
# Setup
using MLJ, SoleXplorer, MLJBase
using MLJDecisionTreeInterface
using MLJModelInterface
using DataFrames, Random
using Plots

using NearestNeighborModels         # For KNN models
using MLJMultivariateStatsInterface # For PCA models

Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

dsc = train_test(
    Xc, yc;
    model=(;type=:decisiontree),
    resample=(;type=CV),
    measures=(log_loss, accuracy),
    preprocess=(;rng=Xoshiro(1))
)

dsr = train_test(
    Xr, yr;
    model=(;type=:decisiontree),
    resample=(;type=CV),
    measures=(log_loss, accuracy),
    preprocess=(;rng=Xoshiro(1))
)

model = dsc
weights = nothing
class_weights = nothing
per_observation = true

# bench
Tree = @load DecisionTreeClassifier pkg=DecisionTree
tree = Tree()
e1t = evaluate(
    tree, Xc, yc;
    resampling=CV(nfolds=6, shuffle=false, rng=Xoshiro(1)),
    measures=[log_loss, accuracy, confusion_matrix, kappa],
    per_observation=true,
    verbosity=0
)

function eval_measures!(model::Modelset)::Measures
    # mach::Machine,
    # resampling,
    # weights,
    # class_weights,
    # rows,
    # verbosity,
    # repeats,
    # measures,
    # operations,
    # acceleration,
    # force,
    # per_observation_flag,
    # logger,
    # user_resampling,
    # compact,
    # )

    # @btime begin
    #     _measures = MLJBase._actual_measures([model.setup.measures...], model.mach.model)
    #     _operations = MLJBase._actual_operations(nothing, _measures, model.mach.model, 0)
    # end
    # 4.842 μs (42 allocations: 1.33 KiB)

    # @btime begin
    #     _measures = MLJBase._actual_measures([SoleXplorer.get_setup_meas(model)...], SoleXplorer.get_mach_model(model))
    #     _operations = MLJBase._actual_operations(nothing, _measures, SoleXplorer.get_mach_model(model), 0)
    # end
    # 4.714 μs (42 allocations: 1.33 KiB)

    # cava i vari SoleXplorer.
    _measures = MLJBase._actual_measures([SoleXplorer.get_setup_meas(model)...], SoleXplorer.get_mach_model(model))
    _operations = MLJBase._actual_operations(nothing, _measures, SoleXplorer.get_mach_model(model), 0)

    y = SoleXplorer.get_mach_y(model)
    tt = SoleXplorer.get_setup_tt(model)
    nfolds = length(tt)
    test_fold_sizes = [length(tt[k][1]) for k in 1:nfolds]

    nmeasures = length(SoleXplorer.get_setup_meas(model))

    # weights used to aggregate per-fold measurements, which depends on a measures
    # external mode of aggregation:
    fold_weights(mode) = nfolds .* test_fold_sizes ./ sum(test_fold_sizes)
    fold_weights(::MLJBase.StatisticalMeasuresBase.Sum) = nothing

    # @btime begin
        measurements_vector = mapreduce(vcat, 1:nfolds) do k
            yhat_given_operation = Dict(op=>op(SoleXplorer.get_mach(model), rows=tt[k][1]) for op in unique(_operations))
            test = tt[k][1]

            # [per_observation ? begin
            #     map(_measures, _operations) do m, op
            #         MLJBase.StatisticalMeasuresBase.measurements(
            #             m,
            #             yhat_given_operation[op],
            #             y[test],
            #             MLJBase._view(weights, test),
            #             class_weights,
            #         )
            #     end
            # end : begin
            #     map(_measures, _operations) do m, op
            #         m(
            #             yhat_given_operation[op],
            #             y[test],
            #             MLJBase._view(weights, test),
            #             class_weights,
            #         )
            #     end
            # end]

            [map(_measures, _operations) do m, op
                m(
                    yhat_given_operation[op],
                    y[test],
                    MLJBase._view(weights, test),
                    class_weights,
                )
            end]
        end

        # @btime begin
        #     measurements_flat = vcat(measurements_vector...)
        #     measurements_matrix = permutedims(
        #         reshape(collect(measurements_flat), (nmeasures, nfolds))
        #     )
        # end
        # # 331.274 ns (8 allocations: 576 bytes)

        # @btime measurements_matrix = permutedims(reshape(vcat(measurements_vector...), (nmeasures, nfolds)))
        # # 291.755 ns (6 allocations: 416 bytes)

        # @btime measurements_matrix = permutedims(reduce(hcat, measurements_vector))
        # # 149.251 ns (4 allocations: 352 bytes)

        # measurements_flat = vcat(measurements_vector...)
        # m_original = permutedims(
        #     reshape(collect(measurements_flat), (nmeasures, nfolds))
        # )

        # m_proposed = permutedims(reduce(hcat, measurements_vector))

        # @assert m_original == m_proposed # true

        measurements_matrix = permutedims(reduce(hcat, measurements_vector))
    # end
    # 605.570 μs (7858 allocations: 434.73 KiB)

    #####

    # @btime begin
    #     measurements_matrix2 = permutedims(mapreduce(hcat, 1:nfolds) do k
    #         yhat_given_operation = Dict(op=>op(SoleXplorer.get_mach(model), rows=tt[k][1]) for op in unique(_operations))
    #         test = tt[k][1]
            
    #         map(_measures, _operations) do m, op
    #             m(yhat_given_operation[op], y[test], MLJBase._view(weights, test), class_weights)
    #         end
    #     end)
    # end
    # 605.570 μs (7858 allocations: 434.73 KiB)

    # @btime begin
    #     measurements_matrix3 = permutedims(hcat([
    #         let 
    #             yhat_given_operation = Dict(op=>op(SoleXplorer.get_mach(model), rows=tt[k][1]) for op in unique(_operations))
    #             test = tt[k][1]
    #             map(_measures, _operations) do m, op
    #                 m(yhat_given_operation[op], y[test], MLJBase._view(weights, test), class_weights)
    #             end
    #         end
    #         for k in 1:nfolds
    #     ]...))
    # end
    # 611.978 μs (7848 allocations: 434.48 KiB)

    # @btime begin
    # measurements_matrix4 = Matrix{Float64}(undef, nfolds, nmeasures)
    #     for k in 1:nfolds
    #         yhat_given_operation = Dict(op=>op(SoleXplorer.get_mach(model), rows=tt[k][1]) for op in unique(_operations))
    #         test = tt[k][1]
            
    #         measurements_matrix4[k, :] = map(_measures, _operations) do m, op
    #             m(yhat_given_operation[op], y[test], MLJBase._view(weights, test), class_weights)
    #         end
    #     end
    # end
    # 623.638 μs (7869 allocations: 434.64 KiB)

    # @btime begin
    #     measurements_matrix = mapreduce(hcat, 1:nfolds) do k
    #         yhat_given_operation = Dict(op=>op(SoleXplorer.get_mach(model), rows=tt[k][1]) for op in unique(_operations))
    #         test = tt[k][1]
            
    #         map(_measures, _operations) do m, op
    #             m(yhat_given_operation[op], y[test], MLJBase._view(weights, test), class_weights)
    #         end
    #     end |> permutedims
    # end
    # 610.553 μs (7844 allocations: 434.44 KiB)

    # measurements for each observation:
    # _observation = if per_observation
    #    map(1:nmeasures) do k
    #        measurements_matrix[:,k]
    #    end
    # else
    #     fill(missing, nmeasures)
    # end

    # measurements for each fold:
    # _fold = if per_observation
    #     map(1:nmeasures) do k
    #         m = SoleXplorer.get_setup_meas(model)[k]
    #         mode = MLJBase.StatisticalMeasuresBase.external_aggregation_mode(m)
    #         map(_observation[k]) do v
    #             MLJBase.StatisticalMeasuresBase.aggregate(v; mode)
    #         end
    #     end
    # else
    #     map(1:nmeasures) do k
    #         measurements_matrix[:,k]
    #     end
    # end

    # measurements for each fold:
    _fold = map(1:nmeasures) do k
        measurements_matrix[:,k]
    end

    # overall aggregates:
    _measures_values = map(1:nmeasures) do k
        m = SoleXplorer.get_setup_meas(model)[k]
        mode = MLJBase.StatisticalMeasuresBase.external_aggregation_mode(m)
        MLJBase.StatisticalMeasuresBase.aggregate(
            _fold[k];
            mode,
            weights=fold_weights(mode),
        )
    end

    Measures(
        _fold,
        _measures,
        _measures_values,
        _operations,
    )
end

# ---------------------------------------------------------------------------- #
#                                   Weigths                                    #
# ---------------------------------------------------------------------------- #
using MLJ, SoleXplorer, MLJBase
using MLJDecisionTreeInterface
using MLJModelInterface
using DataFrames, Random
using Plots

using NearestNeighborModels         # For KNN models
using MLJMultivariateStatsInterface # For PCA models
using MultivariateStats

Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

amc = models(matching(Xc, yc))

Tree = @load EvoTreeClassifier pkg=EvoTrees
tree = Tree()

e1t = MLJ.evaluate(
    tree, Xc, yc;
    resampling=CV(shuffle=false),
    measures=[accuracy],
    per_observation=false,
    verbosity=0
)

### all nothing

# ---------------------------------------------------------------------------- #
#                        Measures results comparision                          #
# ---------------------------------------------------------------------------- #
# to be included in SoleXplorer tests
using Test
using BenchmarkTools
using MLJ, SoleXplorer
using MLJDecisionTreeInterface
using MLJModelInterface
using DataFrames, Random

Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

dsc = symbolic_analysis(
    Xc, yc;
    model=(;type=:decisiontree),
    resample = (type=Holdout, params=(shuffle=true, rng=Xoshiro(1))),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    measures=(log_loss, accuracy, kappa, confusion_matrix),
)

Tree = @load DecisionTreeClassifier pkg=DecisionTree
tree = Tree()
Random.seed!(1)
e1t = evaluate(
    tree, Xc, yc;
    resampling=Holdout(rng=Xoshiro(1)),
    measures=[log_loss, accuracy, kappa, confusion_matrix],
    per_observation=false,
    verbosity=0,
)

@test dsc.measures.measures[1] == e1t.measure[1]
@test dsc.measures.measures[2] == e1t.measure[2]
@test dsc.measures.measures[3] == e1t.measure[3]
@test dsc.measures.measures[4] == e1t.measure[4]

@test dsc.measures.measures_values[1] == e1t.measurement[1]
@test dsc.measures.measures_values[2] == e1t.measurement[2]
@test dsc.measures.measures_values[3] == e1t.measurement[3]
@test dsc.measures.measures_values[4] == e1t.measurement[4]

@btime begin
    dsc = symbolic_analysis(
        Xc, yc;
        model=(;type=:decisiontree),
        resample = (type=Holdout, params=(shuffle=true, rng=Xoshiro(1))),
        preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
        measures=(log_loss, accuracy, kappa, confusion_matrix),
    )
end
# 753.927 μs (4772 allocations: 334.00 KiB)
# but if we take out of the equation the process of converting the decisiontree in a sole model
# 395.653 μs (2773 allocations: 165.50 KiB)

@btime begin
    Tree = @load DecisionTreeClassifier pkg=DecisionTree
    tree = Tree()
    Random.seed!(1)
    e1t = evaluate(
        tree, Xc, yc;
        resampling=Holdout(rng=Xoshiro(1)),
        measures=[log_loss, accuracy, kappa, confusion_matrix],
        verbosity=0,
    )
end
# 644.247 μs (3234 allocations: 199.06 KiB)