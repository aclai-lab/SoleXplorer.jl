# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
"""
    AbstractModelSet

Abstract type for containers that hold symbolic model analysis results.

# Concrete Implementations
- [`ModelSet`](@ref): The primary implementation containing complete
  analysis results

See also: [`solexplorer`](@ref)
"""
abstract type AbstractModelSet end

# ---------------------------------------------------------------------------- #
#                                  modelset                                    #
# ---------------------------------------------------------------------------- #
"""
    ModelSet{S} <: AbstractModelSet

Wrapper for complete symbolic model analysis results.

This structure holds all components of a symbolic analysis workflow
including the dataset configuration, sole trained models, extracted
rules, and performance measures.

# Type Parameters
- `S`: The sole model type (e.g., `DecisionTreeClassifier`)

# Fields
- `ds::DataSet`: Dataset configuration with cross-validation setup,
  plus all settings needed by modal analysis.
- `sole::Vector{AbstractModel}`: Vector of trained symbolic models
  (one per CV fold).
- `measures::Union{Nothing,Measures}`: Performance evaluation measures,
  or `nothing` if evaluation has not been performed.

# Accessing Components
- [`get_ds`](@ref): Extract dataset configuration
- [`get_sole`](@ref): Extract trained models
- [`get_measures`](@ref): Extract performance measures
- [`get_values`](@ref): Extract computed measure values

# See also: [`solexplorer`](@ref)
"""
mutable struct ModelSet{S} <: AbstractModelSet
    ds::DataSet
    sole::Vector{AbstractModel}
    measures::Union{Nothing,Measures}

    function ModelSet(
        ds::DataSet,
        sole::SoleModel{S};
        measures::Union{Nothing,Measures}=nothing
    ) where S
        new{S}(ds, solemodels(sole), measures)
    end
end

# ---------------------------------------------------------------------------- #
#                                 constructors                                 #
# ---------------------------------------------------------------------------- #
"""
    get_ds(m::ModelSet) -> DataSet

Returns the dataset configuration from a `ModelSet`.
"""
get_ds(m::ModelSet) = m.ds

"""
    get_sole(m::ModelSet) -> Vector{AbstractModel}

Returns the vector of trained sole symbolic models from a `ModelSet`.
"""
get_sole(m::ModelSet) = m.sole

"""
    get_measures(m::ModelSet) -> Union{Nothing,Measures}

Returns the performance evaluation measures from a `ModelSet`.
"""
get_measures(m::ModelSet) = m.measures

"""
    get_values(m::ModelSet) -> Vector

Returns the computed performance measure values from a `ModelSet`.
"""
get_values(m::ModelSet) = get_measures(m).measures_values

# ---------------------------------------------------------------------------- #
#                                  base show                                   #
# ---------------------------------------------------------------------------- #
function Base.show(io::IO, m::ModelSet{S}) where S
    print(io, "ModelSet{$S}(")
    print(io, "models=$(length(solemodels(m)))")

    isnothing(measures(m))     || print(io, ", measures=$(length(measures(m)))")

    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", m::ModelSet{S}) where S
    println(io, "ModelSet{$S}:")
    println(io, "  Dataset: $(typeof(get_ds(m)))")
    println(io, "  Models:  $(length(get_sole(m))) symbolic models")

    isnothing(get_measures(m)) ?
        println(io, "  Measures: none") : begin
            println(io, "  Measures:")
            measures = get_measures(m)
            for (measure, value) in
                zip(measures.measures, measures.measures_values)
                println(io, "    $(measure) = $(value)")
            end
        end
end

function show_measures(m::ModelSet)
    println("Performance Measures:")
    for (ms, v) in zip(get_measures(m), get_values(m))
        v isa Real ?
            println("  $(ms) = $(round(v, digits=2))") :
            println("  $(ms) = $(v)")
    end
end

# ---------------------------------------------------------------------------- #
#                                 utilities                                    #
# ---------------------------------------------------------------------------- #
function supporting_predictions(solem::AbstractModel)
    return solem.info isa Base.RefValue ?
        solem.info[].supporting_predictions :
        solem.info.supporting_predictions
end

sole_predict_mode(solem::AbstractModel, y_test::AbstractVector{<:Label}) =
    supporting_predictions(solem)

# if it's a classification task, convert, if needed,
# predictions in categorical values.
# if not, wnon't do anything.
function sole_predict(solem::AbstractModel, y_test::AbstractVector{<:Label})
    preds = supporting_predictions(solem)
    return eltype(preds) <: CLabel ?
        begin
            classes_seen = unique(y_test)
            eltype(preds) <: CategoricalArrays.CategoricalValue ||
                (preds = categorical(preds, levels=levels(classes_seen)))
            [UnivariateFinite([p], [1.0]) for p in preds]
        end :
        preds
end

# set the random number generator for a rule extraction strategy
function set_rng(r::RuleExtractor, rng::Random.AbstractRNG)::RuleExtractor
    T = typeof(r)

    fnames = fieldnames(T)
    fvalues = map(fnames) do fn
        fn === :rng ? rng : getfield(r, fn)
    end
    
    return T(; NamedTuple{fnames}(fvalues)...)
end

# ---------------------------------------------------------------------------- #
#                                eval measures                                 #
# ---------------------------------------------------------------------------- #
# Adapted from MLJ's evaluate
function eval_measures(
    ds::DataSet,
    solem::Vector{AbstractModel},
    measures::Tuple{Vararg{FussyMeasure}},
    y_test::Vector{<:AbstractVector{<:Label}}
)
    mach_model = get_mach_model(ds)
    measures = MLJBase._actual_measures([measures...], mach_model)
    operations = get_operations(measures, MLJBase.prediction_type(mach_model))

    nfolds = length(ds)
    test_fold_sizes = [length(y_test[k]) for k in 1:nfolds]
    nmeasures = length(measures)

    # weights used to aggregate per-fold measurements,
    # which depends on a measures
    # external mode of aggregation:
    fold_weights(mode) = nfolds .* test_fold_sizes ./ sum(test_fold_sizes)
    fold_weights(::MLJBase.StatisticalMeasuresBase.Sum) = nothing
    
    measurements_vector = mapreduce(vcat, 1:nfolds) do k
        yhat_given_operation =
            Dict(op=>op(solem[k], y_test[k]) for op in unique(operations))

        # Forced to convert to string as some statistical measures don't accept
        # categorical arrays, like confusion matrix and kappa
        test = eltype(y_test[k]) <: CLabel ? String.(y_test[k]) : y_test[k]

        [map(measures, operations) do m, op
            m(
                yhat_given_operation[op],
                test,
                # MLJBase._view(weights, test),
                # class_weights
                MLJBase._view(nothing, test),
                nothing # TODO introduce class_weights
            )
        end]
    end

    measurements_matrix = permutedims(reduce(hcat, measurements_vector))

    # measurements for each fold:
    fold = map(1:nmeasures) do k
        measurements_matrix[:,k]
    end

    # overall aggregates:
    measures_values = map(1:nmeasures) do k
        m = measures[k]
        mode = MLJBase.StatisticalMeasuresBase.external_aggregation_mode(m)
        MLJBase.StatisticalMeasuresBase.aggregate(
            fold[k];
            mode,
            weights=fold_weights(mode)
        )
    end

    Measures(fold, measures, measures_values, operations)
end

# ---------------------------------------------------------------------------- #
#                            internal solexplorer                              #
# ---------------------------------------------------------------------------- #
function _solexplorer!(
    modelset::AbstractModelSet;
    measures::Tuple{Vararg{FussyMeasure}}=()
)
    ds = get_ds(modelset)
    solem = get_sole(modelset)

    y_test = get_y(ds, :test)
    isempty(measures) && (measures = _DefaultMeasures(first(y_test)))

    modelset.measures = eval_measures(ds, solem, measures, y_test)

    return modelset
end

function _solexplorer(
    ds::DataSet,
    solem::SoleModel;
    kwargs...
)
    modelset = ModelSet(ds, solem)
    _solexplorer!(modelset; kwargs...)
    return modelset
end

# ---------------------------------------------------------------------------- #
#                                 solexplorer                                  #
# ---------------------------------------------------------------------------- #
"""
    solexplorer!(modelset::ModelSet; kwargs...) -> ModelSet

Perform additional analysis on an existing `ModelSet` in-place.

Adds or updates performance measures on an existing `ModelSet`.

# Keyword Arguments
- `measures::Tuple{Vararg{FussyMeasure}}=()`: Performance measures to
  compute. If empty, default measures for the task type are used.

# See also: [`solexplorer`](@ref), [`ModelSet`](@ref)
"""
solexplorer!(modelset::ModelSet; kwargs...) = _solexplorer!(modelset; kwargs...)

"""
    solexplorer(
        X::AbstractDataFrame,
        y::AbstractVector{<:Label},
        args...;
        measures::Tuple{Vararg{FussyMeasure}}=(),
        kwargs...
    ) -> ModelSet

    solexplorer(
        dt::DT.DataTreatment,
        args...;
        measures::Tuple{Vararg{FussyMeasure}}=(),
        kwargs...
    ) -> ModelSet

    solexplorer(ds::DataSet, solem::SoleModel; kwargs...) -> ModelSet

Complete end-to-end symbolic model analysis workflow.

This is the main entry point for symbolic analysis. It performs the
complete workflow:
1. **Dataset Setup**: Configures cross-validation and preprocessing.
2. **Model Configuration**: Sets up the MLJ machine.
3. **Model Training**: Trains symbolic models on each CV fold.
4. **Evaluation**: Computes comprehensive performance metrics.

# Arguments
- `X::AbstractDataFrame`: Feature matrix with observations as rows.
- `y::AbstractVector{<:Label}`: Target variable (class labels or
  continuous values).
- `dt::DT.DataTreatment`: Pre-built data treatment object, e.g. from
  `DataTreatments.load_dataset`. Use this for modal (time-series) data.
- `ds::DataSet`: A pre-configured dataset. Combined with `solem` to
  skip training and go straight to evaluation.
- `solem::SoleModel`: A pre-trained sole model paired with `ds`.
- `args...`: Optional positional arguments forwarded to
  [`setup_dataset`](@ref) (e.g., a `TreatmentGroup`).

# Keyword Arguments
- `measures::Tuple{Vararg{FussyMeasure}}=()`: Performance measures tuple.
  If empty, default measures for the task type are used.
- `kwargs...`: Additional options forwarded to [`setup_dataset`](@ref)
  (e.g., `model`, `resampling`, `rng`, `tuning`).

# Examples
```julia
# Basic usage with default settings
modelset = solexplorer(X, y)

# Classification with cross-validation and tuning
range = SoleXplorer.range(
    :min_purity_increase; lower=0.001, upper=1.0, scale=:log
)
modelset = solexplorer(
    X, y;
    model=DecisionTreeClassifier(),
    resampling=CV(nfolds=5, shuffle=true),
    rng=1,
    tuning=GridTuning(
        resolution=10,
        resampling=CV(nfolds=3),
        range=range,
        measure=accuracy,
        repeats=2
    ),
    measures=(accuracy, log_loss, confusion_matrix, kappa)
)

# Time series classification with modal decision tree
modelset = solexplorer(
    X, y;
    model=ModalRandomForest(),
    resampling=Holdout(fraction_train=0.7, shuffle=true),
    rng=1,
    measures=(log_loss, accuracy, confusion_matrix, kappa)
)

# From a pre-built DataTreatment
modelset = solexplorer(dt; model=ModalDecisionTree(), rng=1)

# Accessing results
ds     = get_ds(modelset)
models = get_sole(modelset)
perf   = get_measures(modelset)
vals   = get_values(modelset)
```

# See also: [`ModelSet`](@ref), [`setup_dataset`](@ref), [`solexplorer!`](@ref)
"""
function solexplorer(
    X::AbstractDataFrame,
    y::AbstractVector{<:Label},
    args...;
    # w::Union{Nothing,Vector}=nothing,
    measures::Tuple{Vararg{FussyMeasure}}=(),
    kwargs...
)
    ds = setup_dataset(X, y, args...; kwargs...)
    solem = _train_test(ds)
    _solexplorer(ds, solem; measures)
end

function solexplorer(
    dt::DT.DataTreatment,
    args...;
    measures::Tuple{Vararg{FussyMeasure}}=(),
    kwargs...
)
    ds = setup_dataset(dt, args...; kwargs...)
    solem = _train_test(ds)
    _solexplorer(ds, solem; measures)
end

function solexplorer(
    ds::DataSet,
    solem::SoleModel;
    kwargs...
)
    _solexplorer(ds, solem; kwargs...)
end

solexplorer(X::AbstractArray, vnames::AbstractVector, args...; kwargs...) =
    solexplorer(DataFrame(X, vnames), args...; kwargs...)

