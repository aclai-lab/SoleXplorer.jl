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

Container returned by [`solexplorer`](@ref), holding the results of a symbolic
model-analysis run.

# Type Parameters
- `S`: Concrete model type associated with the input `SoleModel`.

# Fields
- `ds::DataSet`: Dataset, machine, and partition configuration. The
  partitioning strategy may be holdout or cross-validation.
- `sole::Vector{AbstractModel}`: One trained symbolic model per partition.
- `measures::Union{Nothing,Measures}`: Evaluation results, or `nothing` when
  evaluation has not yet been performed.

# See also
[`solexplorer`](@ref), [`solexplorer!`](@ref), [`DataSet`](@ref)
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
    get_ds(m::ModelSet)

Returns the dataset configuration from a `ModelSet`.
"""
get_ds(m::ModelSet) = m.ds

"""
    get_sole(m::ModelSet)

Returns the vector of trained sole symbolic models from a `ModelSet`.
"""
get_sole(m::ModelSet) = m.sole

"""
    get_measures(m::ModelSet)

Returns the performance evaluation measures from a `ModelSet`.
"""
get_measures(m::ModelSet) = m.measures

"""
    get_values(m::ModelSet)

Return the aggregate values of the performance measures stored in `m`.

This accessor requires `get_measures(m)` not to be `nothing`.
"""
get_values(m::ModelSet) = get_measures(m).measures_values

"""
    get_X(m::ModelSet, partition::Symbol)

Return the feature data vector.
"""
get_X(m::ModelSet, partition::Symbol) = get_X(m.ds, partition)

"""
    get_y(m::ModelSet, partition::Symbol)

Return the target vectors.
"""
get_y(m::ModelSet, partition::Symbol) = get_y(m.ds, partition)

# ---------------------------------------------------------------------------- #
#                                  base show                                   #
# ---------------------------------------------------------------------------- #
function Base.show(io::IO, m::ModelSet{S}) where S
    print(io, "ModelSet{$S}(")
    print(io, "models=$(length(get_sole(m)))")

    isnothing(measures(m)) || print(io, ", measures=$(length(get_measures(m)))")

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

"""
    show_measures(m::ModelSet) -> nothing

Print each performance measure and its aggregate value.

Requires `m` to contain evaluation results.
"""
function show_measures(m::ModelSet)
    println("Performance Measures:")
    for (ms, v) in zip(get_measures(m).measures, get_values(m))
        v isa Real ?
            println("  $(ms) = $(round(v, digits=2))") :
            println("  $(ms) = $(v)")
    end
end

# ---------------------------------------------------------------------------- #
#                                 utilities                                    #
# ---------------------------------------------------------------------------- #
# return deterministic predictions stored in `solem`.
# y_test is accepted for compatibility with MLJ prediction operations.
function supporting_predictions(solem::AbstractModel)
    return solem.info isa Base.RefValue ?
        solem.info[].supporting_predictions :
        solem.info.supporting_predictions
end

# return MLJ-compatible predictions for a symbolic model.
# for classification, deterministic class predictions are converted to
# degenerate `UnivariateFinite` distributions using the classes observed in
# y_test. Regression predictions are returned unchanged.
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

# ---------------------------------------------------------------------------- #
#                                eval measures                                 #
# ---------------------------------------------------------------------------- #
# adapted from MLJ's evaluate
# evaluate `measures` on the test partition of every fold and aggregate their
# values according to each measure's external aggregation mode.
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
# evaluate an existing `ModelSet` in-place and return it.
# existing measures are replaced. When `measures` is empty, task-appropriate
# default measures are selected from the target type.
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
    solexplorer!(modelset::ModelSet; kwargs...)

Perform additional analysis on an existing `ModelSet` in-place.

Adds or updates performance measures on an existing `ModelSet`.

# Keyword Arguments
- `measures::Tuple{Vararg{FussyMeasure}}=()`: Performance measures to
  compute. If empty, default measures for the task type are used.

# See also: [`solexplorer`](@ref), [`ModelSet`](@ref)
"""
solexplorer!(modelset::ModelSet; kwargs...) = _solexplorer!(modelset; kwargs...)

"""
    solexplorer(X::AbstractDataFrame, y::AbstractVector{<:Label}, args...;
                model, measures=(), kwargs...)
    solexplorer(X::AbstractArray, vnames::AbstractVector,
                y::AbstractVector{<:Label}, args...; model, measures=(),
                kwargs...)
    solexplorer(dt::DT.DataTreatment, args...; model, measures=(), kwargs...)
    solexplorer(ds::DataSet; measures=())
    solexplorer(ds::DataSet, solem::SoleModel; measures=())

Run the complete symbolic-model analysis workflow.

For raw data or a `DataTreatment`, this function configures a dataset, trains
one symbolic model for each partition, and evaluates those models on the
corresponding test partitions. Passing a `DataSet` skips dataset setup;
passing both a `DataSet` and `SoleModel` skips training as well.

# Arguments
- `X`: Tabular feature data, with observations in rows.
- `vnames`: Names used when converting an array input to a `DataFrame`.
- `y`: Target labels or continuous targets.
- `dt`: Pre-configured `DataTreatment`, typically used for modal data.
- `ds`: Pre-configured dataset.
- `solem`: Pre-trained symbolic model associated with `ds`.
- `args...`: Positional options forwarded to [`setup_dataset`](@ref).

# Keyword Arguments
- `model::MLJ.Model`: Required when constructing a dataset from `X` or `dt`.
- `measures::Tuple{Vararg{FussyMeasure}}=()`: Measures to evaluate. Empty
  tuples select task-appropriate defaults.
- `kwargs...`: Additional options forwarded to [`setup_dataset`](@ref), such
  as `resampling`, `rng`, and data-treatment options.

# Examples
```julia
using SoleXplorer, MLJ

X, y = @load_iris

modelset = solexplorer(
    DataFrame(X),
    y;
    model=DecisionTreeClassifier(),
    resampling=CV(nfolds=5, shuffle=true),
    rng=1,
)

show_measures(modelset)
```

# See also
[`ModelSet`](@ref), [`setup_dataset`](@ref), [`solexplorer!`](@ref)
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

function solexplorer(
    ds::DataSet;
    kwargs...
)
    _solexplorer(ds, _train_test(ds); kwargs...)
end

solexplorer(X::AbstractArray, vnames::AbstractVector, args...; kwargs...) =
    solexplorer(DataFrame(X, vnames), args...; kwargs...)

