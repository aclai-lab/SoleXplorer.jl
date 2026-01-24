# symbolic_analysis.jl — Unified Symbolic Model Analysis Interface

# this module provides the main entry point for complete symbolic model analysis workflows:

# 1. Dataset setup and cross-validation training (via train_test.jl)
# 2. Training and testing models (via train_test.jl)
# 3. Rule extraction from symbolic models (via extractrules.jl) 
# 4. Performance evaluation using MLJ measures

# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
"""
    AbstractModelSet

Abstract type for containers that hold symbolic model analysis results.

# Concrete Implementations
- [`ModelSet`](@ref): The primary implementation containing complete analysis results

See also: [`symbolic_analysis`](@ref)
"""
abstract type AbstractModelSet end

# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
const MaybeRules         = Maybe{Union{Vector{DecisionSet}, Vector{LumenResult}}}
const MaybeMeasures      = Maybe{Measures}
# const MaybeAssociaRules  = Maybe{Vector{ARule}}
# const MaybeAssociation   = Maybe{AbstractAssociationRuleExtractor}
const MaybeRuleExtractor = Maybe{RuleExtractor}

# ---------------------------------------------------------------------------- #
#                                  modelset                                    #
# ---------------------------------------------------------------------------- #
"""
    ModelSet{S} <: AbstractModelSet

Wrapper for complete symbolic model analysis results.

This structure holds all components of a symbolic analysis workflow including
the dataset configuration, sole trained models, extracted rules, association rules,
and performance measures.

# Type Parameters
- `S`: The sole model type (e.g., DecisionTreeClassifier)

# Fields
- `ds::AbstractDataSet`: Dataset configuration with cross-validation setup,
   plus all settings needed by modal analysis.
- `sole::Vector{AbstractModel}`: Vector of trained symbolic models (one per CV fold)
### Optional
- `rules::MaybeRules`: Extracted rules
- `associations::MaybeAssociations`: Association rules between features
- `measures::MaybeMeasures`: Performance evaluation measures

# Accessing Components
- [`dsetup`](@ref): Extract dataset configuration
- [`solemodels`](@ref): Extract trained models
- [`rules`](@ref): Extract decision rules
- [`associations`](@ref): Extract association rules

See also: [`symbolic_analysis`](@ref)
"""
mutable struct ModelSet{S} <: AbstractModelSet
    ds           :: AbstractDataSet
    sole         :: Vector{AbstractModel}
    rules        :: MaybeRules
    # associations :: MaybeAssociaRules
    measures     :: MaybeMeasures

    function ModelSet(
        ds       :: AbstractDataSet,
        sole     :: SoleModel{S};
        rules    :: MaybeRules=nothing,
        # miner    :: MaybeAssociaRules=nothing,
        measures :: MaybeMeasures=nothing
    ) where S
        # new{S}(ds, solemodels(sole), rules, miner, measures)
        new{S}(ds, solemodels(sole), rules, measures)
    end
end

# ---------------------------------------------------------------------------- #
#                                 constructors                                 #
# ---------------------------------------------------------------------------- #
"""
    dsetup(m::ModelSet) -> AbstractDataSet

Returns the dataset configuration from a ModelSet.

See also: [`ModelSet`](@ref), [`symbolic_analysis`](@ref)
"""
dsetup(m::ModelSet) = m.ds

"""
    solemodels(m::ModelSet) -> Vector{AbstractModel}

Returns the trained sole symbolic models from a ModelSet.

See also: [`ModelSet`](@ref), [`symbolic_analysis`](@ref)
"""
solemodels(m::ModelSet) = m.sole

"""
    rules(m::ModelSet) -> MaybeRules

Returns the rules extracted from a ModelSet.
Returns nothing if rule extraction isn't yet performed.

See also: [`ModelSet`](@ref), [`symbolic_analysis`](@ref)
"""
rules(m::ModelSet) = m.rules

# """
#     associations(m::ModelSet) -> MaybeAssociaRules

# Returns the association rules extracted from a ModelSet.
# Returns nothing if association rules isn't yet performed.

# See also: [`ModelSet`](@ref), [`symbolic_analysis`](@ref)
# """
# associations(m::ModelSet) = m.associations

"""
    performance(m::ModelSet) -> MaybeMeasures

Extract the performance evaluation measures from a ModelSet.

See also: [`ModelSet`](@ref), [`symbolic_analysis`](@ref)
"""
performance(m::ModelSet) = m.measures

"""
    measures(m::ModelSet) -> Vector

Extract the performance measure objects from a ModelSet.

# See also: [`performance`](@ref), [`ModelSet`](@ref)
"""
measures(m::ModelSet) = performance(m).measures

"""
    values(m::ModelSet) -> Vector

Extract the computed performance measure values from a ModelSet.

# See also: [`performance`](@ref), [`ModelSet`](@ref)
"""
values(m::ModelSet) = performance(m).measures_values

# ---------------------------------------------------------------------------- #
#                                  base show                                   #
# ---------------------------------------------------------------------------- #
function Base.show(io::IO, m::ModelSet{S}) where S
    print(io, "ModelSet{$S}(")
    print(io, "models=$(length(solemodels(m)))")

    isnothing(rules(m))        || print(io, ", rules=$(length(rules(m)))")
    # isnothing(associations(m)) || print(io, ", associations=$(length(associations(m)))")
    isnothing(measures(m))     || print(io, ", measures=$(length(measures(m)))")

    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", m::ModelSet{S}) where S
    println(io, "ModelSet{$S}:")
    println(io, "  Dataset: $(typeof(dsetup(m)))")
    println(io, "  Models:  $(length(solemodels(m))) symbolic models")
    
    isnothing(rules(m)) ?
        println(io, "  Rules: none") :
        println(io, "  Rules: $(length(first(rules(m)))) extracted rules per model")

    # isnothing(associations(m)) ?
    #     println(io, "  Associations: none") :
    #     println(io, "  Associations: $(length(associations(m))) associated rules per model")
    
    isnothing(performance(m)) ?
        println(io, "  Measures: none") : begin
            println(io, "  Measures:")
            for (measure, value) in zip(measures(m), values(m))
                println(io, "    $(measure) = $(value)")
            end
        end
end

function show_measures(m::ModelSet)
    println("Performance Measures:")
    for (m, v) in zip(measures(m), values(m))
        v isa Real ? println("  $(m) = $(round(v, digits=2))") : println("  $(m) = $(v)")
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

sole_predict_mode(solem::AbstractModel, y_test::AbstractVector{<:Label}) = supporting_predictions(solem)

# if it's a classification task, convert, if needed,
# predictions in categorical values.
# if not, wnon't do anything.
function sole_predict(solem::AbstractModel, y_test::AbstractVector{<:Label})
    preds = supporting_predictions(solem)
    return eltype(preds) <: CLabel ?
        begin
            classes_seen = unique(y_test)
            eltype(preds) <: MLJ.CategoricalValue ||
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
    ds::AbstractDataSet,
    solem::Vector{AbstractModel},
    measures::Tuple{Vararg{FussyMeasure}},
    y_test::Vector{<:AbstractVector{<:Label}}
)::Measures
    mach_model      = get_mach_model(ds)
    measures        = MLJBase._actual_measures([measures...], mach_model)
    operations      = get_operations(measures, MLJBase.prediction_type(mach_model))

    nfolds          = length(ds)
    test_fold_sizes = [length(y_test[k]) for k in 1:nfolds]
    nmeasures       = length(measures)

    # weights used to aggregate per-fold measurements, which depends on a measures
    # external mode of aggregation:
    fold_weights(mode) = nfolds .* test_fold_sizes ./ sum(test_fold_sizes)
    fold_weights(::MLJBase.StatisticalMeasuresBase.Sum) = nothing
    
    measurements_vector = mapreduce(vcat, 1:nfolds) do k
        yhat_given_operation = Dict(op=>op(solem[k], y_test[k]) for op in unique(operations))

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
#                         internal symbolic_analysis                           #
# ---------------------------------------------------------------------------- #
function _symbolic_analysis!(
    modelset    :: AbstractModelSet;
    extractor   :: Union{MaybeRuleExtractor,Tuple{RuleExtractor,NamedTuple}}=nothing,
    # extractor::MaybeRuleExtractor=nothing,
    # association :: MaybeAbstractAssociationRuleExtractor=nothing,
    measures    :: Tuple{Vararg{FussyMeasure}}=()
)::ModelSet
    ds    = dsetup(modelset)
    solem = solemodels(modelset)

    !isnothing(extractor) && (modelset.rules = begin
        if extractor isa Tuple
            params    = last(extractor)
            extractor = first(extractor)
        else
            params = NamedTuple(;)
        end

        :rng ∈ fieldnames(typeof(extractor)) && (extractor = set_rng(extractor, get_rng(ds)))
        extractrules(extractor, params, ds, solem)
    end)

    # !isnothing(association) && (modelset.associations = mas_caller(ds, association))

    y_test = get_y(ds, :test)
    isempty(measures) && (measures = _DefaultMeasures(first(y_test)))

    modelset.measures = eval_measures(ds, solem, measures, y_test)

    return modelset
end

function _symbolic_analysis(
    ds::AbstractDataSet,
    solem::SoleModel;
    kwargs...
)::ModelSet
    modelset = ModelSet(ds, solem)
    _symbolic_analysis!(modelset; kwargs...)
    return modelset
end

# ---------------------------------------------------------------------------- #
#                              symbolic_analysis                               #
# ---------------------------------------------------------------------------- #
"""
    symbolic_analysis!(modelset::ModelSet; kwargs...)

Perform additional analysis on an existing ModelSet.

In-place version that adds or updates analysis components (rules, associations,
measures) on an existing ModelSet.

# Examples
```julia
# initial analysis
modelset = symbolic_analysis(X, y)

# add rule extraction later
symbolic_analysis!(modelset; extractor=Lumen())

# add association mining later
symbolic_analysis!(modelset; association=Apriori())
```

See also: [`symbolic_analysis`](@ref), [`ModelSet`](@ref)
"""
symbolic_analysis!(modelset::ModelSet; kwargs...)::ModelSet = _symbolic_analysis!(modelset; kwargs...)

"""
    symbolic_analysis(
        X::AbstractDataFrame,
        y::AbstractVector,
        w::MaybeVector=nothing;
        extractor::MaybeRuleExtractor=nothing,
        association::Union{Nothing,AbstractAssociationRuleExtractor}=nothing,
        measures::Tuple{Vararg{FussyMeasure}}=(),
        kwargs...
    ) -> ModelSet

Complete end-to-end symbolic model analysis workflow.

This is the main entry point for symbolic analysis.
It performs the complete workflow:
1. **Dataset Setup**: Configures cross-validation and time series preprocessing.
2. **Model Configuration**: Sets up the MLJ machine.
3. **Model Training**: Trains symbolic models on each CV fold.
4. **Rule Extraction**: Extracts interpretable rules from trained models.
5. **Association Mining**: Discovers feature relationships and patterns.
6. **Evaluation**: Computes comprehensive performance metrics.

# Arguments
- `X::AbstractDataFrame`: Feature matrix with observations as rows
- `y::AbstractVector`: Target variable (labels for classification)
- `w::MaybeVector`: Sample weights (optional)

## Analysis Options
- `extractor`: Rule extraction method:
  See [SolePostHoc](https://github.com/aclai-lab/SolePostHoc.jl)
- `association`: Association rule mining:
  See [ModalAssociationRules](https://github.com/aclai-lab/ModalAssociationRules.jl)
- `measures`: Performance measures tuple:
  - `(accuracy, auc, f1_score)`: Custom measures
  - `()`: Use default measures for task type

## Dataset & Training Options (kwargs)
  See [`setup_dataset`](@Ref)

# Examples
```julia
# Basic analysis with default settings
modelset = symbolic_analysis(X, y)

# Full analysis example
range = SoleXplorer.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
modelset = symbolic_analysis(X, y;
    model=RandomForestClassifier(),
    resampling=CV(nfolds=5, shuffle=true),
    seed=1,
    tuning=GridTuning(resolution=10, resampling=CV(nfolds=3), range=range, measure=accuracy, repeats=2),
    extractor=InTreesRuleExtractor(),
    measures=(accuracy, log_loss, confusion_matrix, kappa)   
)

# Time series analysis example
modelset = symbolic_analysis(X, y;
    model=ModalRandomForest(),
    resampling=Holdout(fraction_train=0.7, shuffle=true),
    seed=1,
    features=(minimum, maximum),
    measures=(log_loss, accuracy, confusion_matrix, kappa)
)

# Access modelset
setup = dsetup(modelset)
models = solemodels(modelset)
rules = rules(modelset)  
associations = associations(modelset)
performance = performance(modelset)
```

See also: [`ModelSet`](@ref), [`setup_dataset`](@Ref), [`train_test`](@Ref)
"""
function symbolic_analysis(
    X::AbstractDataFrame,
    y::AbstractVector,
    w::MaybeVector=nothing;
    extractor::MaybeRuleExtractor=nothing,
    # association::Union{Nothing,AbstractAssociationRuleExtractor}=nothing,
    measures::Tuple{Vararg{FussyMeasure}}=(),
    kwargs...
)::ModelSet
    ds = setup_dataset(X, y, w; kwargs...)
    solem = _train_test(ds)
    # _symbolic_analysis(ds, solem; extractor, association, measures)
    _symbolic_analysis(ds, solem; extractor, measures)
end

"""
    symbolic_analysis(ds::AbstractDataSet, solem::SoleModel; kwargs...) -> ModelSet

Perform complete symbolic analysis on pre-trained models.

# Arguments  
- `ds::AbstractDataSet`: Dataset used for training the models.
- `solem::SoleModel`: Trained sole symbolic models.
- `kwargs...`: Analysis options (extractor, association, measures).

# Examples
```julia
# analyze pre-trained models
ds = setup_dataset(
    X, y;
    model=DecisionTreeClassifier(),
    resampling=CV(nfolds=5, shuffle=true)
)
solem = train_test(ds)
results = symbolic_analysis(
    ds, solem; 
    extractor=Lumen(),
    measures=(accuracy, kappa)
)
```

See also: [`symbolic_analysis!`](@ref), [`setup_dataset`](@ref), [`train_test`](@ref)
"""
function symbolic_analysis(
    ds::AbstractDataSet,
    solem::SoleModel;
    kwargs...
)::ModelSet
    _symbolic_analysis(ds, solem; kwargs...)
end

"""
    symbolic_analysis(X::Any, args...; kwargs...) -> ModelSet

Convenience method that converts input data to DataFrame format.

See also: [`symbolic_analysis`](@ref)
"""
symbolic_analysis(X::Any, args...; kwargs...)::ModelSet = symbolic_analysis(DataFrame(X), args...; kwargs...)

