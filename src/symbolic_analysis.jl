"""
symbolic_analysis.jl — Unified Symbolic Model Analysis Interface

This module provides the main entry point for complete symbolic model analysis workflows:

1. Dataset setup and cross-validation training (via train_test.jl)
2. Training and testing models (via train_test.jl)
3. Rule extraction from symbolic models (via extractrules.jl) 
4. Performance evaluation using MLJ measures
"""

# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
"""
    AbstractModelSet  

Base type for comprehensive analysis result containers.
"""
abstract type AbstractModelSet end

# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
const MaybeRules         = Maybe{Union{Vector{DecisionSet}, Vector{LumenResult}}}
const MaybeMeasures      = Maybe{Measures}
const MaybeAssociations  = Maybe{Vector{ARule}}

const MaybeRuleExtractor = Maybe{RuleExtractor}
    # association::Union{Nothing,AbstractAssociationRuleExtractor}

# ---------------------------------------------------------------------------- #
#                                  modelset                                    #
# ---------------------------------------------------------------------------- #
"""
    ModelSet{S} <: AbstractModelSet

Comprehensive container for symbolic model analysis results.

# Fields
- `ds::EitherDataSet`: Dataset wrapper used for training
- `sole::Vector{AbstractModel}`: Symbolic models from each CV fold
- `rules::MaybeRules`: Extracted decision rules (optional)
- `measures::MaybeMeasures`: Performance evaluation results (optional)
"""
mutable struct ModelSet{S} <: AbstractModelSet
    ds           :: EitherDataSet
    sole         :: Vector{AbstractModel}
    rules        :: MaybeRules
    associations :: MaybeAssociations
    measures     :: MaybeMeasures

    function ModelSet(
        ds       :: EitherDataSet,
        sole     :: SoleModel{S};
        rules    :: MaybeRules=nothing,
        miner    :: MaybeAssociations=nothing,
        measures :: MaybeMeasures=nothing
    ) where S
        new{S}(ds, solemodels(sole), rules, miner, measures)
    end
end

function Base.show(io::IO, m::ModelSet{S}) where S
    print(io, "ModelSet{$S}(")
    print(io, "models=$(length(m.sole))")
    if !isnothing(m.rules)
        print(io, ", rules=$(length(m.rules.rules))")
    end
        if !isnothing(m.associations)
        print(io, ", associations=$(length(m.associations))")
    end
    if !isnothing(m.measures)
        print(io, ", measures=$(length(m.measures.measures))")
    end
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", m::ModelSet{S}) where S
    println(io, "ModelSet{$S}:")
    println(io, "  Dataset: $(typeof(m.ds))")
    println(io, "  Models: $(length(m.sole)) symbolic models")
    
    if !isnothing(m.rules)
        println(io, "  Rules: $(length(first(m.rules))) extracted rules per model")
    else
        println(io, "  Rules: none")
    end

    if !isnothing(m.associations)
        println(io, "  Associations: $(length(m.associations)) associated rules per model")
    else
        println(io, "  Associations: none")
    end
    
    if !isnothing(m.measures)
        println(io, "  Measures:")
        for (measure, value) in zip(m.measures.measures, m.measures.measures_values)
            println(io, "    $(measure) = $(value)")
        end
    else
        println(io, "  Measures: none")
    end
end

# ---------------------------------------------------------------------------- #
#                                 utilities                                    #
# ---------------------------------------------------------------------------- #
"""
    supporting_predictions(solem::AbstractModel) -> Vector

Extract supporting predictions from a symbolic model.
"""
function supporting_predictions(solem::AbstractModel)
    return solem.info isa Base.RefValue ?
        solem.info[].supporting_predictions :
        solem.info.supporting_predictions
end

"""
    sole_predict(solem::AbstractModel, y_test::AbstractVector{<:Label}) -> Vector

Convert symbolic model predictions to MLJ probabilistic format.
"""
function sole_predict(solem::AbstractModel, y_test::AbstractVector{<:Label})
    preds = supporting_predictions(solem)
    eltype(preds) <: CLabel ?
        begin
            classes_seen = unique(y_test)
            eltype(preds) <: MLJ.CategoricalValue ||
                (preds = categorical(preds, levels=levels(classes_seen)))
            [UnivariateFinite([p], [1.0]) for p in preds]
        end :
        preds
end

"""
    sole_predict_mode(solem::AbstractModel, y_test::AbstractVector{<:Label}) -> Vector

Return deterministic predictions from symbolic model.
"""
sole_predict_mode(solem::AbstractModel, y_test::AbstractVector{<:Label}) = supporting_predictions(solem)

# ---------------------------------------------------------------------------- #
#                              symbolic_analysis                               #
# ---------------------------------------------------------------------------- #
function _symbolic_analysis!(
    modelset::ModelSet;
    extractor::MaybeRuleExtractor=nothing,
    association::MaybeAbstractAssociationRuleExtractor=nothing,
    measures::Tuple{Vararg{FussyMeasure}}=()
)::Nothing
    ds = dsetup(modelset)
    solem = solemodels(modelset)

    modelset.rules = isnothing(extractor) ? nothing : begin
        # TODO propaga rng, dovrai fare intrees mutable struct
        if extractor isa Tuple
            params = last(extractor)
            extractor = first(extractor)
        else
            params = NamedTuple(;)
        end
        extractrules(extractor, params, ds, solem)
    end

    modelset.associations = isnothing(association) ? nothing : mas_caller(ds, association)

    y_test = get_y_test(ds)
    isempty(measures) && (measures = _DefaultMeasures(first(y_test)))
    # all_classes = unique(Iterators.flatten(y_test))
    modelset.measures = eval_measures(ds, solem, measures, y_test)

    return nothing
end

function _symbolic_analysis(
    ds::EitherDataSet,
    solem::SoleModel;
    kwargs...
)::ModelSet
    modelset = ModelSet(ds, solem)
    _symbolic_analysis!(modelset; kwargs...)
    return modelset
end

"""
    symbolic_analysis(ds::EitherDataSet, solem::SoleModel; 
                     extractor=nothing, measures=()) -> ModelSet

Perform symbolic analysis on pre-trained models.

Use when you already have trained symbolic models and want to add
rule extraction and/or performance evaluation.
"""
function symbolic_analysis(
    ds::EitherDataSet,
    solem::SoleModel;
    kwargs...
)::ModelSet
    _symbolic_analysis(ds, solem; kwargs...)
end

function symbolic_analysis!(
    modelset::ModelSet; 
    kwargs...
)::ModelSet
    _symbolic_analysis!(modelset; kwargs...)
    return modelset
end

"""
    symbolic_analysis(X::AbstractDataFrame, y::AbstractVector, [w];
        [extractor::MaybeRuleExtracton], 
        [association::MaybeAbstractAssociationRuleExtractor],
        [measures::Tuple{Vararg{FussyMeasure}}],
        kwargs...) -> ModelSet

End-to-end symbolic analysis starting from raw data.

# Arguments:
- `X, y, w`    : Features, targets, and optional weights
- `extractor`  : Rule extraction strategy
- `association`: Rule association strategy
- `measures`   : Performance measures to evaluate (accuracy, auc, etc.)
- `kwargs`     : Passed to dataset setup (model, cv_folds, etc.)

# extractor

# association

# measures

See [`setup_dataset`](@ref) for dataset setup parameter descriptions.
"""
function symbolic_analysis(
    X::AbstractDataFrame,
    y::AbstractVector,
    w::MaybeVector=nothing;
    extractor::MaybeRuleExtractor=nothing,
    association::Union{Nothing,AbstractAssociationRuleExtractor}=nothing,
    measures::Tuple{Vararg{FussyMeasure}}=(),
    kwargs...
)::ModelSet
    ds = _setup_dataset(X, y, w; kwargs...)
    solem = _train_test(ds)
    _symbolic_analysis(ds, solem; extractor, association, measures)
end

symbolic_analysis(X::Any, args...; kwargs...) = symbolic_analysis(DataFrame(X), args...; kwargs...)

# ---------------------------------------------------------------------------- #
#                                 constructors                                 #
# ---------------------------------------------------------------------------- #
"""
    dsetup(m::ModelSet) -> EitherDataSet

Extract the dataset setup from a ModelSet.
"""
dsetup(m::ModelSet) = m.ds

"""
    solemodels(m::ModelSet) -> Vector{AbstractModel}

Extract the vector of symbolic models from a ModelSet.
"""
solemodels(m::ModelSet) = m.sole

"""
    rules(m::ModelSet) -> DecisionSet

Extract the vector of rules from a ModelSet.
"""
rules(m::ModelSet) = m.rules

"""
    associations(m::ModelSet) -> Vector{ARule}

Extract the vector of associations from a ModelSet.
"""
associations(m::ModelSet) = m.associations
