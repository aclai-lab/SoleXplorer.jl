# ---------------------------------------------------------------------------- #
#                                 utilities                                    #
# ---------------------------------------------------------------------------- #
get_featid(s::Branch{T}) where T = s.antecedent.value.metacond.feature.i_variable
get_cond(s::Branch{T}) where T = s.antecedent.value.metacond.test_operator
get_thr(s::Branch{T}) where T = s.antecedent.value.threshold

function set_predictions(
    info::NamedTuple,
    preds::Vector{T},
    y::AbstractVector{S}
)::NamedTuple where {T,S<:Label}
    typeof(info)(merge(MLJ.params(info), (supporting_predictions=preds, supporting_labels=y)))
end

# ---------------------------------------------------------------------------- #
#                            propositional apply                               #
# ---------------------------------------------------------------------------- #
function propositional_apply!() end

function propositional_apply!(
    solem::DecisionTree{T},
    X::AbstractDataFrame,
    y::AbstractVector{S}
)::Nothing where {T, S<:CLabel}
    predictions = CLabel[propositional_apply!(solem.root, x) for x in eachrow(X)]
    solem.info = set_predictions(solem.info, predictions, y)
    return nothing
end

function propositional_apply!(
    solem::DecisionTree{T},
    X::AbstractDataFrame,
    y::AbstractVector{S}
)::Nothing where {T, S<:RLabel}
    predictions = RLabel[propositional_apply!(solem.root, x) for x in eachrow(X)]
    solem.info = set_predictions(solem.info, predictions, y)
    return nothing
end

function propositional_apply!(soleroot::Branch{T}, x::DataFrameRow)::T where T
    featid, cond, thr = get_featid(soleroot), get_cond(soleroot), get_thr(soleroot)
    feature_value = x[featid]
    condition_result = cond(feature_value, thr)
    
    return condition_result ?
        propositional_apply!(soleroot.posconsequent, x) :
        propositional_apply!(soleroot.negconsequent, x)
end

function propositional_apply!(leaf::ConstantModel{T}, ::DataFrameRow)::T where T
    leaf.outcome
end