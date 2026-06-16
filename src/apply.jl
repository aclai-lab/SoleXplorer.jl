# apply.jl - Unified Model Prediction Interface
#
# Overview
# --------
# `apply(m, X, y)` converts a fitted MLJ Machine into a SoleModels symbolic model.
#
# Input
#   m :: Machine   – a fitted MLJ machine (regular or TunedModel-wrapped)
#   X :: AbstractDataFrame – feature matrix
#   y :: AbstractVector    – target labels/values
#
# Output :: AbstractModel (SoleModels)
#   DecisionTree      – single tree models (classifier or regressor)
#   DecisionEnsemble  – forest/boosting ensemble models
#   DecisionXGBoost   – XGBoost-specific ensemble
#
# Pipeline (for each method)
#   1. extract featurenames, classlabels, weights from m.report / m.fitresult
#   2. extract the raw fitted model (tree/forest/stumps/XGBoost trees)
#   3. build symbolic model via `solemodel(...)`
#   4. wrap X into a `PropositionalLogiset`
#   5. annotate model with ground-truth via `apply!(solem, logiset, y)`
#   6. return the annotated symbolic model
#
# SUPPORTED MACHINES
# ------------------
# Package               | Model type                | Output
# ----------------------|---------------------------|----------------
# DecisionTree.jl       | DecisionTreeClassifier    | DecisionTree
#                       | DecisionTreeRegressor     | DecisionTree
#                       | RandomForestClassifier    | DecisionEnsemble
#                       | RandomForestRegressor     | DecisionEnsemble
#                       | AdaBoostStumpClassifier   | DecisionEnsemble
# ModalDecisionTrees.jl | ModalDecisionTree         | DecisionTree
#                       | ModalRandomForest         | DecisionEnsemble
#                       | ModalAdaBoost             | DecisionEnsemble
# XGBoost.jl            | XGBoostClassifier         | DecisionXGBoost
#                       | XGBoostRegressor          | DecisionXGBoost
#
# All models support both plain Machine and TunedModel-wrapped variants.
# Tuned variants access params via `.best_report` / `.best_fitted_params`.

# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
const ModalApply = Union{
    Machine{ModalDecisionTree},
    Machine{ModalRandomForest},
    Machine{ModalAdaBoost},
}

const TunedModalApply = Union{
    Machine{<:MLJ.MLJTuning.EitherTunedModel{<:Any, <:ModalDecisionTree}},
    Machine{<:MLJ.MLJTuning.EitherTunedModel{<:Any, <:ModalRandomForest}},
    Machine{<:MLJ.MLJTuning.EitherTunedModel{<:Any, <:ModalAdaBoost}}
}

# ---------------------------------------------------------------------------- #
#                              xgboost utilities                               #
# ---------------------------------------------------------------------------- #
# extract base_score from XGBoost models,
# handling both regular and tuned variants
function get_base_score(m::Machine)
    if m.model isa MLJTuning.EitherTunedModel
        return hasproperty(m.model.model, :base_score) ?
            m.model.model.base_score :
            nothing
    else
        return hasproperty(m.model, :base_score) ?
            m.model.base_score :
            nothing
    end
end

# ---------------------------------------------------------------------------- #
#                             DecisionTree package                             #
# ---------------------------------------------------------------------------- #
function apply(
    m::Machine{DecisionTreeClassifier},
    X::AbstractDataFrame,
    y::AbstractVector{<:CLabel}
)::DecisionTree
    featurenames = MLJ.report(m).features
    classlabels = sort(MLJ.report(m).classes_seen)
    solem = solemodel(
        MLJ.fitted_params(m).tree;
        featurenames,
        classlabels
    )
    logiset = PropositionalLogiset(X)
    apply!(solem, logiset, y)
    return solem
end

function apply(
    m::Machine{<:MLJ.MLJTuning.EitherTunedModel{
        <:Any,<:DecisionTreeClassifier}},
    X::AbstractDataFrame,
    y::AbstractVector
)::DecisionTree
    featurenames = MLJ.report(m).best_report.features
    classlabels = sort(MLJ.report(m).best_report.classes_seen)
    solem = solemodel(
        MLJ.fitted_params(m).best_fitted_params.tree;
        featurenames,
        classlabels
    )
    logiset = PropositionalLogiset(X)
    apply!(solem, logiset, y)
    return solem
end

# ---------------------------------------------------------------------------- #
function apply(
    m::Machine{RandomForestClassifier},
    X::AbstractDataFrame,
    y::AbstractVector
)::DecisionEnsemble
    featurenames = MLJ.report(m).features
    classlabels = m.fitresult[2][sortperm((m).fitresult[3])]
    solem = solemodel(
        MLJ.fitted_params(m).forest;
        featurenames,
        classlabels,
        tiebreaker=:alphanumeric
    )
    logiset = PropositionalLogiset(X)
    apply!(solem, logiset, y)
    return solem
end

function apply(
    m::Machine{<:MLJ.MLJTuning.EitherTunedModel{
        <:Any,<:RandomForestClassifier}},
    X::AbstractDataFrame,
    y::AbstractVector
)::DecisionEnsemble
    featurenames = MLJ.report(m).best_report.features
    classlabels = m.fitresult.fitresult[2][sortperm((m).fitresult.fitresult[3])]
    solem = solemodel(
        MLJ.fitted_params(m).best_fitted_params.forest;
        featurenames,
        classlabels
    )
    logiset = PropositionalLogiset(X)
    apply!(solem, logiset, y)
    return solem
end

# ---------------------------------------------------------------------------- #
function apply(
    m::Machine{DecisionTreeRegressor},
    X::AbstractDataFrame,
    y::AbstractVector
)::DecisionTree
    featurenames = MLJ.report(m).features
    solem = solemodel(
        MLJ.fitted_params(m).tree;
        featurenames
    )
    logiset = PropositionalLogiset(X)
    apply!(solem, logiset, y)
    return solem
end

function apply(
    m::Machine{<:MLJ.MLJTuning.EitherTunedModel{
        <:Any,<:DecisionTreeRegressor}},
    X::AbstractDataFrame,
    y::AbstractVector
)::DecisionTree
    featurenames = MLJ.report(m).best_report.features
    solem = solemodel(
        MLJ.fitted_params(m).best_fitted_params.tree;
        featurenames
    )
    logiset = PropositionalLogiset(X)
    apply!(solem, logiset, y)
    return solem
end

# ---------------------------------------------------------------------------- #
function apply(
    m::Machine{RandomForestRegressor},
    X::AbstractDataFrame,
    y::AbstractVector
)::DecisionEnsemble
    featurenames = MLJ.report(m).features
    solem = solemodel(
        MLJ.fitted_params(m).forest;
        featurenames
    )
    logiset = PropositionalLogiset(X)
    apply!(solem, logiset, y)
    return solem
end

function apply(
    m::Machine{<:MLJ.MLJTuning.EitherTunedModel{
        <:Any,<:RandomForestRegressor}},
    X::AbstractDataFrame,
    y::AbstractVector
)::DecisionEnsemble
    featurenames = MLJ.report(m).best_report.features
    solem = solemodel(
        MLJ.fitted_params(m).best_fitted_params.forest;
        featurenames
    )
    logiset = PropositionalLogiset(X)
    apply!(solem, logiset, y)
    return solem
end

# ---------------------------------------------------------------------------- #
function apply(
    m::Machine{AdaBoostStumpClassifier},
    X::AbstractDataFrame,
    y::AbstractVector
)::DecisionEnsemble
    featurenames = MLJ.report(m).features
    classlabels = sort(string.(m.fitresult[3]))
    weights = m.fitresult[2]
    solem = solemodel(
        MLJ.fitted_params(m).stumps;
        featurenames,
        classlabels,
        weights
    )
    logiset = PropositionalLogiset(X)
    apply!(solem, logiset, y)
    return solem
end

function apply(
    m::Machine{<:MLJ.MLJTuning.EitherTunedModel{
        <:Any,<:AdaBoostStumpClassifier}},
    X::AbstractDataFrame,
    y::AbstractVector
)::DecisionEnsemble
    featurenames = MLJ.report(m).best_report.features
    classlabels = sort(m.fitresult.fitresult[3])
    weights = m.fitresult.fitresult[2]
    solem = solemodel(
        MLJ.fitted_params(m).best_fitted_params.stumps;
        featurenames,
        classlabels,
        weights
    )
    logiset = PropositionalLogiset(X)
    apply!(solem, logiset, y)
    return solem
end

# ---------------------------------------------------------------------------- #
#                           ModalDecisionTrees package                         #
# ---------------------------------------------------------------------------- #
function apply(
    m::ModalApply,
    X::AbstractDataFrame,
    y::AbstractVector
)::Union{DecisionTree, DecisionEnsemble}
    (_, solem) = MLJ.report(m).sprinkle(X, y)
    return solem
end

function apply(
    m::TunedModalApply,
    X::AbstractDataFrame,
    y::AbstractVector
)::Union{DecisionTree, DecisionEnsemble}
    (_, solem) = MLJ.report(m).best_report.sprinkle(X, y)
    return solem
end

# ---------------------------------------------------------------------------- #
#                                XGBoost package                               #
# ---------------------------------------------------------------------------- #
function apply(
    m::Machine{XGBoostClassifier},
    X::AbstractDataFrame,
    y::AbstractVector
)::DecisionXGBoost
    trees = XGBoost.trees(m.fitresult[1])
    featurenames = m.report.vals[1].features
    classlabels = MLJ.classes(m.fitresult[2])
    solem = solemodel(
        trees,
        Matrix(X),
        y;
        featurenames,
        classlabels
    )
    logiset = PropositionalLogiset(mapcols(col -> Float32.(col), X))
    apply!(solem, logiset, y)
    return solem
end

function apply(
    m::Machine{<:MLJ.MLJTuning.EitherTunedModel{
        <:Any,<:XGBoostClassifier}},
    X::AbstractDataFrame,
    y::AbstractVector
)::DecisionXGBoost
    trees = XGBoost.trees(m.fitresult.fitresult[1])
    featurenames = m.fitresult.report.vals[1].features
    classlabels = MLJ.classes(m.fitresult.fitresult[2])
    solem = solemodel(
        trees,
        Matrix(X),
        y;
        featurenames,
        classlabels
    )
    logiset = PropositionalLogiset(mapcols(col -> Float32.(col), X))
    apply!(solem, logiset, y)
    return solem
end

# ---------------------------------------------------------------------------- #
function apply(
    m::Machine{XGBoostRegressor},
    X::AbstractDataFrame,
    y::AbstractVector
)::DecisionXGBoost
    base_score = get_base_score(m) == -Inf ? mean(m.y[train]) : 0.5
    m.model.base_score = base_score

    trees = XGBoost.trees(m.fitresult[1])
    featurenames = m.report.vals[1].features
    solem = solemodel(
        trees,
        Matrix(X),
        y;
        featurenames
    )
    logiset = PropositionalLogiset(mapcols(col -> Float32.(col), X))
    apply!(solem, logiset, y; base_score)
    return solem
end

function apply(
    m::Machine{<:MLJ.MLJTuning.EitherTunedModel{
        <:Any,<:XGBoostRegressor}},
    X::AbstractDataFrame,
    y::AbstractVector,
)::DecisionXGBoost
    base_score = get_base_score(m) == -Inf ? mean(m.y[train]) : 0.5
    m.model.model.base_score = base_score

    trees = XGBoost.trees(m.fitresult.fitresult[1])
    featurenames = m.fitresult.report.vals[1].features
    solem = solemodel(
        trees,
        Matrix(X),
        y;
        featurenames
    )
    logiset = PropositionalLogiset(mapcols(col -> Float32.(col), X))
    apply!(solem, logiset, y; base_score)
    return solem
end
