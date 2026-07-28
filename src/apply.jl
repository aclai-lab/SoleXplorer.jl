# apply.jl - Unified Model Prediction Interface
#
# Overview
# --------
# `apply(m, X, y)` converts a fitted MLJ Machine into a SoleModels symbolic model.
#
# Input
#   m :: Machine   – a fitted MLJ machine
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

# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
const ModalApply = Union{
    Machine{ModalDecisionTree},
    Machine{ModalRandomForest},
    Machine{ModalAdaBoost},
}

# ---------------------------------------------------------------------------- #
#                              xgboost utilities                               #
# ---------------------------------------------------------------------------- #
# extract base_score from XGBoost models,
function get_base_score(m::Machine)
    return hasproperty(m.model, :base_score) ?
        m.model.base_score :
        nothing
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

