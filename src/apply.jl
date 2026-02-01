# apply.jl - Unified Model Prediction Interface

# this methods provide `apply(m, X, y)` methods that:
# 1. extract fitted models from MLJ machine wrappers
# 2. convert them to SoleModels symbolic representations
# 3. create logical datasets and apply symbolic models
# 4. return symbolic models ready for rule extraction and analysis

# m = MLJ.Machine

# Supported ML packages:
# - DecisionTree.jl
# - ModalDecisionTrees.jl
# - XGBoost.jl

# all methods handle both regular and hyperparameter-tuned models

# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
const EitherBalancedModel{T,M} = Union{
    MLJ.MLJBalancing.BalancedModelProbabilistic{T,M},
    MLJ.MLJBalancing.BalancedModelDeterministic{T,M},
    MLJ.MLJBalancing.BalancedModelInterval{T,M}
}

const ModalApply = Union{
    Machine{ModalDecisionTree},
    Machine{ModalRandomForest},
    Machine{ModalAdaBoost},
}

const BalancedModalApply = Union{
    Machine{<:EitherBalancedModel{<:Any, <:ModalDecisionTree}},
    Machine{<:EitherBalancedModel{<:Any, <:ModalRandomForest}},
    Machine{<:EitherBalancedModel{<:Any, <:ModalAdaBoost}}
}

const TunedModalApply = Union{
    Machine{<:MLJ.MLJTuning.EitherTunedModel{<:Any, <:ModalDecisionTree}},
    Machine{<:MLJ.MLJTuning.EitherTunedModel{<:Any, <:ModalRandomForest}},
    Machine{<:MLJ.MLJTuning.EitherTunedModel{<:Any, <:ModalAdaBoost}}
}

const TunedBalancedModalApply = Union{
    Machine{<:MLJ.MLJTuning.EitherTunedModel{<:Any, <:EitherBalancedModel{<:Any, <:ModalDecisionTree}}},
    Machine{<:MLJ.MLJTuning.EitherTunedModel{<:Any, <:EitherBalancedModel{<:Any, <:ModalRandomForest}}},
    Machine{<:MLJ.MLJTuning.EitherTunedModel{<:Any, <:EitherBalancedModel{<:Any, <:ModalAdaBoost}}}
}

# ---------------------------------------------------------------------------- #
#                              xgboost utilities                               #
# ---------------------------------------------------------------------------- #
# extract base_score from XGBoost models, handling both regular and tuned variants
function get_base_score(m::Machine)
    if m.model isa MLJTuning.EitherTunedModel
        return hasproperty(m.model.model, :base_score) ? m.model.model.base_score : nothing
    else
        return hasproperty(m.model, :base_score) ? m.model.base_score : nothing
    end
end

# create mapping from internal class indices to MLJ class labels
get_encoding(classes_seen) = Dict(MLJ.int(c) => c for c in MLJ.classes(classes_seen))

# extract ordered class labels from encoding dictionary
get_classlabels(encoding)  = [string(encoding[i]) for i in sort(keys(encoding) |> collect)]

# ---------------------------------------------------------------------------- #
#                             DecisionTree package                             #
# ---------------------------------------------------------------------------- #
function apply(
    m :: Machine{DecisionTreeClassifier},
    X :: AbstractDataFrame,
    y :: AbstractVector{<:CLabel}
)::DecisionTree
    featurenames = MLJ.report(m).features
    classlabels  = sort(MLJ.report(m).classes_seen)
    solem        = solemodel(MLJ.fitted_params(m).tree; featurenames, classlabels)
    logiset      = scalarlogiset(X, allow_propositional=true)
    apply!(solem, logiset, y)
    return solem
end

function apply(
    m :: Machine{<:MLJ.MLJTuning.EitherTunedModel{<:Any, <:DecisionTreeClassifier}},
    X :: AbstractDataFrame,
    y :: AbstractVector
)::DecisionTree
    featurenames = MLJ.report(m).best_report.features
    classlabels  = sort(MLJ.report(m).best_report.classes_seen)
    solem        = solemodel(MLJ.fitted_params(m).best_fitted_params.tree; featurenames, classlabels)
    logiset      = scalarlogiset(X, allow_propositional=true)
    apply!(solem, logiset, y)
    return solem
end

function apply(
    m :: Machine{<:EitherBalancedModel{<:Any, <:DecisionTreeClassifier}},
    X :: AbstractDataFrame,
    y :: AbstractVector
)::DecisionTree
    featurenames = MLJ.report(m).model.features
    classlabels  = sort(MLJ.report(m).model.classes_seen)
    solem        = solemodel(MLJ.fitted_params(m).model.tree; featurenames, classlabels)
    logiset      = scalarlogiset(X, allow_propositional=true)
    apply!(solem, logiset, y)
    return solem
end

function apply(
    m :: Machine{<:MLJ.MLJTuning.EitherTunedModel{<:Any, <:EitherBalancedModel{<:Any, <:DecisionTreeClassifier}}},
    X :: AbstractDataFrame,
    y :: AbstractVector
)::DecisionTree
    featurenames = MLJ.report(m).best_report.model.features
    classlabels  = sort(MLJ.report(m).best_report.model.classes_seen)
    solem        = solemodel(MLJ.fitted_params(m).best_fitted_params.model.tree; featurenames, classlabels)
    logiset      = scalarlogiset(X, allow_propositional=true)
    apply!(solem, logiset, y)
    return solem
end

# ---------------------------------------------------------------------------- #

function apply(
    m :: Machine{RandomForestClassifier},
    X :: AbstractDataFrame,
    y :: AbstractVector
)::DecisionEnsemble
    featurenames = MLJ.report(m).features
    classlabels  = m.fitresult[2][sortperm((m).fitresult[3])]
    solem        = solemodel(MLJ.fitted_params(m).forest; featurenames, classlabels, tiebreaker=:alphanumeric)
    logiset      = scalarlogiset(X, allow_propositional=true)
    apply!(solem, logiset, y)
    return solem
end

function apply(
    m :: Machine{<:MLJ.MLJTuning.EitherTunedModel{<:Any, <:RandomForestClassifier}},
    X :: AbstractDataFrame,
    y :: AbstractVector
)::DecisionEnsemble
    featurenames = MLJ.report(m).best_report.features
    classlabels  = m.fitresult.fitresult[2][sortperm((m).fitresult.fitresult[3])]
    solem        = solemodel(MLJ.fitted_params(m).best_fitted_params.forest; featurenames, classlabels)
    logiset      = scalarlogiset(X, allow_propositional=true)
    apply!(solem, logiset, y)
    return solem
end

function apply(
    m :: Machine{<:EitherBalancedModel{<:Any, <:RandomForestClassifier}},
    X :: AbstractDataFrame,
    y :: AbstractVector
)::DecisionEnsemble
    featurenames = MLJ.report(m).model.features
    bm           = m.fitresult.interface.predict.machine
    classlabels  = bm.fitresult[2][sortperm((bm).fitresult[3])]
    solem        = solemodel(MLJ.fitted_params(m).model.forest; featurenames, classlabels, tiebreaker=:alphanumeric)
    logiset      = scalarlogiset(X, allow_propositional=true)
    apply!(solem, logiset, y)
    return solem
end

function apply(
    m :: Machine{<:MLJ.MLJTuning.EitherTunedModel{<:Any, <:EitherBalancedModel{<:Any, <:RandomForestClassifier}}},
    X :: AbstractDataFrame,
    y :: AbstractVector
)::DecisionEnsemble
    featurenames = MLJ.report(m).best_report.model.features
    bm           = m.fitresult.fitresult.interface.predict.machine
    classlabels  = bm.fitresult[2][sortperm((bm).fitresult[3])]
    solem        = solemodel(MLJ.fitted_params(m).best_fitted_params.model.forest; featurenames, classlabels)
    logiset      = scalarlogiset(X, allow_propositional=true)
    apply!(solem, logiset, y)
    return solem
end

# ---------------------------------------------------------------------------- #

function apply(
    m :: Machine{DecisionTreeRegressor},
    X :: AbstractDataFrame,
    y :: AbstractVector
)::DecisionTree
    featurenames = MLJ.report(m).features
    solem        = solemodel(MLJ.fitted_params(m).tree; featurenames)
    logiset      = scalarlogiset(X, allow_propositional=true)
    apply!(solem, logiset, y)
    return solem
end

function apply(
    m :: Machine{<:MLJ.MLJTuning.EitherTunedModel{<:Any, <:DecisionTreeRegressor}},
    X :: AbstractDataFrame,
    y :: AbstractVector
)::DecisionTree
    featurenames = MLJ.report(m).best_report.features
    solem        = solemodel(MLJ.fitted_params(m).best_fitted_params.tree; featurenames)
    logiset      = scalarlogiset(X, allow_propositional=true)
    apply!(solem, logiset, y)
    return solem
end

# ---------------------------------------------------------------------------- #

function apply(
    m :: Machine{RandomForestRegressor},
    X :: AbstractDataFrame,
    y :: AbstractVector
)::DecisionEnsemble
    featurenames = MLJ.report(m).features
    solem        = solemodel(MLJ.fitted_params(m).forest; featurenames)
    logiset      = scalarlogiset(X, allow_propositional=true)
    apply!(solem, logiset, y)
    return solem
end

function apply(
    m :: Machine{<:MLJ.MLJTuning.EitherTunedModel{<:Any, <:RandomForestRegressor}},
    X :: AbstractDataFrame,
    y :: AbstractVector
)::DecisionEnsemble
    featurenames = MLJ.report(m).best_report.features
    solem        = solemodel(MLJ.fitted_params(m).best_fitted_params.forest; featurenames)
    logiset      = scalarlogiset(X, allow_propositional=true)
    apply!(solem, logiset, y)
    return solem
end

# ---------------------------------------------------------------------------- #

function apply(
    m :: Machine{AdaBoostStumpClassifier},
    X :: AbstractDataFrame,
    y :: AbstractVector
)::DecisionEnsemble
    featurenames = MLJ.report(m).features
    classlabels  = sort(string.(m.fitresult[3]))
    weights      = m.fitresult[2]
    solem        = solemodel(MLJ.fitted_params(m).stumps; featurenames, classlabels, weights)
    logiset      = scalarlogiset(X, allow_propositional=true)
    apply!(solem, logiset, y)
    return solem
end

function apply(
    m :: Machine{<:MLJ.MLJTuning.EitherTunedModel{<:Any, <:AdaBoostStumpClassifier}},
    X :: AbstractDataFrame,
    y :: AbstractVector
)::DecisionEnsemble
    featurenames = MLJ.report(m).best_report.features
    classlabels  = sort(m.fitresult.fitresult[3])
    weights      = m.fitresult.fitresult[2]
    solem        = solemodel(MLJ.fitted_params(m).best_fitted_params.stumps; featurenames, classlabels, weights)
    logiset      = scalarlogiset(X, allow_propositional=true)
    apply!(solem, logiset, y)
    return solem
end

function apply(
    m :: Machine{<:EitherBalancedModel{<:Any, <:AdaBoostStumpClassifier}},
    X :: AbstractDataFrame,
    y :: AbstractVector
)::DecisionEnsemble
    featurenames = MLJ.report(m).model.features
    bm = m.fitresult.interface.predict.machine
    classlabels  = sort(string.(bm.fitresult[3]))
    weights      = bm.fitresult[2]
    solem        = solemodel(MLJ.fitted_params(m).model.stumps; featurenames, classlabels, weights)
    logiset      = scalarlogiset(X, allow_propositional=true)
    apply!(solem, logiset, y)
    return solem
end

function apply(
    m :: Machine{<:MLJ.MLJTuning.EitherTunedModel{<:Any, <:EitherBalancedModel{<:Any, <:AdaBoostStumpClassifier}}},
    X :: AbstractDataFrame,
    y :: AbstractVector
)::DecisionEnsemble
    featurenames = MLJ.report(m).best_report.model.features
    bm           = m.fitresult.fitresult.interface.predict.machine
    classlabels  = sort(bm.fitresult[3])
    weights      = bm.fitresult[2]
    solem        = solemodel(MLJ.fitted_params(m).best_fitted_params.model.stumps; featurenames, classlabels, weights)
    logiset      = scalarlogiset(X, allow_propositional=true)
    apply!(solem, logiset, y)
    return solem
end

# ---------------------------------------------------------------------------- #
#                           ModalDecisionTrees package                         #
# ---------------------------------------------------------------------------- #
function apply(
    m :: ModalApply,
    X :: AbstractDataFrame,
    y :: AbstractVector
)::Union{DecisionTree, DecisionEnsemble}
    (_, solem) = MLJ.report(m).sprinkle(X, y)
    return solem
end

function apply(
    m :: TunedModalApply,
    X :: AbstractDataFrame,
    y :: AbstractVector
)::Union{DecisionTree, DecisionEnsemble}
    (_, solem) = MLJ.report(m).best_report.sprinkle(X, y)
    return solem
end

function apply(
    m :: BalancedModalApply,
    X :: AbstractDataFrame,
    y :: AbstractVector
)::Union{DecisionTree, DecisionEnsemble}
    (_, solem) = MLJ.report(m).model.sprinkle(X, y)
    return solem
end

function apply(
    m :: TunedBalancedModalApply,
    X :: AbstractDataFrame,
    y :: AbstractVector
)::Union{DecisionTree, DecisionEnsemble}
    (_, solem) = MLJ.report(m).best_report.model.sprinkle(X, y)
    return solem
end

# ---------------------------------------------------------------------------- #
#                                XGBoost package                               #
# ---------------------------------------------------------------------------- #
function apply(
    m :: Machine{XGBoostClassifier},
    X :: AbstractDataFrame,
    y :: AbstractVector
)::DecisionXGBoost
    trees        = XGBoost.trees(m.fitresult[1])
    encoding     = get_encoding(m.fitresult[2])
    featurenames = m.report.vals[1].features
    classlabels  = get_classlabels(encoding)
    solem        = solemodel(trees, Matrix(X), y; featurenames, classlabels)
    logiset      = scalarlogiset(mapcols(col -> Float32.(col), X), allow_propositional=true)
    apply!(solem, logiset, y)
    return solem
end

function apply(
    m :: Machine{<:MLJ.MLJTuning.EitherTunedModel{<:Any, <:XGBoostClassifier}},
    X :: AbstractDataFrame,
    y :: AbstractVector
)::DecisionXGBoost
    trees        = XGBoost.trees(m.fitresult.fitresult[1])
    encoding     = get_encoding(m.fitresult.fitresult[2])
    featurenames = m.fitresult.report.vals[1].features
    classlabels  = get_classlabels(encoding)
    solem        = solemodel(trees, Matrix(X), y; featurenames, classlabels)
    logiset      = scalarlogiset(mapcols(col -> Float32.(col), X), allow_propositional=true)
    apply!(solem, logiset, y)
    return solem
end

function apply(
    m :: Machine{<:EitherBalancedModel{<:Any, <:XGBoostClassifier}},
    X :: AbstractDataFrame,
    y :: AbstractVector
)::DecisionXGBoost
    bm           = m.fitresult.interface.predict.machine
    trees        = XGBoost.trees(bm.fitresult[1])
    encoding     = get_encoding(bm.fitresult[2])
    featurenames = m.report.vals[1].model.features
    classlabels  = get_classlabels(encoding)
    solem        = solemodel(trees, Matrix(X), y; featurenames, classlabels)
    logiset      = scalarlogiset(mapcols(col -> Float32.(col), X), allow_propositional=true)
    apply!(solem, logiset, y)
    return solem
end

function apply(
    m :: Machine{<:MLJ.MLJTuning.EitherTunedModel{<:Any, <:EitherBalancedModel{<:Any, <:XGBoostClassifier}}},
    X :: AbstractDataFrame,
    y :: AbstractVector
)::DecisionXGBoost
    bm           = m.fitresult.fitresult.interface.predict.machine
    trees        = XGBoost.trees(bm.fitresult[1])
    encoding     = get_encoding(bm.fitresult[2])
    featurenames = bm.report.vals[1].features
    classlabels  = get_classlabels(encoding)
    solem        = solemodel(trees, Matrix(X), y; featurenames, classlabels)
    logiset      = scalarlogiset(mapcols(col -> Float32.(col), X), allow_propositional=true)
    apply!(solem, logiset, y)
    return solem
end

# ---------------------------------------------------------------------------- #

function apply(
    m :: Machine{XGBoostRegressor},
    X :: AbstractDataFrame,
    y :: AbstractVector
)::DecisionXGBoost
    base_score = get_base_score(m) == -Inf ? mean(m.y[train]) : 0.5
    m.model.base_score = base_score

    trees        = XGBoost.trees(m.fitresult[1])
    featurenames = m.report.vals[1].features
    solem        = solemodel(trees, Matrix(X), y; featurenames)
    logiset      = scalarlogiset(mapcols(col -> Float32.(col), X), allow_propositional=true)
    apply!(solem, logiset, y; base_score)
    return solem
end

function apply(
    m :: Machine{<:MLJ.MLJTuning.EitherTunedModel{<:Any, <:XGBoostRegressor}},
    X :: AbstractDataFrame,
    y :: AbstractVector,
)::DecisionXGBoost
    base_score = get_base_score(m) == -Inf ? mean(m.y[train]) : 0.5
    m.model.model.base_score = base_score

    trees        = XGBoost.trees(m.fitresult.fitresult[1])
    featurenames = m.fitresult.report.vals[1].features
    solem        = solemodel(trees, Matrix(X), y; featurenames)
    logiset      = scalarlogiset(mapcols(col -> Float32.(col), X), allow_propositional=true)
    apply!(solem, logiset, y; base_score)
    return solem
end