"""
apply.jl - Unified Model Conversion Interface

This module provides `apply(ds, X, y)` methods that:
1. Extract fitted models from MLJ machine wrappers
2. Convert them to SoleModels symbolic representations
3. Create logical datasets and apply symbolic models
4. Return symbolic models ready for rule extraction and analysis

Supported ML packages:
- DecisionTree.jl (trees, random forests, AdaBoost)
- ModalDecisionTrees.jl (modal trees, forests, boosting)
- XGBoost.jl (classifiers and regressors)

All methods handle both regular and hyperparameter-tuned models.
"""

# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
const Mach  = MLJ.Machine

TunedMach(T) = Union{
    PropositionalDataSet{<:MLJTuning.EitherTunedModel{<:Any, <:T}},
    ModalDataSet{<:MLJTuning.EitherTunedModel{<:Any, <:T}},
}

const ModalDecisionTreeApply = Union{
    ModalDataSet{ModalDecisionTree},
    ModalDataSet{ModalRandomForest},
    ModalDataSet{ModalAdaBoost}
}

const TunedModalDecisionTreeApply = Union{
    TunedMach(ModalDecisionTree),
    TunedMach(ModalRandomForest),
    TunedMach(ModalAdaBoost)
}

# ---------------------------------------------------------------------------- #
#                              xgboost utilities                               #
# ---------------------------------------------------------------------------- #
"""
    get_base_score(m::MLJ.Machine) -> Union{Number,Nothing}

Extract base_score from XGBoost models, handling both regular and tuned variants.
Returns `nothing` if the model doesn't have a base_score property.
"""
function get_base_score(m::MLJ.Machine)
    if m.model isa MLJTuning.EitherTunedModel
        return hasproperty(m.model.model, :base_score) ? m.model.model.base_score : nothing
    else
        return hasproperty(m.model, :base_score) ? m.model.base_score : nothing
    end
end

"""
    get_encoding(classes_seen) -> Dict

Create mapping from internal class indices to MLJ class labels.
Used for preserving class information in classification models.
"""
get_encoding(classes_seen) = Dict(MLJ.int(c) => c for c in MLJ.classes(classes_seen))

"""
    get_classlabels(encoding) -> Vector{String}

Extract ordered class labels from encoding dictionary.
Ensures consistent class ordering across model conversions.
"""
get_classlabels(encoding)  = [string(encoding[i]) for i in sort(keys(encoding) |> collect)]

# ---------------------------------------------------------------------------- #
#                             DecisionTree package                             #
# ---------------------------------------------------------------------------- #
function apply(
    ds :: PropositionalDataSet{DecisionTreeClassifier},
    X  :: AbstractDataFrame,
    y  :: AbstractVector
)
    featurenames = MLJ.report(ds.mach).features
    classlabels  = sort(MLJ.report(ds.mach).classes_seen)
    solem        = solemodel(MLJ.fitted_params(ds.mach).tree; featurenames, classlabels)
    logiset      = scalarlogiset(X, allow_propositional = true)
    apply!(solem, logiset, y)
    return solem
end

function apply(
    ds :: TunedMach(DecisionTreeClassifier),
    X  :: AbstractDataFrame,
    y  :: AbstractVector
)
    featurenames = MLJ.report(ds.mach).best_report.features
    classlabels  = sort(MLJ.report(ds.mach).best_report.classes_seen)
    solem        = solemodel(MLJ.fitted_params(ds.mach).best_fitted_params.tree; featurenames, classlabels)
    logiset      = scalarlogiset(X, allow_propositional = true)
    apply!(solem, logiset, y)
    return solem
end

# randomforest
function apply(
    ds :: PropositionalDataSet{RandomForestClassifier},
    X  :: AbstractDataFrame,
    y  :: AbstractVector
)
    featurenames = MLJ.report(ds.mach).features
    classlabels  = ds.mach.fitresult[2][sortperm((ds.mach).fitresult[3])]
    solem        = solemodel(MLJ.fitted_params(ds.mach).forest; featurenames, classlabels)
    logiset      = scalarlogiset(X, allow_propositional = true)
    apply!(solem, logiset, y)
    return solem
end

function apply(
    ds :: TunedMach(RandomForestClassifier),
    X  :: AbstractDataFrame,
    y  :: AbstractVector
)
    featurenames = MLJ.report(ds.mach).best_report.features
    classlabels  = ds.mach.fitresult.fitresult[2][sortperm((ds.mach).fitresult.fitresult[3])]
    solem        = solemodel(MLJ.fitted_params(ds.mach).best_fitted_params.forest; featurenames, classlabels)
    logiset      = scalarlogiset(X, allow_propositional = true)
    apply!(solem, logiset, y)
    return solem
end

function apply(
    ds :: PropositionalDataSet{DecisionTreeRegressor},
    X  :: AbstractDataFrame,
    y  :: AbstractVector
)
    featurenames = MLJ.report(ds.mach).features
    solem        = solemodel(MLJ.fitted_params(ds.mach).tree; featurenames)
    logiset      = scalarlogiset(X, allow_propositional = true)
    apply!(solem, logiset, y)
    return solem
end

function apply(
    ds :: TunedMach(DecisionTreeRegressor),
    X  :: AbstractDataFrame,
    y  :: AbstractVector
)
    featurenames = MLJ.report(ds.mach).best_report.features
    solem        = solemodel(MLJ.fitted_params(ds.mach).best_fitted_params.tree; featurenames)
    logiset      = scalarlogiset(X, allow_propositional = true)
    apply!(solem, logiset, y)
    return solem
end

function apply(
    ds :: PropositionalDataSet{RandomForestRegressor},
    X  :: AbstractDataFrame,
    y  :: AbstractVector
)
    featurenames = MLJ.report(ds.mach).features
    solem        = solemodel(MLJ.fitted_params(ds.mach).forest; featurenames)
    logiset      = scalarlogiset(X, allow_propositional = true)
    apply!(solem, logiset, y)
    return solem
end

function apply(
    ds :: TunedMach(RandomForestRegressor),
    X  :: AbstractDataFrame,
    y  :: AbstractVector
)
    featurenames = MLJ.report(ds.mach).best_report.features
    solem        = solemodel(MLJ.fitted_params(ds.mach).best_fitted_params.forest; featurenames)
    logiset      = scalarlogiset(X, allow_propositional = true)
    apply!(solem, logiset, y)
    return solem
end

# adaboost
function apply(
    ds :: PropositionalDataSet{AdaBoostStumpClassifier},
    X  :: AbstractDataFrame,
    y  :: AbstractVector
)
    featurenames = MLJ.report(ds.mach).features
    classlabels  = sort(string.(ds.mach.fitresult[3]))
    weights      = ds.mach.fitresult[2]
    solem        = solemodel(MLJ.fitted_params(ds.mach).stumps; featurenames, classlabels, weights)
    logiset      = scalarlogiset(X, allow_propositional = true)
    apply!(solem, logiset, y)
    return solem
end

function apply(
    ds :: TunedMach(AdaBoostStumpClassifier),
    X  :: AbstractDataFrame,
    y  :: AbstractVector
)
    featurenames = MLJ.report(ds.mach).best_report.features
    classlabels  = sort(ds.mach.fitresult.fitresult[3])
    weights      = ds.mach.fitresult.fitresult[2]
    solem        = solemodel(MLJ.fitted_params(ds.mach).best_fitted_params.stumps; featurenames, classlabels, weights)
    logiset      = scalarlogiset(X, allow_propositional = true)
    apply!(solem, logiset, y)
    return solem
end

# ---------------------------------------------------------------------------- #
#                           ModalDecisionTrees package                         #
# ---------------------------------------------------------------------------- #
function apply(
    ds :: ModalDecisionTreeApply,
    X  :: AbstractDataFrame,
    y  :: AbstractVector
)
    (_, solem) = MLJ.report(ds.mach).sprinkle(X, y)
    return solem
end

function apply(
    ds :: TunedModalDecisionTreeApply,
    X  :: AbstractDataFrame,
    y  :: AbstractVector
)
    (_, solem) = MLJ.report(ds.mach).best_report.sprinkle(X, y)
    return solem
end

# ---------------------------------------------------------------------------- #
#                                XGBoost package                               #
# ---------------------------------------------------------------------------- #
function apply(
    ds :: PropositionalDataSet{XGBoostClassifier},
    X  :: AbstractDataFrame,
    y  :: AbstractVector
)
    trees        = XGBoost.trees(ds.mach.fitresult[1])
    encoding     = get_encoding(ds.mach.fitresult[2])
    featurenames = ds.mach.report.vals[1].features
    classlabels  = get_classlabels(encoding)
    solem        = solemodel(trees, Matrix(X), y; featurenames, classlabels)
    logiset      = scalarlogiset(mapcols(col -> Float32.(col), X), allow_propositional = true)
    apply!(solem, logiset, y)
    return solem
end

function apply(
    ds :: TunedMach(XGBoostClassifier),
    X  :: AbstractDataFrame,
    y  :: AbstractVector
)
    trees        = XGBoost.trees(ds.mach.fitresult.fitresult[1])
    encoding     = get_encoding(ds.mach.fitresult.fitresult[2])
    featurenames = ds.mach.fitresult.report.vals[1].features
    classlabels  = get_classlabels(encoding)
    solem        = solemodel(trees, Matrix(X), y; featurenames, classlabels)
    logiset      = scalarlogiset(mapcols(col -> Float32.(col), X), allow_propositional = true)
    apply!(solem, logiset, y)
    return solem
end

function apply(
    ds :: PropositionalDataSet{XGBoostRegressor},
    X  :: AbstractDataFrame,
    y  :: AbstractVector
)
    base_score = get_base_score(ds.mach) == -Inf ? mean(ds.y[train]) : 0.5
    ds.mach.model.base_score = base_score

    trees        = XGBoost.trees(ds.mach.fitresult[1])
    featurenames = ds.mach.report.vals[1].features
    solem        = solemodel(trees, Matrix(X), y; featurenames)
    logiset      = scalarlogiset(mapcols(col -> Float32.(col), X), allow_propositional = true)
    apply!(solem, logiset, y; base_score)
    return solem
end

function apply(
    ds :: TunedMach(XGBoostRegressor),
    X  :: AbstractDataFrame,
    y  :: AbstractVector,
)
    base_score = get_base_score(ds.mach) == -Inf ? mean(ds.y[train]) : 0.5
    ds.mach.model.model.base_score = base_score

    trees        = XGBoost.trees(ds.mach.fitresult.fitresult[1])
    featurenames = ds.mach.fitresult.report.vals[1].features
    solem        = solemodel(trees, Matrix(X), y; featurenames)
    logiset      = scalarlogiset(mapcols(col -> Float32.(col), X), allow_propositional = true)
    apply!(solem, logiset, y; base_score)
    return solem
end