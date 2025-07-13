# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
const Mach  = MLJ.Machine

TunedMach(T) = Union{
    PropositionalDataSet{<:MLJ.MLJTuning.EitherTunedModel{<:Any, <:T}},
    ModalDataSet{<:MLJ.MLJTuning.EitherTunedModel{<:Any, <:T}},
}

const DecisionTreeApply = Union{
    PropositionalDataSet{DecisionTreeClassifier}, 
    PropositionalDataSet{DecisionTreeRegressor},
}

const TunedDecisionTreeApply = Union{
    TunedMach(DecisionTreeClassifier),
    TunedMach(DecisionTreeRegressor)
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
function get_base_score(m::MLJ.Machine)
    if m.model isa MLJ.MLJTuning.EitherTunedModel
        return hasproperty(m.model.model, :base_score) ? m.model.model.base_score : nothing
    else
        return hasproperty(m.model, :base_score) ? m.model.base_score : nothing
    end
end
get_encoding(classes_seen) = Dict(MLJ.int(c) => c for c in MLJ.classes(classes_seen))
get_classlabels(encoding)  = [string(encoding[i]) for i in sort(keys(encoding) |> collect)]

# ---------------------------------------------------------------------------- #
#                             DecisionTree package                             #
# ---------------------------------------------------------------------------- #
function apply(
    model  :: DecisionTreeApply,
    X      :: AbstractDataFrame,
    y      :: AbstractVector
)
    solem = solemodel(MLJ.fitted_params(model.mach).tree)
    apply!(solem, X, y)
    return solem
end

function apply(
    model  :: TunedDecisionTreeApply,
    X      :: AbstractDataFrame,
    y      :: AbstractVector
)
    solem = solemodel(MLJ.fitted_params(model.mach).best_fitted_params.tree)
    apply!(solem, X, y)
    return solem
end

# randomforest
function apply(
    model  :: PropositionalDataSet{RandomForestClassifier},
    X      :: AbstractDataFrame,
    y      :: AbstractVector
)
    classlabels  = string.(model.mach.fitresult[2][sortperm((model.mach).fitresult[3])])
    featurenames = MLJ.report(model.mach).features
    solem        = solemodel(MLJ.fitted_params(model.mach).forest; classlabels, featurenames)
    apply!(solem, X, y)
    return solem
end

function apply(
    model  :: TunedMach(RandomForestClassifier),
    X      :: AbstractDataFrame,
    y      :: AbstractVector
)
    classlabels  = string.(model.mach.fitresult.fitresult[2][sortperm((model.mach).fitresult.fitresult[3])])
    featurenames = MLJ.report(model.mach).best_report.features
    solem        = solemodel(MLJ.fitted_params(model.mach).best_fitted_params.forest; classlabels, featurenames)
    apply!(solem, X, y)
    return solem
end

function apply(
    model  :: PropositionalDataSet{RandomForestRegressor},
    X      :: AbstractDataFrame,
    y      :: AbstractVector
)
    featurenames = MLJ.report(model.mach).features
    solem        = solemodel(MLJ.fitted_params(model.mach).forest; featurenames)
    apply!(solem, X, y)
    return solem
end

function apply(
    model  :: TunedMach(RandomForestRegressor),
    X      :: AbstractDataFrame,
    y      :: AbstractVector
)
    featurenames = MLJ.report(model.mach).best_report.features
    solem        = solemodel(MLJ.fitted_params(model.mach).best_fitted_params.forest; featurenames)
    apply!(solem, X, y)
    return solem
end

# adaboost
function apply(
    model  :: PropositionalDataSet{AdaBoostStumpClassifier},
    X      :: AbstractDataFrame,
    y      :: AbstractVector
)
    weights      = model.mach.fitresult[2]
    classlabels  = sort(string.(model.mach.fitresult[3]))
    featurenames = MLJ.report(model.mach).features
    solem        = solemodel(MLJ.fitted_params(model.mach).stumps; weights, classlabels, featurenames)
    apply!(solem, X, y)
    return solem
end

function apply(
    model  :: TunedMach(AdaBoostStumpClassifier),
    X      :: AbstractDataFrame,
    y      :: AbstractVector
)
    weights      = model.mach.fitresult.fitresult[2]
    classlabels  = sort(string.(model.mach.fitresult.fitresult[3]))
    featurenames = MLJ.report(model.mach).best_report.features
    solem        = solemodel(MLJ.fitted_params(model.mach).best_fitted_params.stumps; weights, classlabels, featurenames)
    apply!(solem, X, y)
    return solem
end

# ---------------------------------------------------------------------------- #
#                           ModalDecisionTrees package                         #
# ---------------------------------------------------------------------------- #
function apply(
    model  :: ModalDecisionTreeApply,
    X      :: AbstractDataFrame,
    y      :: AbstractVector
)
    (_, solem) = MLJ.report(model.mach).sprinkle(X, y)
    return solem
end

function apply(
    model  :: TunedModalDecisionTreeApply,
    X      :: AbstractDataFrame,
    y      :: AbstractVector
)
    (_, solem) = MLJ.report(model.mach).best_report.sprinkle(X, y)
    return solem
end

# ---------------------------------------------------------------------------- #
#                                XGBoost package                               #
# ---------------------------------------------------------------------------- #
function apply(
    model  :: PropositionalDataSet{XGBoostClassifier},
    X      :: AbstractDataFrame,
    y      :: AbstractVector
)
    trees        = XGBoost.trees(model.mach.fitresult[1])
    encoding     = get_encoding(model.mach.fitresult[2])
    classlabels  = string.(get_classlabels(encoding))
    featurenames = model.mach.report.vals[1].features
    solem        = solemodel(trees, Matrix(X), y; classlabels, featurenames)
    apply!(solem, mapcols(col -> Float32.(col), X), y)
    return solem
end

function apply(
    model  :: TunedMach(XGBoostClassifier),
    X      :: AbstractDataFrame,
    y      :: AbstractVector
)
    trees        = XGBoost.trees(model.mach.fitresult.fitresult[1])
    encoding     = get_encoding(model.mach.fitresult.fitresult[2])
    classlabels  = string.(get_classlabels(encoding))
    featurenames = model.mach.fitresult.report.vals[1].features
    solem        = solemodel(trees, Matrix(X), y; classlabels, featurenames)
    apply!(solem, mapcols(col -> Float32.(col), X), y)
    return solem
end

function apply(
    model   :: PropositionalDataSet{XGBoostRegressor},
    X       :: AbstractDataFrame,
    y       :: AbstractVector
)
    base_score = get_base_score(model.mach) == -Inf ? mean(ds.y[train]) : 0.5
    model.mach.model.base_score = base_score

    trees        = XGBoost.trees(model.mach.fitresult[1])
    featurenames = model.mach.report.vals[1].features
    solem        = solemodel(trees, Matrix(X), y; featurenames)
    apply!(solem, mapcols(col -> Float32.(col), X), y; base_score)
    return solem
end

function apply(
    model   :: TunedMach(XGBoostRegressor),
    X       :: AbstractDataFrame,
    y       :: AbstractVector,
)
    base_score = get_base_score(model.mach) == -Inf ? mean(ds.y[train]) : 0.5
    model.mach.model.model.base_score = base_score

    trees        = XGBoost.trees(model.mach.fitresult.fitresult[1])
    featurenames = model.mach.fitresult.report.vals[1].features
    solem        = solemodel(trees, Matrix(X), y; featurenames)
    apply!(solem, mapcols(col -> Float32.(col), X), y; base_score)
    return solem
end