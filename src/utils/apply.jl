# ---------------------------------------------------------------------------- #
#                             DecisionTree package                             #
# ---------------------------------------------------------------------------- #
const DecisionTreeApply = Union{
    MLJ.Machine{<:MLJDecisionTreeInterface.DecisionTreeClassifier,<:Any,true},
    MLJ.Machine{<:MLJDecisionTreeInterface.DecisionTreeRegressor, <:Any,true}
}

function apply(
    mach   :: DecisionTreeApply,
    tuning :: Bool,
    X      :: AbstractDataFrame,
    y      :: AbstractVector
)
    tuning === false ? begin
        solem = solemodel(MLJ.fitted_params(mach).tree)
        apply!(solem, X, y)
    end : begin
        solem = solemodel(MLJ.fitted_params(mach).best_fitted_params.tree)
        apply!(solem, X, y)
    end

    return solem
end

# randomforest
function apply(
    mach   :: MLJ.Machine{<:MLJDecisionTreeInterface.RandomForestClassifier,<:Any,true},
    tuning :: Bool,
    X      :: AbstractDataFrame,
    y      :: AbstractVector
)
    tuning === false ? begin
        classlabels  = string.(mach.fitresult[2][sortperm((mach).fitresult[3])])
        featurenames = MLJ.report(mach).features
        solem        = solemodel(MLJ.fitted_params(mach).forest; classlabels, featurenames)
        apply!(solem, X, y)
    end : begin
        classlabels  = string.(mach.fitresult.fitresult[2][sortperm((mach).fitresult.fitresult[3])])
        featurenames = MLJ.report(mach).best_report.features
        solem        = solemodel(MLJ.fitted_params(mach).best_fitted_params.forest; classlabels, featurenames)
        apply!(solem, X, y)
    end

    return solem
end

function apply(
    mach   :: MLJ.Machine{<:MLJDecisionTreeInterface.RandomForestRegressor,<:Any,true},
    tuning :: Bool,
    X      :: AbstractDataFrame,
    y      :: AbstractVector
)
    tuning === false ? begin
        featurenames = MLJ.report(mach).features
        solem        = solemodel(MLJ.fitted_params(mach).forest; featurenames)
        apply!(solem, X, y)
    end : begin
        featurenames = MLJ.report(mach).best_report.features
        solem        = solemodel(MLJ.fitted_params(mach).best_fitted_params.forest; featurenames)
        apply!(solem, X, y)
    end

    return solem
end

# adaboost
function apply(
    mach   :: MLJ.Machine{<:MLJDecisionTreeInterface.AdaBoostStumpClassifier,<:Any,true},
    tuning :: Bool,
    X      :: AbstractDataFrame,
    y      :: AbstractVector
)
    tuning === false ? begin
        weights      = mach.fitresult[2]
        classlabels  = sort(string.(mach.fitresult[3]))
        featurenames = MLJ.report(mach).features
        solem        = solemodel(MLJ.fitted_params(mach).stumps; weights, classlabels, featurenames)
        apply!(solem, X, y)
    end : begin
        weights      = mach.fitresult.fitresult[2]
        classlabels  = sort(string.(mach.fitresult.fitresult[3]))
        featurenames = MLJ.report(mach).best_report.features
        solem        = solemodel(MLJ.fitted_params(mach).best_fitted_params.stumps; weights, classlabels, featurenames)
        apply!(solem, X, y)
    end

    return solem
end

# ---------------------------------------------------------------------------- #
#                           ModalDecisionTrees package                         #
# ---------------------------------------------------------------------------- #
const ModalDecisionTreeApply = Union{
    MLJ.Machine{<:ModalDecisionTrees.ModalDecisionTree,<:Any,true},
    MLJ.Machine{<:ModalDecisionTrees.ModalRandomForest, <:Any,true},
    MLJ.Machine{<:ModalDecisionTrees.ModalAdaBoost, <:Any,true}
}

function apply(
    mach   :: ModalDecisionTreeApply,
    tuning :: Bool,
    X      :: AbstractDataFrame,
    y      :: AbstractVector
)
    tuning === false ? begin
        (_, solem) = MLJ.report(mach).sprinkle(X, y)
    end : begin
        (_, solem) = MLJ.report(mach).best_report.sprinkle(X, y)
    end

    return solem
end

# ---------------------------------------------------------------------------- #
#                                XGBoost package                               #
# ---------------------------------------------------------------------------- #
function apply(
    mach   :: MLJ.Machine{<:MLJXGBoostInterface.XGBoostClassifier,<:Any,true},
    tuning :: Bool,
    X      :: AbstractDataFrame,
    y      :: AbstractVector
)
    tuning === false ? begin
        trees        = XGB.trees(mach.fitresult[1])
        encoding     = get_encoding(mach.fitresult[2])
        classlabels  = string.(get_classlabels(encoding))
        featurenames = mach.report.vals[1].features
        solem        = solemodel(trees, Matrix(X), y; classlabels, featurenames)
        apply!(solem, mapcols(col -> Float32.(col), X), y)
    end : begin
        trees        = XGB.trees(mach.fitresult.fitresult[1])
        encoding     = get_encoding(mach.fitresult.fitresult[2])
        classlabels  = string.(get_classlabels(encoding))
        featurenames = mach.fitresult.report.vals[1].features
        solem        = solemodel(trees, Matrix(X), y; classlabels, featurenames)
        apply!(solem, mapcols(col -> Float32.(col), X), y)
    end

    return solem
end

function apply(
    mach    :: MLJ.Machine{<:MLJXGBoostInterface.XGBoostRegressor,<:Any,true},
    tuning  :: Bool,
    X       :: AbstractDataFrame,
    y       :: AbstractVector,
    bs      :: AbstractFloat
)
    tuning === false ? begin
        trees        = XGB.trees(mach.fitresult[1])
        featurenames = mach.report.vals[1].features
        solem        = solemodel(trees, Matrix(X), y; featurenames)
        apply!(solem, mapcols(col -> Float32.(col), X), y; base_score=bs)
    end : begin
        trees        = XGB.trees(mach.fitresult.fitresult[1])
        featurenames = mach.fitresult.report.vals[1].features
        solem        = solemodel(trees, Matrix(X), y; featurenames)
        apply!(solem, mapcols(col -> Float32.(col), X), y; base_score=bs)
    end

    return solem
end