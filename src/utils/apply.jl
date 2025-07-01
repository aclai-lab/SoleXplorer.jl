const Mach  = MLJ.Machine

TunedMach(T) = Union{
    Mach{<:MLJ.MLJTuning.ProbabilisticTunedModel{<:Any, <:T}},
    Mach{<:MLJ.MLJTuning.DeterministicTunedModel{<:Any, <:T}}
}

# ---------------------------------------------------------------------------- #
#                             DecisionTree package                             #
# ---------------------------------------------------------------------------- #
const DecisionTreeApply = Union{
    Mach{<:DecisionTreeClassifier}, 
    Mach{<:DecisionTreeRegressor}
}

const TunedDecisionTreeApply = Union{
    TunedMach(DecisionTreeClassifier),
    TunedMach(DecisionTreeRegressor)
}

function apply(
    mach   :: DecisionTreeApply,
    X      :: AbstractDataFrame,
    y      :: AbstractVector
)
    solem = solemodel(MLJ.fitted_params(mach).tree)
    apply!(solem, X, y)
    return solem
end

function apply(
    mach   :: TunedDecisionTreeApply,
    X      :: AbstractDataFrame,
    y      :: AbstractVector
)
    solem = solemodel(MLJ.fitted_params(mach).best_fitted_params.tree)
    apply!(solem, X, y)
    return solem
end

# randomforest
function apply(
    mach   :: Mach{<:RandomForestClassifier},
    X      :: AbstractDataFrame,
    y      :: AbstractVector
)
    classlabels  = string.(mach.fitresult[2][sortperm((mach).fitresult[3])])
    featurenames = MLJ.report(mach).features
    solem        = solemodel(MLJ.fitted_params(mach).forest; classlabels, featurenames)
    apply!(solem, X, y)
    return solem
end

function apply(
    mach   :: TunedMach(RandomForestClassifier),
    X      :: AbstractDataFrame,
    y      :: AbstractVector
)
    classlabels  = string.(mach.fitresult.fitresult[2][sortperm((mach).fitresult.fitresult[3])])
    featurenames = MLJ.report(mach).best_report.features
    solem        = solemodel(MLJ.fitted_params(mach).best_fitted_params.forest; classlabels, featurenames)
    apply!(solem, X, y)
    return solem
end

function apply(
    mach   :: Mach{<:RandomForestRegressor},
    X      :: AbstractDataFrame,
    y      :: AbstractVector
)
    featurenames = MLJ.report(mach).features
    solem        = solemodel(MLJ.fitted_params(mach).forest; featurenames)
    apply!(solem, X, y)
    return solem
end

function apply(
    mach   :: TunedMach(RandomForestRegressor),
    X      :: AbstractDataFrame,
    y      :: AbstractVector
)
    featurenames = MLJ.report(mach).best_report.features
    solem        = solemodel(MLJ.fitted_params(mach).best_fitted_params.forest; featurenames)
    apply!(solem, X, y)
    return solem
end

# adaboost
function apply(
    mach   :: Mach{<:AdaBoostStumpClassifier},
    X      :: AbstractDataFrame,
    y      :: AbstractVector
)
    weights      = mach.fitresult[2]
    classlabels  = sort(string.(mach.fitresult[3]))
    featurenames = MLJ.report(mach).features
    solem        = solemodel(MLJ.fitted_params(mach).stumps; weights, classlabels, featurenames)
    apply!(solem, X, y)
    return solem
end

function apply(
    mach   :: TunedMach(AdaBoostStumpClassifier),
    X      :: AbstractDataFrame,
    y      :: AbstractVector
)
    weights      = mach.fitresult.fitresult[2]
    classlabels  = sort(string.(mach.fitresult.fitresult[3]))
    featurenames = MLJ.report(mach).best_report.features
    solem        = solemodel(MLJ.fitted_params(mach).best_fitted_params.stumps; weights, classlabels, featurenames)
    apply!(solem, X, y)
    return solem
end

# ---------------------------------------------------------------------------- #
#                           ModalDecisionTrees package                         #
# ---------------------------------------------------------------------------- #
const ModalDecisionTreeApply = Union{
    Mach{<:ModalDecisionTree},
    Mach{<:ModalRandomForest},
    Mach{<:ModalAdaBoost}
}

const TunedModalDecisionTreeApply = Union{
    TunedMach(ModalDecisionTree),
    TunedMach(ModalRandomForest),
    TunedMach(ModalAdaBoost)
}

function apply(
    mach   :: ModalDecisionTreeApply,
    X      :: AbstractDataFrame,
    y      :: AbstractVector
)
    (_, solem) = MLJ.report(mach).sprinkle(X, y)
    return solem
end

function apply(
    mach   :: TunedModalDecisionTreeApply,
    X      :: AbstractDataFrame,
    y      :: AbstractVector
)
    (_, solem) = MLJ.report(mach).best_report.sprinkle(X, y)
    return solem
end

# ---------------------------------------------------------------------------- #
#                                XGBoost package                               #
# ---------------------------------------------------------------------------- #
function apply(
    mach   :: Mach{<:XGBoostClassifier},
    X      :: AbstractDataFrame,
    y      :: AbstractVector
)
    trees        = XGB.trees(mach.fitresult[1])
    encoding     = get_encoding(mach.fitresult[2])
    classlabels  = string.(get_classlabels(encoding))
    featurenames = mach.report.vals[1].features
    solem        = solemodel(trees, Matrix(X), y; classlabels, featurenames)
    apply!(solem, mapcols(col -> Float32.(col), X), y)
    return solem
end

function apply(
    mach   :: TunedMach(XGBoostClassifier),
    X      :: AbstractDataFrame,
    y      :: AbstractVector
)
    trees        = XGB.trees(mach.fitresult.fitresult[1])
    encoding     = get_encoding(mach.fitresult.fitresult[2])
    classlabels  = string.(get_classlabels(encoding))
    featurenames = mach.fitresult.report.vals[1].features
    solem        = solemodel(trees, Matrix(X), y; classlabels, featurenames)
    apply!(solem, mapcols(col -> Float32.(col), X), y)
    return solem
end

function apply(
    mach    :: Mach{<:XGBoostRegressor},
    X       :: AbstractDataFrame,
    y       :: AbstractVector,
    bs      :: AbstractFloat
)
    trees        = XGB.trees(mach.fitresult[1])
    featurenames = mach.report.vals[1].features
    solem        = solemodel(trees, Matrix(X), y; featurenames)
    apply!(solem, mapcols(col -> Float32.(col), X), y; base_score=bs)
    return solem
end

function apply(
    mach    :: TunedMach(XGBoostRegressor),
    X       :: AbstractDataFrame,
    y       :: AbstractVector,
    bs      :: AbstractFloat
)
    trees        = XGB.trees(mach.fitresult.fitresult[1])
    featurenames = mach.fitresult.report.vals[1].features
    solem        = solemodel(trees, Matrix(X), y; featurenames)
    apply!(solem, mapcols(col -> Float32.(col), X), y; base_score=bs)
    return solem
end