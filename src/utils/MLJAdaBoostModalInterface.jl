module MLJAdaBoostModalInterface

import MLJModelInterface as MMI
import ModalDecisionTrees as MDT
import DecisionTree as DT

using SoleData
import SoleData: TestOperator
using ScientificTypes

import Tables
using CategoricalArrays

using Random
import Random.GLOBAL_RNG

const SymbolicModel = AdaBoostModalClassifier

const PKG = "MLJAdaBoostModalInterface"

export AdaBoostModalClassifier

# ---------------------------------------------------------------------------- #
#                          modal adaboost tree struct                          #
# ---------------------------------------------------------------------------- #
abstract type AbstractAdaBoostModalClassifier <: MMI.Probabilistic end

mutable struct AdaBoostModalClassifier <: MMI.Probabilistic
    ## Pruning conditions
    max_depth               ::Union{Nothing,Int}
    min_samples_leaf        ::Union{Nothing,Int}
    min_purity_increase     ::Union{Nothing,Float64}
    max_purity_at_leaf      ::Union{Nothing,Float64}
    max_modal_depth         ::Union{Nothing,Int}

    ## Logic parameters

    # Relation set
    relations               ::Union{
        Nothing,                                            # defaults to a well-known relation set, depending on the data;
        Symbol,                                             # one of the relation sets specified in AVAILABLE_RELATIONS;
        Vector{<:AbstractRelation},                         # explicitly specify the relation set;
        # Vector{<:Union{Symbol,Vector{<:AbstractRelation}}}, # MULTIMODAL CASE: specify a relation set for each modality;
        Function                                            # A function worldtype -> relation set.
    }

    # Condition set
    features                ::Union{
        Nothing,                                                                   # defaults to scalar conditions (with ≥ and <) on well-known feature functions (e.g., minimum, maximum), applied to all variables;
        Vector{<:Union{SoleData.VarFeature,Base.Callable}},                        # scalar conditions with ≥ and <, on an explicitly specified feature set (callables to be applied to each variable, or VarFeature objects);
        Vector{<:Tuple{Base.Callable,Integer}},                                    # scalar conditions with ≥ and <, on a set of features specified as a set of callables to be applied to a set of variables each;
        Vector{<:Tuple{TestOperator,<:Union{SoleData.VarFeature,Base.Callable}}},  # explicitly specify the pairs (test operator, feature);
        Vector{<:SoleData.ScalarMetaCondition},                                    # explicitly specify the scalar condition set.
    }
    conditions              ::Union{
        Nothing,                                                                   # defaults to scalar conditions (with ≥ and <) on well-known feature functions (e.g., minimum, maximum), applied to all variables;
        Vector{<:Union{SoleData.VarFeature,Base.Callable}},                        # scalar conditions with ≥ and <, on an explicitly specified feature set (callables to be applied to each variable, or VarFeature objects);
        Vector{<:Tuple{Base.Callable,Integer}},                                    # scalar conditions with ≥ and <, on a set of features specified as a set of callables to be applied to a set of variables each;
        Vector{<:Tuple{TestOperator,<:Union{SoleData.VarFeature,Base.Callable}}},  # explicitly specify the pairs (test operator, feature);
        Vector{<:SoleData.ScalarMetaCondition},                                    # explicitly specify the scalar condition set.
    }
    # Type for the extracted feature values
    featvaltype             ::Type

    # Initial conditions
    initconditions          ::Union{
        Nothing,                                                                   # defaults to standard conditions (e.g., start_without_world)
        Symbol,                                                                    # one of the initial conditions specified in AVAILABLE_INITIALCONDITIONS;
        MDT.InitialCondition,                                                          # explicitly specify an initial condition for the learning algorithm.
    }

    ## Miscellaneous
    downsize                ::Union{Bool,NTuple{N,Integer} where N,Function}
    force_i_variables       ::Bool
    fixcallablenans         ::Bool
    print_progress          ::Bool
    rng                     ::Union{Random.AbstractRNG,Integer}

    ## DecisionTree.jl parameters
    display_depth           ::Union{Nothing,Int}
    min_samples_split       ::Union{Nothing,Int}
    n_subfeatures           ::Union{Nothing,Int,Float64,Function}
    post_prune              ::Bool
    merge_purity_threshold  ::Union{Nothing,Float64}
    feature_importance      ::Symbol

    ## AdaBoost parameters
    n_iter                  ::Int
end

# keyword constructor
function AdaBoostModalClassifier(;
    max_depth = nothing,
    min_samples_leaf = nothing,
    min_purity_increase = nothing,
    max_purity_at_leaf = nothing,
    max_modal_depth = nothing,
    #
    relations = nothing,
    features = nothing,
    conditions = nothing,
    featvaltype = Float64,
    initconditions = nothing,
    #
    downsize = true,
    force_i_variables = true,
    fixcallablenans = false,
    print_progress = false,
    rng = Random.GLOBAL_RNG,
    #
    display_depth = nothing,
    min_samples_split = nothing,
    n_subfeatures = nothing,
    post_prune = false,
    merge_purity_threshold = nothing,
    feature_importance = :split,
    #
    n_iter = 10,
)
    model = AdaBoostModalClassifier(
        max_depth,
        min_samples_leaf,
        min_purity_increase,
        max_purity_at_leaf,
        max_modal_depth,
        #
        relations,
        features,
        conditions,
        featvaltype,
        initconditions,
        #
        downsize,
        force_i_variables,
        fixcallablenans,
        print_progress,
        rng,
        #
        display_depth,
        min_samples_split,
        n_subfeatures,
        post_prune,
        merge_purity_threshold,
        feature_importance,
        n_iter,
    )
    message = MMI.clean!(model)
    isempty(message) || @warn message
    return model
end

function build_adaboost_modal_stumps(
    y::AbstractVector{T},
    X::AbstractMatrix{S},
    n_iterations::Integer;
    rng=Random.GLOBAL_RNG,
) where {S,T}
    N = length(y)
    n_labels = length(unique(y))
    base_coeff = log(n_labels - 1)
    thresh = 1 - 1 / n_labels
    weights = ones(N) / N
    stumps = DT.Node{S,T}[]
    coeffs = Float64[]
    n_features = size(X, 2)

    # Xref, yref, var_grouping, classes_seen, wref = MDT.reformat(X, y)

    for i in 1:n_iterations
        new_stump = MDT.build_stump( # TODO c'è anche in MDT!!!
            X, y, weights; rng=DT.mk_rng(rng), impurity_importance=false
        )
        predictions = MDT.apply_tree(new_stump, X) # TODO c'è anche in MDT!!!
        err = DT._weighted_error(y, predictions, weights)
        if err >= thresh # should be better than random guess
            continue
        end
        # SAMME algorithm
        new_coeff = log((1.0 - err) / err) + base_coeff
        unmatches = labels .!= predictions
        weights[unmatches] *= exp(new_coeff)
        weights /= sum(weights)
        push!(coeffs, new_coeff)
        push!(stumps, new_stump.node)
        if err < 1e-6
            break
        end
    end
    return (DT.Ensemble{S,T}(stumps, n_features, Float64[]), coeffs)
end

# # ADA BOOST STUMP CLASSIFIER
function classes(y)
    p = CategoricalArrays.pool(y)
    [p[i] for i in 1:length(p)]
end

# MMI.@mlj_model mutable struct AdaBoostModalClassifier <: MMI.Probabilistic
#     n_iter::Int = 10::(_ ≥ 1)
#     feature_importance::Symbol = :impurity::(_ ∈ (:impurity, :split))
#     rng::Union{AbstractRNG,Integer} = GLOBAL_RNG
    
#     relations::Union{
#         Nothing,                                            # defaults to a well-known relation set, depending on the data;
#         Symbol,                                             # one of the relation sets specified in AVAILABLE_RELATIONS;
#         Vector{<:AbstractRelation},                         # explicitly specify the relation set;
#         # Vector{<:Union{Symbol,Vector{<:AbstractRelation}}}, # MULTIMODAL CASE: specify a relation set for each modality;
#         Function                                            # A function worldtype -> relation set.
#     }
#     conditions::Union{
#         Nothing,                                                                     # defaults to scalar conditions (with ≥ and <) on well-known feature functions (e.g., minimum, maximum), applied to all variables;
#         Vector{<:Union{SoleData.VarFeature,Base.Callable}},                        # scalar conditions with ≥ and <, on an explicitly specified feature set (callables to be applied to each variable, or VarFeature objects);
#         Vector{<:Tuple{Base.Callable,Integer}},                                      # scalar conditions with ≥ and <, on a set of features specified as a set of callables to be applied to a set of variables each;
#         Vector{<:Tuple{SoleData.TestOperator,<:Union{SoleData.VarFeature,Base.Callable}}},  # explicitly specify the pairs (test operator, feature);
#         Vector{<:SoleData.ScalarMetaCondition},                                    # explicitly specify the scalar condition set.
#     }
#     featvaltype::Type

#     downsize::Union{Bool,NTuple{N,Integer} where N,Function}
#     force_i_variables::Bool
#     fixcallablenans::Bool
# end

function MMI.fit(
    m::AdaBoostModalClassifier,
    verbosity::Int,
    X,
    y,
    features,
    classes,
)
    # Xref, yref, var_grouping, classes_seen, wref = MMI.reformat(m, X, y)

    integers_seen = unique(y)
    classes_seen  = MMI.decoder(classes)(integers_seen)

    stumps, coefs = build_adaboost_modal_stumps(y, X, m.n_iter, rng=m.rng)
    cache  = nothing

    report = (features=features,)

    return (stumps, coefs, classes_seen, integers_seen), cache, report
end

MMI.fitted_params(::AdaBoostModalClassifier, (stumps,coefs,_)) =
    (stumps=stumps,coefs=coefs)

function MMI.predict(m::AdaBoostModalClassifier, fitresult, Xnew)
    stumps, coefs, classes_seen, integers_seen = fitresult
    scores = DT.apply_adaboost_stumps_proba(
        stumps,
        coefs,
        Xnew,
        integers_seen,
    )
    return MMI.UnivariateFinite(classes_seen, scores)
end

MMI.reports_feature_importances(::Type{<:AdaBoostModalClassifier}) = true

# for fit:

# to get column names based on table access type:
_columnnames(X) = _columnnames(X, Val(Tables.columnaccess(X))) |> collect
_columnnames(X, ::Val{true}) = Tables.columnnames(Tables.columns(X))
_columnnames(X, ::Val{false}) = Tables.columnnames(first(Tables.rows(X)))



# MMI.reformat(::AdaBoostModalClassifier, X, y) =
#     (Tables.matrix(X), MMI.int(y), _columnnames(X), classes(y))
function MMI.reformat(m::AdaBoostModalClassifier, X, y, w = nothing; passive_mode = false)
    X, var_grouping = MDT.wrapdataset(X, m; passive_mode = passive_mode)
    y, classes_seen = MDT.fix_y(y)
    (X, y, var_grouping, classes_seen, w)
end
MMI.selectrows(::AdaBoostModalClassifier, I, Xmatrix, y, meta...) =
    (view(Xmatrix, I, :), view(y, I), meta...)

# for predict:
MMI.reformat(::AdaBoostModalClassifier, X) = (Tables.matrix(X),)
MMI.selectrows(::AdaBoostModalClassifier, I, Xmatrix) = (view(Xmatrix, I, :),)


# # FEATURE IMPORTANCES

# get actual arguments needed for importance calculation from various fitresults.
# get_fitresult(
#     m::Union{DecisionTreeClassifier, RandomForestClassifier, DecisionTreeRegressor},
#     fitresult,
# ) = (fitresult[1],)
# get_fitresult(
#     m::RandomForestRegressor,
#     fitresult,
# ) = (fitresult,)
# get_fitresult(m::AdaBoostModalClassifier, fitresult)= (fitresult[1], fitresult[2])



AdaBoostModalClassifier
function MMI.feature_importances(m::AdaBoostModalClassifier, fitresult, report)
    # generate feature importances for report
    if m.feature_importance == :impurity
        feature_importance_func = DT.impurity_importance
    elseif m.feature_importance == :split
        feature_importance_func = DT.split_importance
    end

    mdl = get_fitresult(m, fitresult)
    features = report.features
    fi = feature_importance_func(mdl..., normalize=true)
    fi_pairs = Pair.(features, fi)
    # sort descending
    sort!(fi_pairs, by= x->-x[2])

    return fi_pairs
end

# # METADATA (MODEL TRAITS)

MMI.metadata_model(
    AdaBoostModalClassifier,
    input_scitype = Union{
        MMI.Table(
            ScientificTypes.Continuous,     AbstractArray{<:ScientificTypes.Continuous,0},    AbstractArray{<:ScientificTypes.Continuous,1},    AbstractArray{<:ScientificTypes.Continuous,2},
            Count,          AbstractArray{<:Count,0},         AbstractArray{<:Count,1},         AbstractArray{<:Count,2},
            OrderedFactor,  AbstractArray{<:OrderedFactor,0}, AbstractArray{<:OrderedFactor,1}, AbstractArray{<:OrderedFactor,2},
        ),
    },
    target_scitype = Union{
        AbstractVector{<:Multiclass},
        AbstractVector{<:ScientificTypes.Continuous},
        AbstractVector{<:Count},
        AbstractVector{<:Finite},
        AbstractVector{<:Textual}
    },
    human_name = "Ada-boosted modal classifier",
    load_path = "$PKG.AdaBoostModalClassifier"
)

end

########################## avail models ##########################
# using Random, Sole, SoleXplorer
# X, y = SoleData.load_arff_dataset("NATOPS")
# train_seed = 11;
# features = [mean, maximum]
# rng = Random.Xoshiro(train_seed)
# Random.seed!(train_seed)

# model = SoleXplorer.get_model(model_name)
# ds = SoleXplorer.preprocess_dataset(X, y, model, features=features)

# classifier = AdaBoostModalClassifier(; n_iter=10, feature_importance=:impurity, rng=Random.TaskLocalRNG(),)

