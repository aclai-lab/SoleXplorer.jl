module MLJAdaBoostModalInterface

import MLJModelInterface as MMI
import ModalDecisionTrees as MDT
import DecisionTree as DT

using Random
import Random.GLOBAL_RNG

import Tables

const PKG = "MLJAdaBoostModalInterface"

function build_adaboost_modal_stumps(
    labels::AbstractVector{T},
    features::AbstractMatrix{S},
    n_iterations::Integer;
    rng=Random.GLOBAL_RNG,
) where {S,T}
    N = length(labels)
    n_labels = length(unique(labels))
    base_coeff = log(n_labels - 1)
    thresh = 1 - 1 / n_labels
    weights = ones(N) / N
    stumps = DT.Node{S,T}[]
    coeffs = Float64[]
    n_features = size(features, 2)
    for i in 1:n_iterations
        new_stump = DT.build_stump( # TODO c'è anche in MDT!!!
            labels, features, weights; rng=DT.mk_rng(rng), impurity_importance=false
        )
        predictions = DT.apply_tree(new_stump, features) # TODO c'è anche in MDT!!!
        err = DT._weighted_error(labels, predictions, weights)
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

MMI.@mlj_model mutable struct AdaBoostModalClassifier <: MMI.Probabilistic
    n_iter::Int            = 10::(_ ≥ 1)
    feature_importance::Symbol = :impurity::(_ ∈ (:impurity, :split))
    rng::Union{AbstractRNG,Integer} = GLOBAL_RNG
end

function MMI.fit(
    m::AdaBoostModalClassifier,
    verbosity::Int,
    Xmatrix,
    yplain,
    features,
    classes,
    )

    integers_seen = unique(yplain)
    classes_seen  = MMI.decoder(classes)(integers_seen)

    stumps, coefs = build_adaboost_modal_stumps(yplain, Xmatrix, m.n_iter, rng=m.rng)
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

MMI.reformat(::AdaBoostModalClassifier, X, y) =
    (Tables.matrix(X), MMI.int(y), _columnnames(X), classes(y))
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
        Table(
            Continuous,     AbstractArray{<:Continuous,0},    AbstractArray{<:Continuous,1},    AbstractArray{<:Continuous,2},
            Count,          AbstractArray{<:Count,0},         AbstractArray{<:Count,1},         AbstractArray{<:Count,2},
            OrderedFactor,  AbstractArray{<:OrderedFactor,0}, AbstractArray{<:OrderedFactor,1}, AbstractArray{<:OrderedFactor,2},
        ),
    },
    target_scitype = Union{
        AbstractVector{<:Multiclass},
        AbstractVector{<:Continuous},
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

