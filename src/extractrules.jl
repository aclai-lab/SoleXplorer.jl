# Rule Extraction Adapters

# This module provides adapter interfaces for working with the SolePostHoc.jl
# package from ACLAI Lab, enabling seamless integration of rule extraction
# algorithms within the SoleXplorer framework.

# The adapters wrap various rule extraction methods from SolePostHoc.jl and
# provide a unified interface for extracting interpretable rules from trained
# machine learning models across different cross-validation folds.

# ---------------------------------------------------------------------------- #
#                                 utilities                                    #
# ---------------------------------------------------------------------------- #
# callable constructor for RuleExtractor types that creates an extractor-parameters tuple.
function (RE::Type{<:RuleExtractor})(;kwargs...)
    return (RE(), (;kwargs...))
end

to_namedtuple(x) = NamedTuple{fieldnames(typeof(x))}(ntuple(i -> getfield(x, i), fieldcount(typeof(x))))

# ---------------------------------------------------------------------------- #
#                             InTreesRuleExtractor                             #
# ---------------------------------------------------------------------------- #
function extractrules(
    extractor :: InTreesRuleExtractor,
    _         :: NamedTuple,
    ds        :: AbstractDataSet,
    solem     :: Vector{AbstractModel}
)::Vector{DecisionSet}
    map(enumerate(solem)) do (i, model)
        X_test, y_test = get_X(ds, :test)[i], get_y(ds, :test)[i]
        RuleExtraction.modalextractrules(
            extractor,
            scalarlogiset(X_test; allow_propositional = true),
            y_test,
            model
        )
    end
end

# ---------------------------------------------------------------------------- #
#                              LumenRuleExtractor                              #
# ---------------------------------------------------------------------------- #
function extractrules(
    extractor :: LumenRuleExtractor,
    params    :: NamedTuple,
    _         :: AbstractDataSet,
    solem     :: Vector{AbstractModel}
)::Vector{LumenResult}
    map(enumerate(solem)) do (_, model)
        RuleExtraction.modalextractrules(extractor, model; params...)
    end
end

# ---------------------------------------------------------------------------- #
#                             BATreesRuleExtractor                             #
# ---------------------------------------------------------------------------- #
function extractrules(
    extractor :: BATreesRuleExtractor,
    params    :: NamedTuple,
    _         :: AbstractDataSet,
    solem     :: Vector{AbstractModel}
)::Vector{DecisionSet}
    map(enumerate(solem)) do (_, model)
        RuleExtraction.modalextractrules(extractor, model; params...)
    end
end

# ---------------------------------------------------------------------------- #
#                          RULECOSIPLUSRuleExtractor                           #
# ---------------------------------------------------------------------------- #
# function extractrules(
#     extractor :: RULECOSIPLUSRuleExtractor,
#     params    :: NamedTuple,
#     ds        :: AbstractDataSet,
#     solem     :: Vector{AbstractModel}
# )::Vector{DecisionSet}
#     map(enumerate(solem)) do (i, model)
#         X_test, y_test = get_X(ds, :test)[i], get_y(ds, :test)[i]
#         RuleExtraction.modalextractrules(extractor, model, X_test, y_test; params...)
#     end
# end

# ---------------------------------------------------------------------------- #
#                              REFNERuleExtractor                              #
# ---------------------------------------------------------------------------- #
function extractrules(
    extractor :: REFNERuleExtractor,
    params    :: NamedTuple,
    ds        :: AbstractDataSet,
    solem     :: Vector{AbstractModel}
)::Vector{DecisionSet}
    map(enumerate(solem)) do (i, model)
        X_test = get_X(ds, :test)[i]
        Xmin = map(minimum, eachcol(X_test))
        Xmax = map(maximum, eachcol(X_test))
        RuleExtraction.modalextractrules(extractor, model, Xmin, Xmax; params...)
    end
end

# ---------------------------------------------------------------------------- #
#                              TREPANRuleExtractor                              #
# ---------------------------------------------------------------------------- #
function extractrules(
    extractor :: TREPANRuleExtractor,
    params    :: NamedTuple,
    ds        :: AbstractDataSet,
    solem     :: Vector{AbstractModel}
)::Vector{DecisionSet}
    map(enumerate(solem)) do (i, model)
        X_test = get_X(ds, :test)[i]
        RuleExtraction.modalextractrules(extractor, model, X_test; params...)
    end
end
