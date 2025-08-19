# ---------------------------------------------------------------------------- #
#                                 utilities                                    #
# ---------------------------------------------------------------------------- #
function (RE::Type{<:RuleExtractor})(;kwargs...)
    return (RE(), (;kwargs...))
end

to_namedtuple(x) = NamedTuple{fieldnames(typeof(x))}(ntuple(i -> getfield(x, i), fieldcount(typeof(x))))

# ---------------------------------------------------------------------------- #
#                             InTreesRuleExtractor                             #
# ---------------------------------------------------------------------------- #
function extractrules(
    extractor :: InTreesRuleExtractor,
    params    :: NamedTuple,
    ds        :: EitherDataSet,
    solem     :: SoleModel
)::Vector{DecisionSet}
    params = to_namedtuple(extractor)
    map(enumerate(solemodels(solem))) do (i, model)
        test = get_test(ds.pidxs[i])
        X_test, y_test = get_X(ds)[test, :], get_y(ds)[test]
        RuleExtraction.modalextractrules(extractor, model, X_test, y_test; params...)
    end
end

# ---------------------------------------------------------------------------- #
#                              LumenRuleExtractor                              #
# ---------------------------------------------------------------------------- #
function extractrules(
    extractor :: LumenRuleExtractor,
    params    :: NamedTuple,
    ds        :: EitherDataSet,
    solem     :: SoleModel
)::Vector{DecisionSet}
    map(enumerate(solemodels(solem))) do (i, model)
        RuleExtraction.modalextractrules(extractor, model; params...)
    end
end

# ---------------------------------------------------------------------------- #
#                             BATreesRuleExtractor                             #
# ---------------------------------------------------------------------------- #
function extractrules(
    extractor :: BATreesRuleExtractor,
    params    :: NamedTuple,
    ds        :: EitherDataSet,
    solem     :: SoleModel
)::Vector{DecisionSet}
    map(enumerate(solemodels(solem))) do (i, model)
        RuleExtraction.modalextractrules(extractor, model; params...)
    end
end

# ---------------------------------------------------------------------------- #
#                          RULECOSIPLUSRuleExtractor                           #
# ---------------------------------------------------------------------------- #
function extractrules(
    extractor :: RULECOSIPLUSRuleExtractor,
    params    :: NamedTuple,
    ds        :: EitherDataSet,
    solem     :: SoleModel
)::Vector{DecisionSet}
    map(enumerate(solemodels(solem))) do (i, model)
        test = get_test(ds.pidxs[i])
        X_test, y_test = get_X(ds)[test, :], get_y(ds)[test]
        RuleExtraction.modalextractrules(extractor, model, X_test, y_test; params...)
    end
end

# ---------------------------------------------------------------------------- #
#                              REFNERuleExtractor                              #
# ---------------------------------------------------------------------------- #
function extractrules(
    extractor :: REFNERuleExtractor,
    params    :: NamedTuple,
    ds        :: EitherDataSet,
    solem     :: SoleModel
)::Vector{DecisionSet}
    map(enumerate(solemodels(solem))) do (i, model)
        test = get_test(ds.pidxs[i])
        X_test = get_X(ds)[test, :]
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
    ds        :: EitherDataSet,
    solem     :: SoleModel
)::Vector{DecisionSet}
    map(enumerate(solemodels(solem))) do (i, model)
        test = get_test(ds.pidxs[i])
        X_test = get_X(ds)[test, :]
        RuleExtraction.modalextractrules(extractor, model, X_test; params...)
    end
end
