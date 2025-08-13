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
)::DecisionSet
    params = to_namedtuple(extractor)
    rules = reduce(vcat, map(enumerate(solemodels(solem))) do (i, model)
        test = get_test(ds.pidxs[i])
        X_test, y_test = get_X(ds)[test, :], get_y(ds)[test]
        extracted_rules = RuleExtraction.modalextractrules(extractor, model, X_test, y_test; params...)
        extracted_rules.rules  
    end)

    return DecisionSet(rules)
end

# ---------------------------------------------------------------------------- #
#                              LumenRuleExtractor                              #
# ---------------------------------------------------------------------------- #
function extractrules(
    extractor :: LumenRuleExtractor,
    params    :: NamedTuple,
    ds        :: EitherDataSet,
    solem     :: SoleModel
)::DecisionSet
    rules = reduce(vcat, map(enumerate(solemodels(solem))) do (i, model)
        extracted_rules = RuleExtraction.modalextractrules(extractor, model; params...)
        extracted_rules.decision_set.rules
    end)

    return DecisionSet(rules)
end

# ---------------------------------------------------------------------------- #
#                             BATreesRuleExtractor                             #
# ---------------------------------------------------------------------------- #
function extractrules(
    extractor :: BATreesRuleExtractor,
    params    :: NamedTuple,
    ds        :: EitherDataSet,
    solem     :: SoleModel
)::DecisionSet
    rules = reduce(vcat, map(enumerate(solemodels(solem))) do (i, model)
        extracted_rules = RuleExtraction.modalextractrules(extractor, model; params...)
        extracted_rules.rules 
    end)

    return DecisionSet(rules)
end

# ---------------------------------------------------------------------------- #
#                          RULECOSIPLUSRuleExtractor                           #
# ---------------------------------------------------------------------------- #
function extractrules(
    extractor :: RULECOSIPLUSRuleExtractor,
    params    :: NamedTuple,
    ds        :: EitherDataSet,
    solem     :: SoleModel
)::DecisionSet
    rules = reduce(vcat, map(enumerate(solemodels(solem))) do (i, model)
        test = get_test(ds.pidxs[i])
        X_test, y_test = get_X(ds)[test, :], get_y(ds)[test]
        extracted_rules = RuleExtraction.modalextractrules(extractor, model, X_test, y_test; params...)
        extracted_rules.rules 
    end)

    return DecisionSet(rules)
end

# ---------------------------------------------------------------------------- #
#                              REFNERuleExtractor                              #
# ---------------------------------------------------------------------------- #
function extractrules(
    extractor :: REFNERuleExtractor,
    params    :: NamedTuple,
    ds        :: EitherDataSet,
    solem     :: SoleModel
)::DecisionSet
    rules = reduce(vcat, map(enumerate(solemodels(solem))) do (i, model)
        test = get_test(ds.pidxs[i])
        X_test = get_X(ds)[test, :]
        Xmin = map(minimum, eachcol(X_test))
        Xmax = map(maximum, eachcol(X_test))
        extracted_rules = RuleExtraction.modalextractrules(extractor, model, Xmin, Xmax; params...)
        extracted_rules.rules 
    end)

    return DecisionSet(rules)
end

# ---------------------------------------------------------------------------- #
#                              TREPANRuleExtractor                              #
# ---------------------------------------------------------------------------- #
function extractrules(
    extractor :: TREPANRuleExtractor,
    params    :: NamedTuple,
    ds        :: EitherDataSet,
    solem     :: SoleModel
)::DecisionSet
    rules = reduce(vcat, map(enumerate(solemodels(solem))) do (i, model)
        test = get_test(ds.pidxs[i])
        X_test = get_X(ds)[test, :]
        extracted_rules = RuleExtraction.modalextractrules(extractor, model, X_test; params...)
        extracted_rules.rules 
    end)

    return DecisionSet(rules)
end
