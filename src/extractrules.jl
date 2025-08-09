# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
# dovrà essere sicuramente ampliato con un union
const Rules = DecisionSet

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
)::Rules
    params = to_namedtuple(extractor)
    extracted = reduce(vcat, map(enumerate(solemodels(solem))) do (i, model)
        test = get_test(ds.pidxs[i])
        X_test, y_test = get_X(ds)[test, :], get_y(ds)[test]
        decision_set = RuleExtraction.modalextractrules(extractor, model, X_test, y_test; params...)
        decision_set.rules
    end)

    return DecisionSet(extracted)
end

# ---------------------------------------------------------------------------- #
#                              LumenRuleExtractor                              #
# ---------------------------------------------------------------------------- #
function extractrules(
    extractor :: LumenRuleExtractor,
    params    :: NamedTuple,
    ds        :: EitherDataSet,
    solem     :: SoleModel
)::Rules
    extracted = reduce(vcat, map(enumerate(solemodels(solem))) do (i, model)
        lumen_result = RuleExtraction.modalextractrules(extractor, model; params...)
        lumen_result.decision_set.rules
    end)

    return DecisionSet(extracted)
end



# # ---------------------------------------------------------------------------- #
# #                              Rules extraction                                #
# # ---------------------------------------------------------------------------- #

#     :refne        => (
#         L                       = 10,
#         perc                    = 1.0,
#         max_depth               = -1,
#         n_subfeatures           = -1,
#         partial_sampling        = 0.7,
#         min_samples_leaf        = 5,
#         min_samples_split       = 2,
#         min_purity_increase     = 0.0,
#         seed                    = 3
#     ),

#     :trepan       => (
#         max_depth               = -1,
#         n_subfeatures           = -1,
#         partial_sampling        = 0.5,
#         min_samples_leaf        = 5,
#         min_samples_split       = 2,
#         min_purity_increase     = 0.0,
#         seed                    = 42
#     ),

#     :batrees      => (
#         dataset_name            = "iris",
#         num_trees               = 10,
#         max_depth               = 10,
#         dsOutput                = true
#     ),

#     :rulecosi => NamedTuple(
#         # min_coverage            = 0.0,
#         # min_ncovered            = 0,
#         # min_ninstances          = 0,
#         # min_confidence          = 0.0,
#         # min_lift                = 1.0,
#         # metric_filter_callback  = nothing
#     ),


# const EXTRACT_RULES = Dict{Symbol, Function}(
#     :intrees => (m, ds, _) -> begin
#         method = SolePostHoc.RuleExtraction.InTreesRuleExtractor()

#         reduce(vcat, map(enumerate(m.model)) do (i, model)
#             df = DataFrame(ds.X[ds.tt[i].test, :], ds.info.vnames)
#             RuleExtraction.modalextractrules(method, model, df, ds.y[ds.tt[i].test]; m.setup.rulesparams.params...)
#         end)
#     end,
    
#     :refne => (m, ds, _) -> begin
#         method = SolePostHoc.RuleExtraction.REFNERuleExtractor()

#         reduce(vcat, map(enumerate(m.model)) do (i, model)
#             Xmin = map(minimum, eachcol(ds.X[ds.tt[i].test, :]))
#             Xmax = map(maximum, eachcol(ds.X[ds.tt[i].test, :]))
#             RuleExtraction.modalextractrules(method, model, Xmin, Xmax; m.setup.rulesparams.params...)
#         end)
#     end,
    
#     :trepan => (m, ds, _) -> begin
#         method = SolePostHoc.RuleExtraction.TREPANRuleExtractor()

#         reduce(vcat, map(enumerate(m.model)) do (i, model)
#             RuleExtraction.modalextractrules(method, model, ds.X[ds.tt[i].test, :]; m.setup.rulesparams.params...)
#         end)
#     end,
    
#     # TODO: broken
#     # :batrees => (m, ds, _) -> begin
#     #     # m isa tree_warn && throw(ArgumentError("batrees not supported for decision tree model type"))
#     #     method = SolePostHoc.RuleExtraction.BATreesRuleExtractor()
#     #     # if m.setup.resample === nothing
#     #     #     RuleExtraction.modalextractrules(method, m.model; m.setup.rulesparams.params...)
#     #     # else
#     #         reduce(vcat, map(enumerate(m.model)) do (i, model)
#     #             RuleExtraction.modalextractrules(method, model; m.setup.rulesparams.params...)
#     #         end)
#     #     # end
#     # end,

#     :rulecosi => (m, ds, _) -> begin
#         # m isa tree_warn && throw(ArgumentError("rulecosi not supported for decision tree model type"))
#         method = SolePostHoc.RuleExtraction.RULECOSIPLUSRuleExtractor()

#         reduce(vcat, map(enumerate(m.model)) do (i, model)
#             df = DataFrame(ds.X[ds.tt[i].test, :], ds.info.vnames)
#             RuleExtraction.modalextractrules(method, model, df, String.(ds.y[ds.tt[i].test]); m.setup.rulesparams.params...)
#         end)
#     end,

#     :lumen => (m, ds, mach) -> begin
#         # m isa tree_warn && throw(ArgumentError("lumen not supported for decision tree model type"))
#         method = SolePostHoc.RuleExtraction.LumenRuleExtractor()
#         rawmodel = m.setup.rawmodel(mach)

#         reduce(vcat, map(enumerate(m.model)) do (i, model)
#             RuleExtraction.modalextractrules(method, rawmodel; solemodel=model, apply_function=m.setup.config.rawapply, m.setup.rulesparams.params...)
#         end)
#     end
# )
