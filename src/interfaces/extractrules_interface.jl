# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
abstract type AbstractTypeParams end

# ---------------------------------------------------------------------------- #
#                                     struct                                   #
# ---------------------------------------------------------------------------- #
struct RulesParams <: AbstractTypeParams
    type        :: Symbol
    params      :: NamedTuple
end

# ---------------------------------------------------------------------------- #
#                              Rules extraction                                #
# ---------------------------------------------------------------------------- #
const RULES_PARAMS = Dict{Symbol,NamedTuple}(
    :intrees      => (
        prune_rules             = true,
        pruning_s               = nothing,
        pruning_decay_threshold = nothing,
        rule_selection_method   = :CBC,
        rule_complexity_metric  = :natoms,
        max_rules               = -1,
        min_coverage            = nothing,
        silent                  = true,
        rng                     = TaskLocalRNG(),
        return_info             = false
    ),

    :refne        => (
        L                       = 10,
        perc                    = 1.0,
        max_depth               = -1,
        n_subfeatures           = -1,
        partial_sampling        = 0.7,
        min_samples_leaf        = 5,
        min_samples_split       = 2,
        min_purity_increase     = 0.0,
        seed                    = 3
    ),

    :trepan       => (
        max_depth               = -1,
        n_subfeatures           = -1,
        partial_sampling        = 0.5,
        min_samples_leaf        = 5,
        min_samples_split       = 2,
        min_purity_increase     = 0.0,
        seed                    = 42
    ),

    :batrees      => (
        dataset_name            = "iris",
        num_trees               = 10,
        max_depth               = 10,
        dsOutput                = true
    ),

    :rulecosi => NamedTuple(
        # min_coverage            = 0.0,
        # min_ncovered            = 0,
        # min_ninstances          = 0,
        # min_confidence          = 0.0,
        # min_lift                = 1.0,
        # metric_filter_callback  = nothing
    ),

    :lumen        => (
        # minimization_scheme     = :mitespresso,
        # vertical                = 1.0,
        # horizontal              = 1.0,
        # ott_mode                = false,
        # controllo               = false,
        # start_time              = time(),
        # minimization_kwargs     = (;),
        # filteralphabetcallback  = identity,
        # silent                  = true,
        # return_info             = false,
        # vetImportance           = []
    )
)

const EXTRACT_RULES = Dict{Symbol, Function}(
    :intrees => (m, ds, _) -> begin
        method = SolePostHoc.RuleExtraction.InTreesRuleExtractor()

        reduce(vcat, map(enumerate(m.model)) do (i, model)
            df = DataFrame(ds.X[ds.tt[i].test, :], ds.info.vnames)
            RuleExtraction.modalextractrules(method, model, df, ds.y[ds.tt[i].test]; m.setup.rulesparams.params...)
        end)
    end,
    
    :refne => (m, ds, _) -> begin
        method = SolePostHoc.RuleExtraction.REFNERuleExtractor()

        reduce(vcat, map(enumerate(m.model)) do (i, model)
            Xmin = map(minimum, eachcol(ds.X[ds.tt[i].test, :]))
            Xmax = map(maximum, eachcol(ds.X[ds.tt[i].test, :]))
            RuleExtraction.modalextractrules(method, model, Xmin, Xmax; m.setup.rulesparams.params...)
        end)
    end,
    
    :trepan => (m, ds, _) -> begin
        method = SolePostHoc.RuleExtraction.TREPANRuleExtractor()

        reduce(vcat, map(enumerate(m.model)) do (i, model)
            RuleExtraction.modalextractrules(method, model, ds.X[ds.tt[i].test, :]; m.setup.rulesparams.params...)
        end)
    end,
    
    # TODO: broken
    # :batrees => (m, ds, _) -> begin
    #     # m isa tree_warn && throw(ArgumentError("batrees not supported for decision tree model type"))
    #     method = SolePostHoc.RuleExtraction.BATreesRuleExtractor()
    #     # if m.setup.resample === nothing
    #     #     RuleExtraction.modalextractrules(method, m.model; m.setup.rulesparams.params...)
    #     # else
    #         reduce(vcat, map(enumerate(m.model)) do (i, model)
    #             RuleExtraction.modalextractrules(method, model; m.setup.rulesparams.params...)
    #         end)
    #     # end
    # end,

    :rulecosi => (m, ds, _) -> begin
        # m isa tree_warn && throw(ArgumentError("rulecosi not supported for decision tree model type"))
        method = SolePostHoc.RuleExtraction.RULECOSIPLUSRuleExtractor()

        reduce(vcat, map(enumerate(m.model)) do (i, model)
            df = DataFrame(ds.X[ds.tt[i].test, :], ds.info.vnames)
            RuleExtraction.modalextractrules(method, model, df, String.(ds.y[ds.tt[i].test]); m.setup.rulesparams.params...)
        end)
    end,

    :lumen => (m, ds, mach) -> begin
        # m isa tree_warn && throw(ArgumentError("lumen not supported for decision tree model type"))
        method = SolePostHoc.RuleExtraction.LumenRuleExtractor()
        rawmodel = m.setup.rawmodel(mach)

        reduce(vcat, map(enumerate(m.model)) do (i, model)
            RuleExtraction.modalextractrules(method, rawmodel; solemodel=model, apply_function=m.setup.config.rawapply, m.setup.rulesparams.params...)
        end)
    end
)
