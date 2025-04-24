using SoleXplorer
using DataFrames
using Random

X, y = SoleXplorer.@load_iris
X = SoleXplorer.DataFrame(X)

const list_warn = Union{Modelset{SoleXplorer.TypeDTC}, Modelset{SoleXplorer.TypeDTR}, Modelset{SoleXplorer.TypeMDT}}

# const RULES_PARAMS = Dict(
RULES_PARAMS = Dict(
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
        minimization_scheme     = :mitespresso,
        vertical                = 1.0,
        horizontal              = 1.0,
        ott_mode                = false,
        controllo               = false,
        start_time              = time(),
        minimization_kwargs     = (;),
        filteralphabetcallback  = identity,
        silent                  = true,
        return_info             = false,
        vetImportance           = []
    )
)

# const AVAIL_RULES = Dict(
AVAIL_RULES = Dict(
    :intrees => m -> begin
        if isnothing(m.setup.resample)
            df = DataFrame(m.ds.Xtest, m.ds.info.vnames)
            SolePostHoc.intrees(m.model, df, m.ds.ytest; m.setup.rulesparams.params...)
        else
            reduce(vcat, map(enumerate(m.model)) do (i, model)
                df = DataFrame(m.ds.Xtest[i], m.ds.info.vnames)
                SolePostHoc.intrees(model, df, m.ds.ytest[i]; m.setup.rulesparams.params...)
            end)
        end
    end,
    
    :refne => m -> begin
        if isnothing(m.setup.resample)
            Xmin  = map(minimum, eachcol(m.ds.Xtest))
            Xmax  = map(maximum, eachcol(m.ds.Xtest))
            SolePostHoc.refne(m.model, Xmin, Xmax; m.setup.rulesparams.params...)
        else
            reduce(vcat, map(enumerate(m.model)) do (i, model)
                Xmin = map(minimum, eachcol(m.ds.Xtest[i]))
                Xmax = map(maximum, eachcol(m.ds.Xtest[i]))
                SolePostHoc.refne(model, Xmin, Xmax; m.setup.rulesparams.params...)
            end)
        end
    end,
    
    :trepan => m -> begin
        if isnothing(m.setup.resample)
            SolePostHoc.trepan(m.model, m.ds.Xtest; m.setup.rulesparams.params...)
        else
            reduce(vcat, map(enumerate(m.model)) do (i, model)
                SolePostHoc.trepan(model, m.ds.Xtest[i]; m.setup.rulesparams.params...)
            end)
        end
    end,
    
    :batrees => m -> begin
        m isa ba_warn && throw(ArgumentError("batrees not supported for decision tree model type"))
        if isnothing(m.setup.resample)
            SolePostHoc.batrees(m.model; m.setup.rulesparams.params...)
        else
            reduce(vcat, map(enumerate(m.model)) do (i, model)
                SolePostHoc.batrees(model; m.setup.rulesparams.params...)
            end)
        end
    end,

    :rulecosi => m -> begin
        if isnothing(m.setup.resample)
            df = DataFrame(m.ds.Xtest, m.ds.info.vnames)
            dl = SolePostHoc.rulecosiplus(m.model, df, String.(m.ds.ytest); m.setup.rulesparams.params...)
            ll = listrules(dl, use_shortforms=false) # decision list to list of rules
            rules_obj = SolePostHoc.convert_classification_rules(dl, ll)
            DecisionSet(rules_obj)
        else
            reduce(vcat, map(enumerate(m.model)) do (i, model)
                df = DataFrame(m.ds.Xtest[i], m.ds.info.vnames)
                dl = SolePostHoc.rulecosiplus(model, df, String.(m.ds.ytest[i]); m.setup.rulesparams.params...)
                ll = listrules(dl, use_shortforms=false) # decision list to list of rules
                rules_obj = SolePostHoc.convert_classification_rules(dl, ll)
                DecisionSet(rules_obj)
            end)
        end
    end,

    :lumen => m -> begin
        m isa ba_warn && throw(ArgumentError("lumen not supported for decision tree model type"))
        if isnothing(m.setup.resample)
            rawmodel = m.setup.rawmodel(m.mach)
            SolePostHoc.lumen(rawmodel; solemodel=m.model, apply_function=m.setup.config.rawapply, m.setup.rulesparams.params...)
        else
            reduce(vcat, map(enumerate(m.model)) do (i, model)
                rawmodel = m.setup.rawmodel(m.mach[i])
                SolePostHoc.lumen(rawmodel; solemodel=model, apply_function=m.setup.config.rawapply, m.setup.rulesparams.params...)
            end)
        end
    end,
)

m = symbolic_analysis(X, y; model=(type=:randomforest, params=(;max_depth=2)), preprocess=(;rng=Xoshiro(11)))

# intrees

# refne

# trepan

# batrees

# rulecosi

# lumen
