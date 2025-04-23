using SoleXplorer
using DataFrames
using MLJ

X, y = SoleXplorer.@load_iris
X = SoleXplorer.DataFrame(X)

const ba_warn = Union{Modelset{SoleXplorer.TypeDTC}, Modelset{SoleXplorer.TypeDTR}, Modelset{SoleXplorer.TypeMDT}}

lumen = (
    model          = (mach) -> MLJ.fitted_params(mach).forest,
    tuned_model    = (mach) -> MLJ.fitted_params(mach).best_fitted_params.forest,
    apply_function = apply_forest
)

# const RULES = Dict(
RULES = Dict(
    # intrees(
    #     model,
    #     X,
    #     y::AbstractVector{<:Label};
        
    #     prune_rules::Bool = true,
    #     pruning_s::Union{Float64,Nothing} = nothing,
    #     pruning_decay_threshold::Union{Float64,Nothing} = nothing,
    #     rule_selection_method::Symbol = :CBC,
    #     rule_complexity_metric::Symbol = :natoms,
    #     max_rules::Int = -1,
    #     min_coverage::Union{Float64,Nothing} = nothing,
    #     silent = false,
    #     rng::AbstractRNG = MersenneTwister(1),
    #     return_info::Bool = false,
    # )
    :intrees => m -> begin
        if isnothing(m.setup.resample)
            df = DataFrame(m.ds.Xtest, m.ds.info.vnames)
            rules = SoleXplorer.intrees(m.model, df, m.ds.ytest; silent=true, rng=Xoshiro(11))
        else
            rules = reduce(vcat, map(enumerate(m.model)) do (i, model)
                df = DataFrame(m.ds.Xtest[i], m.ds.info.vnames)
                SoleXplorer.intrees(model, df, m.ds.ytest[i]; silent=true, rng=Xoshiro(11))
            end)
        end
    end,

    # refne(m, Xmin, Xmax; L=100, perc=1.0, max_depth=-1, n_subfeatures=-1, 
    #     partial_sampling=0.7, min_samples_leaf=5, min_samples_split=2, 
    #     min_purity_increase=0.0, seed=3)

    :refne => m -> begin
        if isnothing(m.setup.resample)
            Xmin = map(minimum, eachcol(m.ds.Xtest))
            Xmax = map(maximum, eachcol(m.ds.Xtest))
            rules = SoleXplorer.refne(m.model, Xmin, Xmax; L=10)
        else
            rules = reduce(vcat, map(enumerate(m.model)) do (i, model)
                Xmin = map(minimum, eachcol(m.ds.Xtest[i]))
                Xmax = map(maximum, eachcol(m.ds.Xtest[i]))
                SoleXplorer.refne(model, Xmin, Xmax; L=10)
            end)
        end
    end,
    
    # function trepan(f, X; max_depth=-1, n_subfeatures=-1, partial_sampling=0.5, min_samples_leaf=5, min_samples_split=2, min_purity_increase=0.0, seed=42)
    :trepan => m -> begin
        if isnothing(m.setup.resample)
            rules = SoleXplorer.trepan(m.model, m.ds.Xtest)
        else
            rules = reduce(vcat, map(enumerate(m.model)) do (i, model)
                SoleXplorer.trepan(model, m.ds.Xtest[i])
            end)
        end
    end,
    
    # batrees(f=nothing; dataset_name="iris", num_trees=10, max_depth=10, dsOutput=true)
    :batrees => m -> begin
        m isa ba_warn && throw(ArgumentError("batrees not supported for decision tree model type"))
        if isnothing(m.setup.resample)
            rules = SoleXplorer.batrees(m.model)
        else
            rules = reduce(vcat, map(enumerate(m.model)) do (i, model)
                SoleXplorer.batrees(model)
            end)
        end
    end,

    # rulecosiplus(ensemble::Any, X_train::Any, y_train::Any)
    :rulecosiplus => m -> begin
    if isnothing(m.setup.resample)
        df = DataFrame(m.ds.Xtest, m.ds.info.vnames)
        rules = SoleXplorer.rulecosiplus(m.model, df, m.ds.ytest)
    else
        rules = reduce(vcat, map(enumerate(m.model)) do (i, model)
            df = DataFrame(m.ds.Xtest[i], m.ds.info.vnames)
            SoleXplorer.rulecosiplus(model, df, m.ds.ytest[i])
        end)
    end
end,

    # function lumen(
    #     modelJ, # actualy truth_combinations usa model 
    #     minimization_scheme::Symbol=:mitespresso;
    #     vertical::Real=1.0,
    #     horizontal::Real=1.0,
    #     ott_mode::Bool=false,
    #     controllo::Bool=false,
    #     start_time=time(),
    #     minimization_kwargs::NamedTuple=(;),
    #     filteralphabetcallback=identity,
    #     solemodel=nothing,
    #     apply_function=SoleModels.apply,
    #     silent=false,
    #     return_info=true, # TODO must default to `false`.
    #     vetImportance=[],
    #     kwargs...
    # )
    :lumen => m -> begin
    if isnothing(m.setup.resample)
        rules = SoleXplorer.lumen(lumen.model(m.mach); solemodel=m.model, apply_function=lumen.apply_function, silent=true, return_info=false)
    else
        rules = reduce(vcat, map(enumerate(m.model)) do (i, model)
            SoleXplorer.lumen(model; solemodel=model, silent=true, apply_function=apply_forest, return_info=false)
        end)
    end
end,
)

solem = symbolic_analysis(X, y; model=(type=:randomforest, params=(;max_depth=2)), preprocess=(;rng=Xoshiro(11)))
r = RULES[:lumen](solem)

solem = symbolic_analysis(X, y; model=(type=:decisiontree, params=(;max_depth=2)), preprocess=(;rng=Xoshiro(11)))
rules = RULES[:intrees](solem)
rules = RULES[:refne](solem)
rules = RULES[:trepan](solem)
# rules = RULES[:batrees](solem)
# rules = RULES[:rulecosiplus](solem)
rules = RULES[:lumen](solem)

solem = symbolic_analysis(X, y; model=(type=:randomforest, params=(;max_depth=2)), resample=(type=CV,), preprocess=(;rng=Xoshiro(11)))
rules = RULES[:intrees](solem)
rules = RULES[:refne](solem)
rules = RULES[:trepan](solem)
rules = RULES[:batrees](solem)
# rules = RULES[:rulecosiplus](solem)
rules = RULES[:lumen](solem)

solem = symbolic_analysis(X, y; model=(type=:xgboost, params=(max_depth=2, objective="multi:softprob",)),
    tuning=(method=(;type=latinhypercube), params=(repeats=25, n=10),
        ranges=(
            SoleXplorer.range(:eta, lower=0.1, upper=0.5),
            SoleXplorer.range(:num_round, lower=10, upper=80),
        )
    ), preprocess=(;rng=Xoshiro(11)))
rules = RULES[:intrees](solem)
rules = RULES[:refne](solem)
rules = RULES[:trepan](solem)
rules = RULES[:batrees](solem)
# rules = RULES[:rulecosiplus](solem)
rules = RULES[:lumen](solem)