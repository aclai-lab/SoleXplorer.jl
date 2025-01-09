using Sole
using SoleModels

# ---------------------------------------------------------------------------- #
#                         interesting rule dataframe                           #
# ---------------------------------------------------------------------------- #
"""
    get_rules(
        solemodels;
        method = Sole.listrules,
        min_lift=1.0,
        min_ninstances=0,
        min_coverage=0.10,
        min_ncovered=1,
        normalize=true,
        kwargs...
    )

Extracts interesting rules from one or more (Sole compliant) symbolic models,
returning a DataFrame of rules, and their metrics.
Note that duplicate rules may be returned.

Keyword arguments:
- `method`: a callable method for extracting rules, such as `Sole.listrules` and `Sole.extractrules`
- `min_lift`: minimum lift
- `min_ninstances`: minimum number of instances
- `min_coverage`: minimum coverage
- `min_ncovered`: minimum number of covered instances
- `normalize`: whether to normalize the antecedent
- See [`Sole.listrules`](@ref) or [`Sole.extractrules`](@ref) for additional keyword arguments.
"""
function get_rules(
    model::SoleXplorer.ModelConfig;
    min_lift::Float64=1.0,
    min_ninstances::Int=0,
    min_coverage::Float64=0.10,
    min_ncovered::Int=1,
    normalize::Bool=true,
    threshold_digits::Int=2,
    round_digits::Int=2,
    kwargs...
)
    _X = DataFrame[]

    for m in model.models
        rules = Sole.extractrules(model.rules_method,
            m;
            min_lift=min_lift,
            min_ninstances=min_ninstances,
            min_coverage=min_coverage,
            min_ncovered=min_ncovered,
            normalize=normalize,
            kwargs...
        );

        map(r->(consequent(r), readmetrics(r)), rules)
        irules = sort(rules, by=readmetrics)

        X = DataFrame(antecedent=String[], consequent=Any[]; [name => Vector{Union{Float64, Int}}() for name in keys(readmetrics(irules[1]))]...)

        for rule in irules
            ant = syntaxstring(Sole.antecedent(rule), threshold_digits=threshold_digits)
            cons = SoleModels.leafmodelname(Sole.consequent(rule))
            push!(X, (ant, cons, readmetrics(rule, round_digits=round_digits)...))
        end

        push!(_X, X)
    end

    return vcat(_X...)
end

