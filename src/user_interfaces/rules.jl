# ---------------------------------------------------------------------------- #
#                         interesting rule dataframe                           #
# ---------------------------------------------------------------------------- #
function get_rules(
    model::SoleXplorer.ModelConfig;
    min_lift::Float64=1.0,
    min_ninstances::Int=0,
    min_coverage::Float64=0.10,
    min_ncovered::Int=1,
    normalize::Bool=true,
)
    # sole_dt isa DecisionTree && (sole_dt = [sole_dt,])

    _X = DataFrame[]

    for r in model.rules
        rules = model.rules_method(
            r;
            min_lift=min_lift,
            min_ninstances=min_ninstances,
            min_coverage=min_coverage,
            min_ncovered=min_ncovered,
            normalize=normalize,
        );

        map(r->(consequent(r), readmetrics(r)), rules)
        irules = sort(rules, by=readmetrics)

        isempty(irules) && return nothing
        
        X = DataFrame(antecedent=String[], consequent=String[]; [name => Vector{Union{Float64, Int}}() for name in keys(readmetrics(irules[1]))]...)

        for i in irules
            antecedent = syntaxstring(i.antecedent, threshold_digits=2)
            consequent = i.consequent.outcome
            push!(X, (antecedent, consequent, readmetrics(i, round_digits=2)...))
        end

        push!(_X, X)
    end

    return length(_X) > 1 ? _X : _X[1]
end
