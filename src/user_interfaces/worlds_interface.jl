# ---------------------------------------------------------------------------- #
#                              available filters                               #
# ---------------------------------------------------------------------------- #
const AVAIL_FILTERS = Dict(

)

# ---------------------------------------------------------------------------- #
#                                 get worlds                                   #
# ---------------------------------------------------------------------------- #
function get_wolrds()

end

# filtra i mondi
# f1(x) = length(x) ≥ 3
f1(x) = length(x) == 4
wf = SoleLogics.FunctionalWorldFilter{SoleLogics.Interval{Int},typeof(f1)}(f1)
filtered_worlds = collect(SoleLogics.filterworlds(wf, possible_worlds))

wf_lf = SoleLogics.IntervalLengthFilter(==, 4)
filtered_worlds = collect(SoleLogics.filterworlds(wf_lf, possible_worlds))

# genero un dataset con i le features applicate ai mondi filtrati
features = [maximum, mean]
nwindows = length(filtered_worlds)

X = DataFrame([v => Float64[] for v in [string(j, "(", i, ")w", k) for j in features for i in vnames for k in 1:nwindows]])
for row in eachrow(df)
    push!(X, vec(Float64[world.y ≤ length(row[1]) ? f(row[1][world.x:world.y]) : NaN for f in features, world in filtered_worlds]))
end

@show(X)

y = CategoricalArray(["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"])

dtree = MLJDecisionTreeInterface.DecisionTreeClassifier()

learned_dtree = begin
    mach = machine(dtree, X, y)
    fit!(mach, verbosity=0)
    fitted_params(mach)
end

sole_dtree = solemodel(learned_dtree.tree)
