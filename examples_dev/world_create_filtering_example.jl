using Sole, MultiData, MLJ, MLJDecisionTreeInterface
using SoleXplorer, SoleDecisionTreeInterface
using StatsBase, Random
using DataFrames
using BenchmarkTools
using CategoricalArrays

# X, y = SoleData.load_arff_dataset("NATOPS");
# @test SoleData.islogiseed(X) == true
# ninst, nvars, vnames = SoleData.ninstances(X), SoleData.nvariables(X), SoleData.varnames(X)

# può avere senso pensa di avere un dataframe dove ogni colonna (feature) potrebbe avere lunghezza, tipo o dimensione differente?
# tipo colonna 1, vettore di misurazioni temporali, colonna 2 vettore di misurazioni spaziali, colonna 3 float con una sola misurazione?

# dataframe di 1 colonna, con vettori di lunghezza variabile
# ipotizzo che i mondi vadano settati feature per feature, quindi colonna per colonna
# e mi metto nella situazione d'avere misurazioni variabili == differente lunghezza
Random.seed!(123)
random_vectors = [rand(Float64, rand(5:10)) for _ in 1:10]
df = DataFrame(feature = random_vectors)

# trova tutti i mondi possibili, basandoti sulla lunghezza massima
vnames = [:feature]
maxsize = argmax(length.(df[!, vnames...]))
possible_worlds = SoleData.allworlds(df, maxsize)

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
