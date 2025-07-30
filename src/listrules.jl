# ---------------------------------------------------------------------------- #
#                                  prefazione                                  #
# ---------------------------------------------------------------------------- #
# Sole è una suite di machine learning come, che io sappia, nulla di simile.
# Capace di utilizzare logica simbolica, e di poter estrarre le regole dal modello.
# Sarebbe quindi logico aspettarsi prestazioni leggermente peggiori rispetto alla concorrenza,
# e soprattutto al diretto rivale, MLJ, di cui oltretutto utilizza molto codice.
# Ma vediamo insieme qual'è il punto di partenza, a livello di prestazioni,
# eseguendo lo stesso task sia in MLJ che in Sole.
# NB: il task proposto è proposizionale: MLJ non lavora in modale, quella è una peculiarietà
# di Sole e quindi non esistono comparazioni.

# ---------------------------------------------------------------------------- #
#                          preparazione esperimento                            #
# ---------------------------------------------------------------------------- #
using SoleXplorer
using SoleModels
using MLJ
using DataFrames, Random
using Test, BenchmarkTools
import DecisionTree as DT

# L'esperimento è eseguito sul classico dataset Iris, caricato tramite la macro offerta da MLJ.
Xc, yc = @load_iris
Xc = DataFrame(Xc)

# Esperimento condotto con Sole
@btime begin
    symbolic_analysis(
        Xc, yc,
        model=DecisionTreeClassifier(),
        resample=Holdout(shuffle=true),
        train_ratio=0.7,
        rng=Xoshiro(1),
        measures=(accuracy, kappa)
    )
end
# 444.172 μs (3370 allocations: 247.12 KiB)

# Esperimento condotto con MLJ tramite la funzione evaluate, che meglio rappresenta l'antagonismo con Sole
@btime begin
    Tree = @load DecisionTreeClassifier pkg=DecisionTree verbosity=0
    tree = Tree()
    evaluate(
        tree, Xc, yc;
        resampling=Holdout(shuffle=true),
        measures=[accuracy, kappa],
        per_observation=true,
        verbosity=0
    )
end
# 369.499 μs (1900 allocations: 118.90 KiB)

# Notiamo che il divario tra Sole e Mlj è abbastanza esiguo, anzi: assolutamente trascurabile
# se teniamo conto delle potenzialità aggiuntive di Sole rispetto a MLJ.
# Ma andiamo oltre e proviamo una RandomForest di 100 alberi:
@btime begin
    symbolic_analysis(
        Xc, yc,
        model=RandomForestClassifier(;n_trees=100),
        resample=Holdout(shuffle=true),
        train_ratio=0.7,
        rng=Xoshiro(1),
        measures=(accuracy, kappa)
    )
end
# 14.890 ms (183395 allocations: 9.30 MiB)

@btime begin
    Forest = @load RandomForestClassifier pkg=DecisionTree verbosity=0
    forest = Forest(;n_trees=100)
    evaluate(
        forest, Xc, yc;
        resampling=Holdout(shuffle=true),
        measures=[accuracy, kappa],
        per_observation=true,
        verbosity=0
    )
end
# 2.557 ms (12502 allocations: 1.28 MiB)

# Come potete notare, qui la forbice di prestazioni aumenta considerevolmente.
# E stiamo verificando solamente un esperimento giocattolo; 
# cosa potrebbe succedere nel caso di un utilizzo reale?
# Il problema è già stato evidenziato da Balbo, quindi ho ragionato su una soluzione.

# ---------------------------------------------------------------------------- #
#                          considerazioni iniziali                             #
# ---------------------------------------------------------------------------- #
# da tempo, con Giovanni, Fede, Marco e forse anche altri di voi, ho espresso i miei dubbi sull'utilizzo
# LIMITATAMENTE agli esperimenti proposizionali, del wrapper logiset.
# Si tratta di una struttura fondamentale per la logica modale,
# ma forse è un elefante in un negozio di cristalli per la logica proposizionale?
# Giovanni aveva espresso dubbi sulle mie iniziali considerazioni, e come sempre aveva ragione.
# La risposta è quindi NO: l'utilizzo del logiset non rappresenta un collo di bottiglia,
# ecco la prova, provata.

@btime SoleData.scalarlogiset(Xc; silent=true, allow_propositional=true)
# 375.088 μs (3032 allocations: 990.48 KiB)

# Come potete vedere, le risorse richieste per la creazione di un logiset non sono il punto cardine
# del nostro disavanzo, ne in termini di velocità e nemmeno in termini di consumo memoria:
# assolutamente trascurabili.

# ---------------------------------------------------------------------------- #
#                              collo di bottiglia                              #
# ---------------------------------------------------------------------------- #
# Ho iniziato ad eseguire test su strutture dati alternative e ho trovato il collo di bottiglia.
# Prendiamo un albero Sole:

test = symbolic_analysis(
    Xc, yc,
    model=DecisionTreeClassifier(),
    resample=Holdout(shuffle=true),
    train_ratio=0.7,
    rng=Xoshiro(1),
    measures=(accuracy, kappa)
)
soletree = test.sole.sole[1] 

# julia> soletree.
# info
# root

# Vediamo che la struttura root dell'albero è composta da info e root, dove root è l'albero.
# La struttura info è sicuramente interessante, guardiamola:

# julia> soletree.info
# (featurenames = [:sepal_length, :sepal_width, :petal_length, :petal_width],
#  supporting_predictions = ["versicolor", "versicolor" … "virginica", "setosa"],
#  supporting_labels = ["versicolor", "versicolor" … "virginica", "setosa"],)

# Geniale: info contiene le label originale e le previsioni, nonchè il nome delle features!

# Il problema è che questo si ripete per ogni singolo nodo dell'albero,
# e per di più tutti gli info sono differenti, sia in risultati sia in lunghezza.

# julia> test.sole.sole[1].root.info
# (supporting_predictions = ["versicolor", "versicolor" … "versicolor", "setosa"],
#  supporting_labels = ["versicolor", "versicolor" … "versicolor", "setosa"],)

# julia> test.sole.sole[1].root.posconsequent.info
# (supporting_predictions = ["setosa", "setosa" … "setosa", "setosa"],
#  supporting_labels = ["setosa", "setosa" … "setosa", "setosa"],)

# julia> test.sole.sole[1].root.negconsequent.negconsequent.info
# (supporting_predictions = ["virginica", "virginica" … "virginica", "virginica"],
#  supporting_labels = ["virginica", "virginica" … "virginica", "virginica"],)

# Sicuramente questo è il collo di bottiglia: immaginiamoci, per 100 alberi, segnare per ogni nodo
# 2 vettori stringa. Beh è facile immaginarsi un aumento considerevole della memoria.

# Ora la mia domanda, da ignorante, è questa: non è che c'è un pò di ridondanza?
# Che questo sia IL METODO per lavorare con logica modale e magari il banale logica proposizionale
# abbiamo trovato l'elefante nel negozio di cristalli?

# Andiamo avanti: a che servono tutti questi 'info'?
# a estrarre le regole tramite SoleModels.listrule().
# Solo a quello? Non lo so, se lo sapessi non sarei qui a tediarvi.

# ---------------------------------------------------------------------------- #
#                            listrules alternativo                             #
# ---------------------------------------------------------------------------- #
# La domanda che mi pongo è questa: ma davvero 'listrules' ha bisogno di ogni singolo info, in ogni nodo?
# Mi viene da sospettare che magari basta passargli l'info in root, cioè quello finale e definitivo.
# NB: questo sempre limitatamente alla logica proposizionale, la logica modale la affronteremo più avanti.

# Quindi l'idea potrebbe essere quella di riscrivere listrule, propagando, nell'algoritmo ricorsivo,
# l'info di root, bypassando le strutture info "locali".

# Ecco come lo riscriverei:
# Quello che troverete qui sotto è esattamente l'algoritmo che trovate in rule-extraction.jl di SoleModels
# ho cancellato le parti commentate presenti nell'algoritmo originale per non creare confusione,
# e troverete commentate le mie proposte di modifica, una sorta di git diff manuale.
# NB: per praticità, e per poter comparare i due algoritmi, ho rinominato
# listrule in pasorules, viva l'autostima.

function pasorules(
    m;
    compute_metrics::Union{Nothing,Bool} = false,
    metrics_kwargs::NamedTuple = (;),
    use_shortforms::Bool = true,
    use_leftmostlinearform::Union{Nothing,Bool} = nothing,
    normalize::Bool = false,
    normalize_kwargs::NamedTuple = (; allow_atom_flipping = true, rotate_commutatives = false, ),
    scalar_simplification::Union{Bool,NamedTuple} = normalize ? (; allow_scalar_range_conditions = true) : false,
    force_syntaxtree::Bool = false,
    min_coverage::Union{Nothing,Number} = nothing,
    min_ncovered::Union{Nothing,Number} = nothing,
    min_ninstances::Union{Nothing,Number} = nothing,
    min_confidence::Union{Nothing,Number} = nothing,
    min_lift::Union{Nothing,Number} = nothing,
    metric_filter_callback::Union{Nothing,Base.Callable} = nothing,
    kwargs...,
)
    subkwargs = (;
        use_shortforms = use_shortforms,
        use_leftmostlinearform = use_leftmostlinearform,
        normalize = normalize,
        normalize_kwargs = normalize_kwargs,
        scalar_simplification = scalar_simplification,
        force_syntaxtree = force_syntaxtree,
        metrics_kwargs = metrics_kwargs,
        min_ninstances = min_ninstances,
        min_coverage = min_coverage,
        min_ncovered = min_ncovered,
        min_confidence = min_confidence,
        min_lift = min_lift,
        metric_filter_callback = metric_filter_callback,
        kwargs...)

    @assert compute_metrics in [false] "TODO implement"
    @assert SoleModels.issymbolicmodel(m) "Model m is not symbolic. Please provide method issymbolicmodel(::$(typeof(m)))."

    # inizio a propagare info root
    # questa è l'unica modifica fatta alla funzione listrules
    rules = _pasorules(m, m.info; subkwargs...)

    if compute_metrics || !isnothing(min_confidence) || !isnothing(min_coverage) || !isnothing(min_ncovered) || !isnothing(min_ninstances) || !isnothing(min_lift)
        rules = Iterators.filter(r->begin
            ms = readmetrics(r; metrics_kwargs...)
            compute_metrics && (info!(r, ms))
            return (isnothing(min_ninstances) || (ms.ninstances >= min_ninstances)) &&
            (isnothing(min_coverage) || (ms.coverage >= min_coverage)) &&
            (isnothing(min_ncovered) || (ms.ncovered >= min_ncovered)) &&
            (isnothing(min_confidence) || (ms.confidence >= min_confidence)) &&
            (isnothing(min_lift) || (ms.lift >= min_lift)) &&
            (isnothing(metric_filter_callback) || metric_filter_callback(ms))
        end, rules)
    end

    rules = collect(rules) # TODO remove in the future?

    return rules
end

# aggiunto arg i = info root
function _pasorules(m::AbstractModel, i::NamedTuple; kwargs...)
    error("Please, provide method _pasorules(::$(typeof(m))) ($(typeof(m)) is a symbolic model).")
end

# aggiunto arg i, root.info
# ATTENZIONE! qui, in origine, veniva passata l'info relativa alla subrule.
# ora invece si passa l'info presente in root dell'albero
# _pasorules(m::LeafModel{O}, i::NamedTuple; kwargs...) where {O} = [Rule{O}(⊤, m, SoleModels.info(m))]
_pasorules(m::LeafModel{O}, i::NamedTuple; kwargs...) where {O} = [Rule{O}(⊤, m, i)]

function _pasorules(
    m::Rule{O},
    # aggiunto arg i, root.info
    i::NamedTuple;
    use_leftmostlinearform::Union{Nothing,Bool} = nothing,
    force_syntaxtree::Bool = false,
    kwargs...
) where {O}
    use_leftmostlinearform = !isnothing(use_leftmostlinearform) ? use_leftmostlinearform : false
    [begin
        φ = combine_antecedents(antecedent(m), antecedent(subrule), use_leftmostlinearform, force_syntaxtree)
        # ATTENZIONE! qui, in origine, veniva passata l'info relativa alla subrule.
        # ora invece si passa l'info presente in root dell'albero
        # Rule{O}(φ, consequent(subrule), SoleModels.info(subrule))
        Rule{O}(φ, consequent(subrule), i)
    end for subrule in _pasorules(consequent(m), i; force_syntaxtree = force_syntaxtree, use_leftmostlinearform = use_leftmostlinearform, kwargs...)]
end

function _pasorules(
    m::Branch{O},
    # aggiunto arg i, root.info
    i::NamedTuple;
    use_shortforms::Bool = true,
    use_leftmostlinearform::Union{Nothing,Bool} = nothing,
    normalize::Bool = false,
    normalize_kwargs::NamedTuple = (; allow_atom_flipping = true, rotate_commutatives = false, ),
    scalar_simplification::Union{Bool,NamedTuple} = normalize ? (; allow_scalar_range_conditions = true) : false,
    force_syntaxtree::Bool = false,
    min_confidence::Union{Nothing,Number} = nothing,
    min_coverage::Union{Nothing,Number} = nothing,
    min_ninstances::Union{Nothing,Number} = nothing,
    flip_atoms::Bool = true,
    kwargs...,
) where {O}
    use_leftmostlinearform = !isnothing(use_leftmostlinearform) ? use_leftmostlinearform : (antecedent(m) isa SoleLogics.AbstractSyntaxStructure) # TODO default to true

    subkwargs = (;
        use_shortforms = use_shortforms,
        use_leftmostlinearform = use_leftmostlinearform,
        normalize = normalize,
        normalize_kwargs = normalize_kwargs,
        scalar_simplification = false,
        force_syntaxtree = force_syntaxtree,
        min_confidence = min_confidence,
        min_coverage = min_coverage,
        min_ninstances = min_ninstances,
        kwargs...)
    _subrules = []
    # tutti i SoleModels.info(m), che si riferiscono agli info locali,
    # vengono sostituiti con il parametro in ingresso 'i'
    # che rappresenta l'info della root dell'albero
    # if isnothing(min_ninstances) || (haskey(SoleModels.info(m), :supporting_labels) && length(SoleModels.info(m, :supporting_labels)) >= min_ninstances)
    if isnothing(min_ninstances) || (haskey(i, :supporting_labels) && length(i[:supporting_labels]) >= min_ninstances)
        append!(_subrules, [(true,  r) for r in _pasorules(posconsequent(m), i; subkwargs...)])
        append!(_subrules, [(false, r) for r in _pasorules(negconsequent(m), i; subkwargs...)])
    end

    rules = map(((flag, subrule),)->begin
            known_infokeys = [:supporting_labels, :supporting_predictions, :shortform, :this, :multipathformula]
            # ks = setdiff(keys(SoleModels.info(m)), known_infokeys)
            ks = setdiff(keys(i), known_infokeys)
            if length(ks) > 0
                @warn "Dropping info keys: $(join(repr.(ks), ", "))"
            end

            _info = (;)
            # if haskey(SoleModels.info(m), :supporting_labels) && haskey(SoleModels.info(m), :supporting_predictions)
            if haskey(i, :supporting_labels) && haskey(i, :supporting_predictions)
                _info = merge((;), (;
                    # supporting_labels = SoleModels.info(m).supporting_labels,
                    # supporting_predictions = SoleModels.info(m).supporting_predictions,
                    supporting_labels = i.supporting_labels,
                    supporting_predictions = i.supporting_predictions,
                ))
            # elseif (haskey(SoleModels.info(m), :supporting_labels) != haskey(SoleModels.info(m), :supporting_predictions))
            elseif (haskey(i, :supporting_labels) != haskey(i, :supporting_predictions))
                @warn "List rules encountered an unexpected case. Both " *
                    " supporting_labels and supporting_predictions are necessary for correctly computing performance metrics. "
            end

            antformula, using_shortform = begin
                # if (use_shortforms && haskey(SoleModels.info(subrule), :shortform))
                #     SoleModels.info(subrule)[:shortform], true
                if (use_shortforms && haskey(i, :shortform))
                    i[:shortform], true
                else
                    smart_neg(f) = (f isa Atom && flip_atoms && SoleLogics.hasdual(f) ? SoleLogics.dual(f) : ¬f)
                    _antd = antecedent(m)
                    (flag ? _antd : smart_neg(_antd)), false
                end
            end
            antformula = force_syntaxtree ? tree(antformula) : antformula

            if subrule isa LeafModel
                ant = antformula
                normalize && (ant = SoleLogics.normalize(ant; normalize_kwargs...))
                ant = SoleModels._scalar_simplification(ant, scalar_simplification)
                subi = (;)
                # Rule(ant, subrule, merge(SoleModels.info(subrule), subi, _info))
                Rule(ant, subrule, merge(i, subi, _info))
            elseif subrule isa Rule
                ant = begin
                    if using_shortform
                        antformula
                    else
                        φ = SoleModels.combine_antecedents(antformula, antecedent(subrule), use_leftmostlinearform, force_syntaxtree)
                        φ
                    end
                end
                normalize && (ant = SoleLogics.normalize(ant; normalize_kwargs...))
                ant = SoleModels._scalar_simplification(ant, scalar_simplification)
                # Rule(ant, consequent(subrule), merge(SoleModels.info(subrule), _info))
                Rule(ant, consequent(subrule), merge(i, _info))
            else
                error("Unexpected rule type: $(typeof(subrule)).")
            end
        end, _subrules)

    return rules
end

function _pasorules(
    m::DecisionList,
    # aggiunto arg i, root.info
    i::NamedTuple;
    normalize::Bool = false,
    normalize_kwargs::NamedTuple = (; allow_atom_flipping = true, ),
    scalar_simplification::Union{Bool,NamedTuple} = normalize ? (; allow_scalar_range_conditions = true) : false,
    force_syntaxtree::Bool = false,
    kwargs...
)
    rules = listimmediaterules(m;
        normalize = normalize,
        scalar_simplification = scalar_simplification,
        normalize_kwargs = normalize_kwargs,
        force_syntaxtree = force_syntaxtree,
    )
    return rules
end

# aggiunto arg i, root.info
_pasorules(m::DecisionTree, i::NamedTuple; kwargs...) = _pasorules(root(m), i; kwargs...)

function _pasorules(
    m::DecisionEnsemble,
    # aggiunto arg i, root.info
    i::NamedTuple;
    suppress_parity_warning = true,
    kwargs...
)
    modelrules = [_pasorules(subm, i; kwargs...) for subm in SoleModels.models(m)]
    @assert all(r->consequent(r) isa ConstantModel, Iterators.flatten(modelrules))

    SoleModels.IterTools.imap(rulecombination->begin
        rulecombination = collect(rulecombination)
        ant = SoleModels.join_antecedents(antecedent.(rulecombination))
        o_cons = SoleModels.bestguess(outcome.(consequent.(rulecombination)), m.weights; suppress_parity_warning)
        i_cons = merge(SoleModels.info.(consequent.(rulecombination))...)
        cons = ConstantModel(o_cons, i_cons)
        infos = merge(SoleModels.info.(rulecombination)...)
        Rule(ant, cons, infos)
        end, Iterators.product(modelrules...)
    )
end

# aggiunto arg i, root.info
_pasorules(m::MixedModel, i::NamedTuple; kwargs...) = _pasorules(root(m), i; kwargs...)

# ---------------------------------------------------------------------------- #
#                               test comparativi                               #
# ---------------------------------------------------------------------------- #
# Ora conviene verificare se le regole estratte da pasorules coincidono con le regole estratte da listrule.
# Per farlo eseguo una batteria di test, sempre su Iris, ma con dataset differenti grazie all'utilizzo
# di differenti semi random.

# Partiamo con DecisionTreeClassifier
for seed in 1:200
    ds = setup_dataset(
        Xc, yc;
        model=DecisionTreeClassifier(),
        resample=Holdout(;shuffle=true),
        rng=Xoshiro(seed),
    )
    solem = train_test(ds)

    model = solem.sole[1]
    test = get_test(ds.pidxs[1])
    X, y = get_X(ds)[test, :], get_y(ds)[test]
    test_original = listrules(model)
    test_paso = pasorules(model)

    @test length(test_original) == length(test_paso)
    for i in 1:length(test_paso)
        @test test_original[i].antecedent.grandchildren == test_paso[i].antecedent.grandchildren
        @test test_original[i].consequent.outcome == test_paso[i].consequent.outcome
    end
end

# Esempio signolo giusto per sincerarci
ds = setup_dataset(
    Xc, yc;
    model=DecisionTreeClassifier(),
    resample=Holdout(;shuffle=true),
    rng=Xoshiro(11),
)
solem = train_test(ds)
model = solem.sole[1]

test_original = listrules(model)
test_paso = pasorules(model)

# proviamo anche con RandomForest
for seed in 1:50
    ds = setup_dataset(
        Xc, yc;
        # con 100 alberi si rompe julia!
        model=RandomForestClassifier(n_trees=5),
        resample=Holdout(;shuffle=true),
        rng=Xoshiro(seed),
    )
    solem = train_test(ds)

    model = solem.sole[1]
    test = get_test(ds.pidxs[1])
    X, y = get_X(ds)[test, :], get_y(ds)[test]
    test_original = listrules(model)
    test_paso = pasorules(model)

    @test length(test_original) == length(test_paso)
    for i in 1:length(test_paso)
        @test test_original[i].antecedent.grandchildren == test_paso[i].antecedent.grandchildren
        @test test_original[i].consequent.outcome == test_paso[i].consequent.outcome
    end
end

# Esempio singolo di debug
ds = setup_dataset(
    Xc, yc;
    model=RandomForestClassifier(n_trees=5),
    resample=Holdout(;shuffle=true),
    rng=Xoshiro(11),
)
solem = train_test(ds)
model = solem.sole[1]

test_original = listrules(model)
test_paso = pasorules(model)

# Sembrerebbe che listrules funziona anche se utilizza l'info di root, bypassando tutti gli info salvati sui vari nodi.
# questo potrebbe essere un bel risparmio di memoria.
# Si potrebbe pensare ad una struttura che non colleziona i vari info, ma crea l'info, con le label e le predizioni,
# solo in fase di test (utilizzando il gergo di Sole, solo eseguendo l'apply!)
# Vediamo come potrebbe essere:

# ---------------------------------------------------------------------------- #
#                           solemodel senza info?                              #
# ---------------------------------------------------------------------------- #
# Ora arriva la parte più complessa: rivedere completamente il metodo con cui si costruisce
# l'albero Sole, e il relativo apply.
# Tenterò di essere il meno possibile invasivo, soprattutto perchè
# si potrebbe trovare una soluzione più elegante, a partire dalle strutture dati,
# ma questo, con molta probabilità, romperebbe diversi test
# e francamente, con Parigi alle porte, mi viene da pensare che non ce lo possiamo permettere.
# magari, se questa mia proposta verrà avvallata dalla commissione,
# ne parleremo ad una prossima Sole Reunion

# ---------------------------------------------------------------------------- #
#                       nuova struttura DecisionTree                           #
# ---------------------------------------------------------------------------- #
# Per prima cosa servirebbe modificare le strutture DecisionTree e DecisionEnsemble.
# Semplicemente cambiandole da 'struct' a 'mutable struct'
# La motivazione:
# dovremo scrivere la root info, solo dopo aver fatto il test del modello tramite apply.
# prendendo spunto da MLJ ho notato che non si fanno grossi problemi in tal senso,
# per esempio: una struttura importante come la mach, è in realtà una mutable struct.
# quindi viste le premesse mi permetterei di fare questa modifica.

mutable struct PasoDecisionEnsemble{O,T<:AbstractModel,A<:Base.Callable,W<:Union{Nothing,AbstractVector}} <: SoleModels.AbstractDecisionEnsemble{O}
    models::Vector{T}
    aggregation::A
    weights::W
    info::NamedTuple

    function PasoDecisionEnsemble{O}(
        models::AbstractVector{T},
        aggregation::Union{Nothing,Base.Callable},
        weights::Union{Nothing,AbstractVector},
        info::NamedTuple = (;);
        suppress_parity_warning=false,
        parity_func=x->argmax(x)
    ) where {O,T<:AbstractModel}
        @assert length(models) > 0 "Cannot instantiate empty ensemble!"
        models = wrap.(models)
        if isnothing(aggregation)
            # if a suppress_parity_warning parameter is provided, then the aggregation's suppress_parity_warning defaults to it;
            #  otherwise, it defaults to bestguess's suppress_parity_warning
            # if isnothing(suppress_parity_warning)
            #     aggregation = function (args...; kwargs...) bestguess(args...; suppress_parity_warning, parity_func, kwargs...) end
            # else
                aggregation = function (args...; suppress_parity_warning, kwargs...) bestguess(args...; suppress_parity_warning, parity_func, kwargs...) end
            # end
        else
            !suppress_parity_warning || @warn "Unexpected value for suppress_parity_warning: $(suppress_parity_warning)."
        end
        # T = typeof(models)
        W = typeof(weights)
        A = typeof(aggregation)
        new{O,T,A,W}(collect(models), aggregation, weights, info)
    end
    
    function PasoDecisionEnsemble{O}(
        models::AbstractVector;
        kwargs...
    ) where {O}
        info = (;)
        PasoDecisionEnsemble{O}(models, nothing, nothing, info; kwargs...)
    end

    function PasoDecisionEnsemble{O}(
        models::AbstractVector,
        info::NamedTuple;
        kwargs...
    ) where {O}
        PasoDecisionEnsemble{O}(models, nothing, nothing, info; kwargs...)
    end

    function PasoDecisionEnsemble{O}(
        models::AbstractVector,
        aggregation::Union{Nothing,Base.Callable},
        info::NamedTuple = (;);
        kwargs...
    ) where {O}
        PasoDecisionEnsemble{O}(models, aggregation, nothing, info; kwargs...)
    end

    function PasoDecisionEnsemble{O}(
        models::AbstractVector,
        weights::AbstractVector,
        info::NamedTuple = (;);
        kwargs...
    ) where {O}
        PasoDecisionEnsemble{O}(models, nothing, weights, info; kwargs...)
    end

    function PasoDecisionEnsemble(
        models::AbstractVector,
        args...; kwargs...
    )
        @assert length(models) > 0 "Cannot instantiate empty ensemble!"
        models = wrap.(models)
        O = Union{outcometype.(models)...}
        PasoDecisionEnsemble{O}(models, args...; kwargs...)
    end
end

mutable struct PasoDecisionTree{O} <: AbstractModel{O}
    root::M where {M<:Union{LeafModel{O},Branch{O}}}
    info::NamedTuple

    function PasoDecisionTree(
        root::Union{LeafModel{O},Branch{O}},
        info::NamedTuple = (;),
    ) where {O}
        new{O}(root, info)
    end

    function PasoDecisionTree(
        root::Any,
        info::NamedTuple = (;),
    )
        root = wrap(root)
        M = typeof(root)
        O = outcometype(root)
        @assert M <: Union{LeafModel{O},Branch{O}} "" *
            "Cannot instantiate PasoDecisionTree{$(O)}(...) with root of " *
            "type $(typeof(root)). Note that the should be either a LeafModel or a " *
            "Branch. " *
            "$(M) <: $(Union{LeafModel,Branch{<:O}}) should hold."
        new{O}(root, info)
    end

    function PasoDecisionTree(
        antecedent::Formula,
        posconsequent::Any,
        negconsequent::Any,
        info::NamedTuple = (;),
    )
        posconsequent isa PasoDecisionTree && (posconsequent = root(posconsequent))
        negconsequent isa PasoDecisionTree && (negconsequent = root(negconsequent))
        return PasoDecisionTree(Branch(antecedent, posconsequent, negconsequent, info))
    end
end

# ---------------------------------------------------------------------------- #
#                          nuova funzione solemodel                            #
# ---------------------------------------------------------------------------- #
# Ora potremmo immaginare una funzione solemodel che evita di costruire le varie strutture 'info'
# e si limita a costruire l'albero.
# l'info root verrà creato successivamente dalla funzione 'apply!'
# con le predizioni sul dataset di test

function get_featurenames(tree::Union{DT.Ensemble, DT.InfoNode})
    if !hasproperty(tree, :info)
        throw(ArgumentError("Please provide featurenames."))
    end
    return tree.info.featurenames
end
get_classlabels(tree::Union{DT.Ensemble, DT.InfoNode})::Vector{<:SoleModels.Label} = tree.info.classlabels

function get_condition(featid, featval, featurenames)
    test_operator = (<)
    feature = isnothing(featurenames) ? VariableValue(featid) : VariableValue(featid, featurenames[featid])
    return ScalarCondition(feature, test_operator, featval)
end

function pasomodel(
    model          :: DT.Ensemble{T,O};
    featurenames   :: Vector{Symbol}=Symbol[],
    weights        :: Vector{<:Number}=Number[],
    classlabels    :: AbstractVector{<:SoleModels.Label}=SoleModels.Label[],
    keep_condensed :: Bool=false,
    parity_func    :: Base.Callable=x->first(sort(collect(keys(x))))
)::DecisionEnsemble where {T,O}
    isempty(featurenames) && (featurenames = get_featurenames(model))
    # evito di costruire l'info
    # if keep_condensed && !isempty(classlabels)
    #     info = (;
    #         apply_preprocess=(y->O(findfirst(x -> x == y, classlabels))),
    #         apply_postprocess=(y->classlabels[y]),
    #     )
    #     keep_condensed = !keep_condensed
    # else
        info = (;)
    # end

    trees = map(t -> pasomodel(t, featurenames; classlabels), model.trees)
    # anche qui: l'idea è di costruire l'info solo in apply!
    # info = merge(info, (;
    #         featurenames=featurenames, 
    #         supporting_predictions=vcat([t.info[:supporting_predictions] for t in trees]...),
    #         supporting_labels=vcat([t.info[:supporting_labels] for t in trees]...),
    #     )
    # )

    isnothing(weights) ?
        DecisionEnsemble{O}(trees, info; parity_func) :
        DecisionEnsemble{O}(trees, weights, info; parity_func)
end

function pasomodel(
    tree           :: DT.InfoNode{T,O};
    featurenames   :: Vector{Symbol}=Symbol[],
    keep_condensed :: Bool=false,
)::PasoDecisionTree where {T,O}
    isempty(featurenames) && (featurenames = get_featurenames(tree))
    classlabels  = hasproperty(tree.info, :classlabels) ? get_classlabels(tree) : SoleModels.Label[]

    root, info = begin
        if keep_condensed
            root = pasomodel(tree.node, featurenames; classlabels)
            # anche qui: niente info
            # info = (;
            #     apply_preprocess=(y -> UInt32(findfirst(x -> x == y, classlabels))),
            #     apply_postprocess=(y -> classlabels[y]),
            # )
            info = (;)
            root, info
        else
            root = pasomodel(tree.node, featurenames; classlabels)
            info = (;)
            root, info
        end
    end

    # info = merge(info, (;
    #         featurenames=featurenames,
    #         supporting_predictions=root.info[:supporting_predictions],
    #         supporting_labels=root.info[:supporting_labels],
    #     )
    # )

    PasoDecisionTree(root, info)
end

function pasomodel(
    tree         :: DT.Node,
    featurenames :: Vector{Symbol};
    classlabels  :: AbstractVector{<:SoleModels.Label}=SoleModels.Label[],
)::Branch
    cond = get_condition(tree.featid, tree.featval, featurenames)
    antecedent = Atom(cond)
    lefttree  = pasomodel(tree.left, featurenames; classlabels )
    righttree = pasomodel(tree.right, featurenames; classlabels )

    # a costo di ripetermi...
    # info = (;
    #     supporting_predictions = [lefttree.info[:supporting_predictions]..., righttree.info[:supporting_predictions]...],
    #     supporting_labels = [lefttree.info[:supporting_labels]..., righttree.info[:supporting_labels]...],
    # )
    info = (;)

    return Branch(antecedent, lefttree, righttree, info)
end

function pasomodel(
    tree         :: DT.Leaf,
                 :: Vector{Symbol};
    classlabels  :: AbstractVector{<:SoleModels.Label}=SoleModels.Label[]
)::ConstantModel
    prediction, labels = isempty(classlabels) ? 
        (tree.majority, tree.values) : 
        (classlabels[tree.majority], classlabels[tree.values])

    # ci siamo capiti
    # info = (;
    #     supporting_predictions = fill(prediction, length(labels)),
    #     supporting_labels = labels,
    # )
    info = (;)

    SoleModels.ConstantModel(prediction, info)
end

# ---------------------------------------------------------------------------- #
#                           nuova funzione apply!                              #
# ---------------------------------------------------------------------------- #
# ERROR: type NamedTuple has no field supporting_labels
# bisogna modificare leggermente l'apply! esistente in modo che non vada a cercare
# field che abbiamo volutamente lasciato vuoti
# e che accetti, almeno per ora, PasoDecisionTree
pasoroot(m::PasoDecisionTree) = m.root

function pasoapply!(
    m::PasoDecisionTree,
    d::SoleModels.AbstractInterpretationSet,
    y::AbstractVector;
    mode = :replace,
    leavesonly = false,
    kwargs...
)
    y = SoleModels.__apply_pre(m, d, y)
    preds = pasoapply!(pasoroot(m), d, y;
        mode = mode,
        leavesonly = leavesonly,
        kwargs...
    )
    return SoleModels.__apply!(m, mode, preds, y, leavesonly)
end

function pasoapply!(
    m::Branch,
    d::SoleModels.AbstractInterpretationSet,
    y::AbstractVector;
    check_args::Tuple = (),
    check_kwargs::NamedTuple = (;),
    mode = :replace,
    leavesonly = false,
    # show_progress = true,
    kwargs...
)
    # @assert length(y) == ninstances(d) "$(length(y)) == $(ninstances(d))"
    if mode == :replace
        # non è più  necessario: si parte già con tutto vuoto
        # SoleModels.recursivelyemptysupports!(m, leavesonly)
        mode = :append
    end
    checkmask = SoleModels.checkantecedent(m, d, check_args...; check_kwargs...)
    preds = Vector{outputtype(m)}(undef,length(checkmask))
    @sync begin
        if any(checkmask)
            l = Threads.@spawn apply!(
                posconsequent(m),
                slicedataset(d, checkmask; return_view = true),
                y[checkmask];
                check_args = check_args,
                check_kwargs = check_kwargs,
                mode = mode,
                leavesonly = leavesonly,
                kwargs...
            )
        end
        ncheckmask = (!).(checkmask)
        if any(ncheckmask)
            r = Threads.@spawn apply!(
                negconsequent(m),
                slicedataset(d, ncheckmask; return_view = true),
                y[ncheckmask];
                check_args = check_args,
                check_kwargs = check_kwargs,
                mode = mode,
                leavesonly = leavesonly,
                kwargs...
            )
        end
        if any(checkmask)
            preds[checkmask] .= fetch(l)
        end
        if any(ncheckmask)
            preds[ncheckmask] .= fetch(r)
        end
    end
    return SoleModels.__apply!(m, mode, preds, y, leavesonly)
end



# ---------------------------------------------------------------------------- #
#                        nuova SoleXplorer train_test                          #
# ---------------------------------------------------------------------------- #
# Per poter fare dei benchmark comparativi, preferirei scrivere una nuova funzione train_test
# di Sole, che usa le nuove funzioni

function xplorer_apply(
    ds :: SoleXplorer.DecisionTreeApply,
    X  :: AbstractDataFrame,
    y  :: AbstractVector
)
    featurenames = MLJ.report(ds.mach).features
    solem        = pasomodel(MLJ.fitted_params(ds.mach).tree; featurenames)
    # logiset      = scalarlogiset(X, allow_propositional = true)
    # apply!(solem, X, y)
    return solem
end

function _paso_test(ds::SoleXplorer.EitherDataSet)::SoleXplorer.SoleModel
    n_folds   = length(ds.pidxs)
    solemodel = Vector{AbstractModel}(undef, n_folds)

    # TODO this can be parallelizable
    @inbounds @views for i in 1:n_folds
        train, test = get_train(ds.pidxs[i]), get_test(ds.pidxs[i])
        X_test, y_test = get_X(ds)[test, :], get_y(ds)[test]

        SoleXplorer.has_xgboost_model(ds) && SoleXplorer.set_watchlist!(ds, i)

        MLJ.fit!(ds.mach, rows=train, verbosity=0)
        solemodel[i] = xplorer_apply(ds, X_test, y_test)
    end

    return SoleXplorer.SoleModel(ds, solemodel)
end

function paso_test(args...; kwargs...)::SoleXplorer.SoleModel
    ds = SoleXplorer._setup_dataset(args...; kwargs...)
    _paso_test(ds)
end

paso_test(ds::SoleXplorer.AbstractDataSet)::SoleXplorer.SoleModel = _paso_test(ds)

# per completare l'opera dobbiamo scrivere i metodi di apply! che accettano PasoDecisionTree e PasoEnsemble

# Verifichiamo il corretto funzionamento
dsc = setup_dataset(
    Xc, yc;
    model=DecisionTreeClassifier(),
    resample=Holdout(shuffle=true),
        train_ratio=0.7,
        rng=Xoshiro(1),   
)
solemc = paso_test(dsc)
modelc = symbolic_analysis(
    dsc, solemc;
    extractor=InTreesRuleExtractor(),
    measures=(accuracy, log_loss, confusion_matrix, kappa)
)


train, test = get_train(ds.pidxs[1]), get_test(ds.pidxs[1])
X_test, y_test = get_X(ds)[test, :], get_y(ds)[test]
solemc = paso_test(dsc)
s = solemc.sole[1]
logiset      = scalarlogiset(X_test, allow_propositional = true)

pasoapply!(s, logiset, y_test)


# Esperimento condotto con Sole
@btime begin
    symbolic_analysis(
        Xc, yc,
        model=DecisionTreeClassifier(),
        resample=Holdout(shuffle=true),
        train_ratio=0.7,
        rng=Xoshiro(1),
        measures=(accuracy, kappa)
    )
end
# 444.172 μs (3370 allocations: 247.12 KiB)

# ---------------------------------------------------------------------------- #
#                     DecisionTree apply from DataFrame X                      #
# ---------------------------------------------------------------------------- #
# get_featid(s::Branch) = s.antecedent.value.metacond.feature.i_variable
# get_cond(s::Branch)   = s.antecedent.value.metacond.test_operator
# get_thr(s::Branch)    = s.antecedent.value.threshold

# function set_predictions(
#     info  :: NamedTuple,
#     preds :: Vector{T},
#     y     :: AbstractVector{S}
# )::NamedTuple where {T,S<:SoleModels.Label}
#     merge(info, (supporting_predictions=preds, supporting_labels=y))
# end

# function pasoapply!(
#     solem :: PasoDecisionEnsemble{O,T,A,W},
#     X     :: AbstractDataFrame,
#     y     :: AbstractVector;
#     suppress_parity_warning::Bool=false
# )::Nothing where {O,T,A,W}
#     predictions = permutedims(hcat([pasoapply(s, X, y) for s in get_models(solem)]...))
#     predictions = aggregate(solem, predictions, suppress_parity_warning)
#     solem.info  = set_predictions(solem.info, predictions, y)
#     return nothing
# end

# function pasoapply!(
#     solem :: PasoDecisionTree{T},
#     X     :: AbstractDataFrame,
#     y     :: AbstractVector{S}
# )::Nothing where {T, S<:SoleModels.Label}
#     predictions = [pasoapply(solem.root, x) for x in eachrow(X)]
#     solem.info  = set_predictions(solem.info, predictions, y)
#     return nothing
# end

# function pasoapply(
#     solebranch :: Branch{T},
#     X          :: AbstractDataFrame,
#     y          :: AbstractVector{S}
# ) where {T, S<:SoleModels.Label}
#     predictions     = SoleModels.Label[pasoapply(solebranch, x) for x in eachrow(X)]
#     solebranch.info = set_predictions(solebranch.info, predictions, y)
#     return predictions
# end

# function pasoapply(
#     solebranch :: Branch{T},
#     x          :: DataFrameRow
# )::T where T
#     featid, cond, thr = get_featid(solebranch), get_cond(solebranch), get_thr(solebranch)
#     feature_value     = x[featid]
#     condition_result  = cond(feature_value, thr)
    
#     return condition_result ?
#         pasoapply(solebranch.posconsequent, x) :
#         pasoapply(solebranch.negconsequent, x)
# end

# function pasoapply(leaf::ConstantModel{T}, ::DataFrameRow)::T where T
#     leaf.outcome
# end
