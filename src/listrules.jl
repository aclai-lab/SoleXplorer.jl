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

# L'esperimento è eseguito sul classico dataset Iris, caricato tramite la macro offerta da MLJ.
Xc, yc = @load_iris
Xc = DataFrame(Xc)

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



# using SoleModels: RuleExtractor
# const Optional{T}   = Union{T, Nothing}
# const OptFloat64 = Optional{Float64}
# using SolePostHoc.RuleExtraction: intrees
# using SoleLogics

# using SoleXplorer
# using MLJ
# using DataFrames, Random
# const SX = SoleXplorer

# using SoleModels: _listrules
# using SoleModels: listrules

Xc, yc = @load_iris
Xc = DataFrame(Xc)

for s in 1:50
    dsc = setup_dataset(
        Xc, yc;
        model=DecisionTreeClassifier(),
        resample=Holdout(;shuffle=true),
        rng=Xoshiro(s),
    )
    solemc = train_test(dsc)

    model = solemc.sole[1]
    i = 1
    test = get_test(dsc.pidxs[i])
    X, y = get_X(dsc)[test, :], get_y(dsc)[test]
    rmodel = root(model)
    iii = SoleModels.info(rmodel)
    test_original = _listrules(rmodel)
    test_paso = _pasolistrules(rmodel, iii)

    @test length(test_original) == length(test_paso)
    for i in 1:length(test_paso)
        @test test_original[i].antecedent.grandchildren == test_paso[i].antecedent.grandchildren
        @test test_original[i].consequent.outcome == test_paso[i].consequent.outcome
    end
end

# ---------------------------------------------------------------------------- #
#                                   listrules                                  #
# ---------------------------------------------------------------------------- #
_pasolistrules(m::LeafModel{O}, iii; kwargs...) where {O} = [Rule{O}(⊤, m, iii)]

function _pasolistrules(
    m::Rule{O},
    iii;
    use_leftmostlinearform::Union{Nothing,Bool} = nothing,
    force_syntaxtree::Bool = false,
    kwargs...
) where {O}
    use_leftmostlinearform = !isnothing(use_leftmostlinearform) ? use_leftmostlinearform : false
    [begin
        φ = combine_antecedents(antecedent(m), antecedent(subrule), use_leftmostlinearform, force_syntaxtree)
        Rule{O}(φ, consequent(subrule), iii)
    end for subrule in _pasolistrules(consequent(m), iii; force_syntaxtree = force_syntaxtree, use_leftmostlinearform = use_leftmostlinearform, kwargs...)]
end

function _pasolistrules(
    m::Branch{O},
    iii;
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
        # normalize = false, TODO?
        normalize = normalize,
        normalize_kwargs = normalize_kwargs,
        scalar_simplification = false,
        force_syntaxtree = force_syntaxtree,
        min_confidence = min_confidence,
        min_coverage = min_coverage,
        min_ninstances = min_ninstances,
        kwargs...)
    # @show normalize, normalize_kwargs
    _subrules = []
    if isnothing(min_ninstances) || (haskey(iii, :supporting_labels) && length(info(m, :supporting_labels)) >= min_ninstances)
    # if (haskey(iii, :supporting_labels) && length(info(m, :supporting_labels)) >= min_ninstances) &&
    #     (haskey(iii, :supporting_labels) && length(info(m, :supporting_labels))/ntotinstances >= min_coverage)
        append!(_subrules, [(true,  r) for r in _pasolistrules(posconsequent(m), iii; subkwargs...)])
        append!(_subrules, [(false, r) for r in _pasolistrules(negconsequent(m), iii; subkwargs...)])
    end

    rules = map(((flag, subrule),)->begin
            # @show iii
            known_infokeys = [:supporting_labels, :supporting_predictions, :shortform, :this, :multipathformula]
            ks = setdiff(keys(iii), known_infokeys)
            if length(ks) > 0
                @warn "Dropping info keys: $(join(repr.(ks), ", "))"
            end

            _info = (;)
            if haskey(iii, :supporting_labels) && haskey(iii, :supporting_predictions)
                _info = merge((;), (;
                    supporting_labels = iii.supporting_labels,
                # ))
            # end
            # if haskey(iii, :supporting_predictions)
                # _info = merge((;), (;
                    supporting_predictions = iii.supporting_predictions,
                ))
            elseif (haskey(iii, :supporting_labels) != haskey(iii, :supporting_predictions))
                @warn "List rules encountered an unexpected case. Both " *
                    " supporting_labels and supporting_predictions are necessary for correctly computing performance metrics. "
            end

            antformula, using_shortform = begin
                if (use_shortforms && haskey(iii, :shortform))
                    iii[:shortform], true
                else
                    # Automatic flip.
                    smart_neg(f) = (f isa Atom && flip_atoms && SoleLogics.hasdual(f) ? SoleLogics.dual(f) : ¬f)
                    _antd = antecedent(m)
                    (flag ? _antd : smart_neg(_antd)), false
                end
            end
            antformula = force_syntaxtree ? tree(antformula) : antformula
            # @show using_shortform
            # @show antformula
            # @show typeof(subrule)

            if subrule isa LeafModel
                ant = antformula
                normalize && (ant = SoleLogics.normalize(ant; normalize_kwargs...))
                ant = _scalar_simplification(ant, scalar_simplification)
                subi = (;)
                # if use_shortforms
                #     subi = merge((;), (;
                #         shortform = ant
                #     ))
                # end
                Rule(ant, subrule, merge(iii, subi, _info))
            elseif subrule isa Rule
                ant = begin
                    if using_shortform
                        antformula
                    else
                        # Combine antecedents
                        φ = SoleModels.combine_antecedents(antformula, antecedent(subrule), use_leftmostlinearform, force_syntaxtree)
                        # @show 3
                        # @show φ
                        φ
                    end
                end
                # @show normalize, normalize_kwargs
                normalize && (ant = SoleLogics.normalize(ant; normalize_kwargs...))
                # @show 1
                # @show ant
                ant = SoleModels._scalar_simplification(ant, scalar_simplification)
                # @show 2
                # @show ant
                # readline()
                Rule(ant, consequent(subrule), merge(iii, _info))
            else
                error("Unexpected rule type: $(typeof(subrule)).")
            end
        end, _subrules)

    return rules
end