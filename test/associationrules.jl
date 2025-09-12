using Test
using SoleXplorer
using MLJ
using DataFrames, Random
using ModalAssociationRules
const SX = SoleXplorer

# ---------------------------------------------------------------------------- #
#                                    natops                                    #
# ---------------------------------------------------------------------------- #
natopsloader = NatopsLoader()
Xts, yts = SX.load(natopsloader)
X_have_command, y_have_command = Xts[1:30, :], yts[1:30]

# make a vector of item, that will be the initial state of the mining machine
manual_p = Atom(ScalarCondition(VariableMin(1), >, -0.5))
manual_q = Atom(ScalarCondition(VariableMin(2), <=, -2.2))
manual_r = Atom(ScalarCondition(VariableMin(3), >, -3.6))

manual_lp = box(IA_L)(manual_p)
manual_lq = diamond(IA_L)(manual_q)
manual_lr = box(IA_L)(manual_r)

manual_items = Vector{Item}([
    manual_p, manual_q, manual_r, manual_lp, manual_lq, manual_lr])

manual_v2 = [
    Atom(ScalarCondition(VariableMin(4), >=, 1))
    Atom(ScalarCondition(VariableMin(4), >=, 1.8))
    Atom(ScalarCondition(VariableMin(5), >=, -0.5))
    Atom(ScalarCondition(VariableMax(6), >=, 0))
]
manual_v2_modal = vcat(
    manual_v2,
    (manual_v2)[1] |> diamond(IA_L)
) |> Vector{Item}

# Driver

EXPERIMENTKEYS = (
    :items,
    :itemsetmeasures,
    :rulemeasures,
    :expkwargs
);

EXPERIMENTVALUES = (
    (
        Vector{Item}([manual_p, manual_q, manual_lp, manual_lq]),
        [(gsupport, 0.1, 0.1)],
        [(gconfidence, 0.2, 0.2)],
        ()
    ),

    (
        Vector{Item}([manual_p, manual_q, manual_r]),
        [(gsupport, 0.5, 0.7)],
        [(gconfidence, 0.7, 0.7)],
        ()
    ),

    (
        Vector{Item}([manual_lp, manual_lq, manual_lr]),
        [(gsupport, 0.8, 0.8)],
        [(gconfidence, 0.7, 0.7)],
        ()
    ),

    (
        Vector{Item}([manual_q, manual_r, manual_lp, manual_lr]),
        [(gsupport, 0.4, 0.4)],
        [(gconfidence, 0.7, 0.7)],
        Dict(:itemset_policies => Function[])
    ),

    (
        manual_v2_modal,
        [(gsupport, 0.1, 0.1)],
        [(gconfidence, 0.1, 0.1)],
        ()
    ),

    (
        manual_v2_modal,
        [(gsupport, 0.5, 0.5)],
        [
            (gconfidence, 0.5, 0.5),
            (glift, 0.5, 0.5),          # [-∞,+∞]
            (gconviction, 1.0, 1.0),    # [0,+∞]
            (gleverage, -0.25, -0.25),  # [-0.25,0.25]
        ],
        ()
    ),

    (
        Vector{Item}([manual_p, manual_q, manual_lp, manual_lq]),
        [(gsupport, 0.1, 0.1)],
        [(gconfidence, 0.0, 0.0)],
        ()
    )
);

EXPERIMENTS = [
    NamedTuple{EXPERIMENTKEYS}(values)
    for values in (EXPERIMENTVALUES)
];

modelts = symbolic_analysis(Xts, yts; model=ModalDecisionTree())
@test modelts isa SX.ModelSet

X1 = scalarlogiset(get_X(dsetup(modelts)))

# algorithms to be tested
SX_ALGORITHMS = [Apriori, FPGrowth, Eclat]
MAS_ALGORITHMS = [apriori, fpgrowth, eclat]
ALGORITHMS = collect(zip(SX_ALGORITHMS, MAS_ALGORITHMS))

printstyled("Testing: $([a |> string for a in ALGORITHMS])\n", color=:green)

for (nth,exp) in enumerate(EXPERIMENTS)
    for algo in ALGORITHMS
        printstyled("Running experiment SoleXplorer $(nth), $(algo[2])\n", color=:green)
        symbolic_analysis!(
            modelts;
            association=algo[1](
            exp.items,
            exp.itemsetmeasures,
            exp.rulemeasures;
            exp.expkwargs...
            )
        )
        sx_associations = associations(modelts)

        printstyled("Running experiment MAS $(nth), $(algo[2])\n", color=:green)
        mas_associations = Miner(
            X1 |> deepcopy,
            algo[2],
            exp.items,
            exp.itemsetmeasures,
            exp.rulemeasures;
            exp.expkwargs...
        )
        mine!(mas_associations)

        @test sx_associations == arules(mas_associations)
    end
end

# Other, manual tests
model_command = symbolic_analysis(X_have_command, y_have_command; model=ModalDecisionTree())
@test modelts isa SX.ModelSet

X2 = scalarlogiset(get_X(model_command.ds))

_MANUALEXP = EXPERIMENTS[5]

symbolic_analysis!(
    model_command;
    association=FPGrowth(
    _MANUALEXP.items,
    _MANUALEXP.itemsetmeasures,
    _MANUALEXP.rulemeasures;
    _MANUALEXP.expkwargs...)
)
sx_fpgrowth = associations(model_command)

fpgrowth_miner = Miner(
    X2 |> deepcopy,
    fpgrowth,
    _MANUALEXP.items,
    _MANUALEXP.itemsetmeasures,
    _MANUALEXP.rulemeasures;
    _MANUALEXP.expkwargs...
)
mine!(fpgrowth_miner)
mas_fpgrowth = arules(fpgrowth_miner)

@test sx_fpgrowth == mas_fpgrowth

# ---------------------------------------------------------------------------- #
#                                     iris                                     #
# ---------------------------------------------------------------------------- #
Xc, yc = @load_iris
Xc = DataFrame(Xc)


X = scalarlogiset(Xc;
    relations=AbstractRelation[],
    conditions=Vector{ScalarMetaCondition}(
        collect(Iterators.flatten([
            [ScalarMetaCondition(f, <=) for f in VariableMin.(1:4)],
            [ScalarMetaCondition(f, >=) for f in VariableMin.(1:4)],
            [ScalarMetaCondition(f, <=) for f in VariableMax.(1:4)],
            [ScalarMetaCondition(f, >=) for f in VariableMax.(1:4)],
        ]))
    )
)

_1_items = Vector{Item}(Atom.([
    ScalarCondition(VariableMin(1), >=,  5)
    ScalarCondition(VariableMin(2), <=,  4)
    ScalarCondition(VariableMin(3), <=,  4)
    ScalarCondition(VariableMin(4), <=,  2)
]))

# 1st comparison
_1_itemsetmeasures = [(gsupport, 0.8, 0.8)]
_1_rulemeasures = [(gconfidence, 0.7, 0.7)]

modelc = symbolic_analysis(Xc, yc)

sx_apriori = symbolic_analysis!(modelc; association=Apriori(_1_items, _1_itemsetmeasures, _1_rulemeasures))
sx_fpgrowth = symbolic_analysis!(modelc; association=FPGrowth(_1_items, _1_itemsetmeasures, _1_rulemeasures))

apriori_miner = Miner(X, apriori, _1_items, _1_itemsetmeasures, _1_rulemeasures)
fpgrowth_miner = Miner(X, fpgrowth, _1_items, _1_itemsetmeasures, _1_rulemeasures)
mine!(apriori_miner)
mine!(fpgrowth_miner)

@test associations(sx_fpgrowth) == arules(fpgrowth_miner)
@test Set(associations(sx_apriori)) == Set(arules(apriori_miner))

# 2nd comparison
_2_itemsetmeasures = [(gsupport, 0.9, 0.4)]
_2_rulemeasures = [(gconfidence, 0.3, 0.5)]

sx_apriori = symbolic_analysis!(modelc; association=Apriori(_1_items, _2_itemsetmeasures, _2_rulemeasures))
sx_fpgrowth = symbolic_analysis!(modelc; association=FPGrowth(_1_items, _2_itemsetmeasures, _2_rulemeasures))

apriori_miner = Miner(X, apriori, _1_items, _2_itemsetmeasures, _2_rulemeasures)
fpgrowth_miner = Miner(X, fpgrowth, _1_items, _2_itemsetmeasures, _2_rulemeasures)
mine!(apriori_miner)
mine!(fpgrowth_miner)

@test associations(sx_fpgrowth) == arules(fpgrowth_miner)
@test Set(associations(sx_apriori)) == Set(arules(apriori_miner))

# 3rd comparison
_3_itemsetmeasures = [(gsupport, 0.1, 0.1)]
_3_rulemeasures = [(gconfidence, 0.1, 0.1)]

sx_apriori = symbolic_analysis!(modelc; association=Apriori(_1_items, _3_itemsetmeasures, _3_rulemeasures))
sx_fpgrowth = symbolic_analysis!(modelc; association=FPGrowth(_1_items, _3_itemsetmeasures, _3_rulemeasures))

apriori_miner = Miner(X, apriori, _1_items, _3_itemsetmeasures, _3_rulemeasures)
fpgrowth_miner = Miner(X, fpgrowth, _1_items, _3_itemsetmeasures, _3_rulemeasures)
mine!(apriori_miner)
mine!(fpgrowth_miner)

@test associations(sx_fpgrowth) == arules(fpgrowth_miner)
@test Set(associations(sx_apriori)) == Set(arules(apriori_miner))

# ---------------------------------------------------------------------------- #
#                   example of a complete symbolic analysis                    #
# ---------------------------------------------------------------------------- #
range = SX.range(:eta; lower=0.2, upper=0.7)
_items = Vector{Item}(Atom.([
    ScalarCondition(VariableMin(1), >=,  5)
    ScalarCondition(VariableMin(2), <=,  4)
    ScalarCondition(VariableMin(3), <=,  4)
    ScalarCondition(VariableMin(4), <=,  2)
]))
_itemsetmeasures = [(gsupport, 0.9, 0.4)]
_rulemeasures = [(gconfidence, 0.3, 0.5)]
modelts = symbolic_analysis(
    Xts, yts;
    model=XGBoostClassifier(early_stopping_rounds=20),
    resample=CV(nfolds=5, shuffle=true),
    valid_ratio=0.2,
    rng=Xoshiro(1),
    tuning=GridTuning(resolution=10, resampling=CV(nfolds=3), range=range, measure=accuracy, repeats=2),
    extractor=InTreesRuleExtractor(),
    association=FPGrowth(_items, _itemsetmeasures, _rulemeasures),
    measures=(accuracy, log_loss, confusion_matrix, kappa)      
)
@test modelts isa SX.ModelSet
