using Test
using SoleXplorer
using MLJ
using DataFrames, Random
const SX = SoleXplorer

# ---------------------------------------------------------------------------- #
#                                    natops                                    #
# ---------------------------------------------------------------------------- #
Xts, yts = load_arff_dataset("NATOPS")
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

modelts = symbolic_analysis(
    Xts, yts;
    model=ModalDecisionTree(),
    resample=Holdout(shuffle=true),
    rng=Xoshiro(1),
    association=Apriori(manual_items, [(gsupport, 0.5, 0.7)], [(gconfidence, 0.7, 0.7)]),
    measures=(accuracy, log_loss, confusion_matrix, kappa)      
)
@test modelts isa SX.ModelSet