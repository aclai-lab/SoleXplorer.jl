using Test
using SoleXplorer
const SX = SoleXplorer

# using SoleData
using MLJ
using DataFrames, Random

@show Threads.nthreads()

# ---------------------------------------------------------------------------- #
#                                load dataset                                  #
# ---------------------------------------------------------------------------- #
natopsloader = SX.NatopsLoader()
Xts, yts = SX.load(natopsloader)

# ---------------------------------------------------------------------------- #
#                      realtions in modal decision trees                       #
# ---------------------------------------------------------------------------- #
dsts_ia = solexplorer(
    Xts, yts;
    model=ModalDecisionTree(; relations=:IA),
    resampling=Holdout(fraction_train=0.7, shuffle=true),
    seed=1,
    win=adaptivewindow(nwindows=3, overlap=0.3),
)

dsts_ia3 = solexplorer(
    Xts, yts;
    model=ModalDecisionTree(; relations=:IA3),
    resampling=Holdout(fraction_train=0.7, shuffle=true),
    seed=1,
    win=adaptivewindow(nwindows=3, overlap=0.3),
)

dsts_ia7 = solexplorer(
    Xts, yts;
    model=ModalDecisionTree(; relations=:IA7),
    resampling=Holdout(fraction_train=0.7, shuffle=true),
    seed=1,
    win=adaptivewindow(nwindows=3, overlap=0.3),
)

dsts_rcc5 = solexplorer(
    Xts, yts;
    model=ModalDecisionTree(; relations=:RCC5),
    resampling=Holdout(fraction_train=0.7, shuffle=true),
    seed=1,
    win=adaptivewindow(nwindows=3, overlap=0.3),
)

dsts_rcc8 = solexplorer(
    Xts, yts;
    model=ModalDecisionTree(; relations=:RCC8),
    resampling=Holdout(fraction_train=0.7, shuffle=true),
    seed=1,
    win=adaptivewindow(nwindows=3, overlap=0.3),
)

# ---------------------------------------------------------------------------- #
#                       realtions in modal random forest                       #
# ---------------------------------------------------------------------------- #
dsts_ia = solexplorer(
    Xts, yts;
    model=ModalRandomForest(; relations=:IA),
    resampling=Holdout(fraction_train=0.7, shuffle=true),
    seed=1,
    win=adaptivewindow(nwindows=3, overlap=0.3),
)

dsts_ia3 = solexplorer(
    Xts, yts;
    model=ModalRandomForest(; relations=:IA3),
    resampling=Holdout(fraction_train=0.7, shuffle=true),
    seed=1,
    win=adaptivewindow(nwindows=3, overlap=0.3),
)

dsts_ia7 = solexplorer(
    Xts, yts;
    model=ModalRandomForest(; relations=:IA7),
    resampling=Holdout(fraction_train=0.7, shuffle=true),
    seed=1,
    win=adaptivewindow(nwindows=3, overlap=0.3),
)

dsts_rcc5 = solexplorer(
    Xts, yts;
    model=ModalRandomForest(; relations=:RCC5),
    resampling=Holdout(fraction_train=0.7, shuffle=true),
    seed=1,
    win=adaptivewindow(nwindows=3, overlap=0.3),
)

dsts_rcc8 = solexplorer(
    Xts, yts;
    model=ModalRandomForest(; relations=:RCC8),
    resampling=Holdout(fraction_train=0.7, shuffle=true),
    seed=1,
    win=adaptivewindow(nwindows=3, overlap=0.3),
)
