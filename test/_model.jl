# ---------------------------------------------------------------------------- #
#                           testing validate model                             #
# ---------------------------------------------------------------------------- #
@test_throws ArgumentError symbolic_analysis(
    Xc, yc;
    model=(;params=(min_samples_leaf=1,n_subfeatures=3)),
)

@test_throws ArgumentError symbolic_analysis(
    Xc, yc;
    model=(;type=decisiontreeclassifier, invalid_param=1),
)

@test_throws ArgumentError symbolic_analysis(
    Xc, yc;
    model=(;type=decisiontreeclassifier, invalid_param=1),
)

@test_throws MethodError symbolic_analysis(
    Xc, yc;
    model=(;type=decisiontreeclassifier, params=(invalid=1,)),
)

# ---------------------------------------------------------------------------- #
#                         testing validate parameters                          #
# ---------------------------------------------------------------------------- #
test = symbolic_analysis(
    Xc, yc;
    model=(;type=modaldecisiontree),
)

test = symbolic_analysis(
    Xc, yc;
    model=(type=modaldecisiontree, params=(;conditions=[maximum])),
)

test = symbolic_analysis(
    Xc, yc;
    model=(;type=decisiontreeclassifier),
)

test = symbolic_analysis(
    Xc, yc;
    model=(;type=decisiontreeclassifier),
    preprocess=(;rng=Xoshiro(1)),
)