# ---------------------------------------------------------------------------- #
#                                documentazione                                #
# ---------------------------------------------------------------------------- #
Priorità massima
Serve un metodo preciso per buttarla giù.
Albi sicuramente potrà essere d'aiuto, ma anche Marco.

# ---------------------------------------------------------------------------- #
#                                  XgBoost ext                                 #
# ---------------------------------------------------------------------------- #
Ha un probabile bug in regressione, quando incontra alberi composti da una sola foglia
Bisogna testarlo e verificare come risolvere la questione.

# ---------------------------------------------------------------------------- #
#                                 bug di MLJ #1                                #
# ---------------------------------------------------------------------------- #
modelts = symbolic_analysis(
    Xts, yts;
    model=ModalDecisionTree(),
    resample=(type=CV(;nfolds=4), rng=Xoshiro(1)),
    measures=(log_loss, accuracy, kappa)
)
@test modelts isa SX.ModelSet

resultsts = symbolic_analysis(
    Xts, yts;
    model=XGBoostClassifier(),
    resample=(type=TimeSeriesCV(nfolds=5), rng=Xoshiro(1)),
    win=AdaptiveWindow(nwindows=3, relative_overlap=0.3),
    modalreduce=mean,
    features=[maximum, minimum],
    measures=(accuracy, log_loss, confusion_matrix, kappa)
)

questo non va a causa del fatto che il resample non mischia le classi
quindi potrebbe capitare che un resample non abbia tutte le classi e
quindi quando si va a costruire la matrice di confusione, viene restituita una matrice 
di dimensione "classi viste" e quando vengono sommate da errore perchè non si possono sommare
matrici di dimensione differente.
Andrebbe risolto, situazione abbastanza grave.

# ---------------------------------------------------------------------------- #
#                        features vettore anzichè tupla                        #
# ---------------------------------------------------------------------------- #
tutti i parametri sono tra (), tranne features, tra []
non è carino

# ---------------------------------------------------------------------------- #
#                                 bug di MLJ #2                                #
# ---------------------------------------------------------------------------- #
latinhypercube al momento attuale non va: la chiamata al metodo è sbagliata.
bisognerebbe verificarne il funzionamento nel test incluso in MLJ
e se non funziona aprire una PR in MLJ.
Andrebbe risolto.

# ---------------------------------------------------------------------------- #
#                                    posthoc                                   #
# ---------------------------------------------------------------------------- #
bisognerebbe ridefinire i metodi con cui si chiamano gli estrattori, 
per renderli uguali alle chiamate degli altri parametri di sole.
Banalmente, uguali a comne si chiamano i modelli, cioè
quando creo la struttura posso inserirne i parametri.
Questo ad oggi non è possibile in PostHoc: i parametri vengono passati
con la funzione modalextractrules.
Si può fare, con poco, qualcosa di più idiomatico.
Da parlarne con Marco.

# ---------------------------------------------------------------------------- #
#                            modal association rules                           #
# ---------------------------------------------------------------------------- #
da fare tassativamente entro ottobre

# ---------------------------------------------------------------------------- #
#                               feature selection                              #
# ---------------------------------------------------------------------------- #
da fare entro la laurea

# ---------------------------------------------------------------------------- #
#                                      gui                                     #
# ---------------------------------------------------------------------------- #
sarebbe bello averla impostata entro ottobre, ma ha priorità minore.
Da parlarne col Balbo