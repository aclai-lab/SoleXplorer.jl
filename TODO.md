# ---------------------------------------------------------------------------- #
#                                documentazione                                #
# ---------------------------------------------------------------------------- #
Priorità massima
Serve un metodo preciso per buttarla giù.
Albi sicuramente potrà essere d'aiuto, ma anche Marco.

# ---------------------------------------------------------------------------- #
#                                  load/save                                   #
# ---------------------------------------------------------------------------- #
Importante, vedi di farlo entro Parigi

Edo mi ha suggerito Pkl da scikitlearn
Perry suggerisce serialize.jl

# ---------------------------------------------------------------------------- #
#                                  XgBoost ext                                 #
# ---------------------------------------------------------------------------- #
Ha un probabile bug in regressione, quando incontra alberi composti da una sola foglia
Bisogna testarlo e verificare come risolvere la questione.
POTREBBE ANDARE BENE, VERIFICA I TEST

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
#                                 bug di MLJ #2                                #
# ---------------------------------------------------------------------------- #
latinhypercube al momento attuale non va: la chiamata al metodo è sbagliata.
bisognerebbe verificarne il funzionamento nel test incluso in MLJ
e se non funziona aprire una PR in MLJ.
Andrebbe risolto.

# ---------------------------------------------------------------------------- #
#                            modal association rules                           #
# ---------------------------------------------------------------------------- #
da fare tassativamente entro ottobre
IN LAVORAZIONE

# ---------------------------------------------------------------------------- #
#                               feature selection                              #
# ---------------------------------------------------------------------------- #
da fare entro la laurea

# ---------------------------------------------------------------------------- #
#                                      gui                                     #
# ---------------------------------------------------------------------------- #
sarebbe bello averla impostata entro ottobre, ma ha priorità minore.
Da parlarne col Balbo

# ---------------------------------------------------------------------------- #
#                               ribilanciamento                                #
# ---------------------------------------------------------------------------- #
Guarda su MLJ se esiste

# ---------------------------------------------------------------------------- #
#                              peso delle istanze                              #
# ---------------------------------------------------------------------------- #
in MLJ si usa X, y, w. da implementare? verifica come funziona, magari viene messo da MLJ
direttamente dalla mach. Però bisogna implementarlo in setup_dataset e symbolic_analysis.
Vedi anche se MLJ oppure i modelli (Decision Tree...) usano un default nei pesi.
Prova con dei pesi a caso a vedere se cambia qualcosa.
Prova a guardare anche se c'è il modo di settare il peso sulle classi (tipo class rebalance, che se hai sbilanciate,
le rebilancia coi pesi, per non buttare via istanze).

# ---------------------------------------------------------------------------- #
#                                       rng                                    #
# ---------------------------------------------------------------------------- #
Propagalo anche a PostHoc

# ---------------------------------------------------------------------------- #
#                             apply e extractrules                             #
# ---------------------------------------------------------------------------- #
Bisogna pensare anche ai modelli non supervisionati: apply(m, X)

# ---------------------------------------------------------------------------- #
#                                   SoleModel                                  #
# ---------------------------------------------------------------------------- #
La struttura SoleModel potrebbe essere ridondante

# ---------------------------------------------------------------------------- #
#                                      MAS                                     #
# ---------------------------------------------------------------------------- #
Spiega a Mauro la questione delle finestre, IMPORTANTE.

# ---------------------------------------------------------------------------- #
#                                 Idea di Mauro                                #
# ---------------------------------------------------------------------------- #
@forward Apriori apriori
dovrebbe essere nel pacchetto lazy.jl
cerca in sole chi lo usa