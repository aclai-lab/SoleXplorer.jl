# level 1:
# I have a dataset composed of a matrix (or dtataframe) of measures, and a vector of labels.
# I don't know if there's something interesting inside,
# and I'm not very into ML. Let's play with SoleXplorer.

using SoleXplorer

natopsloader = NatopsLoader()
X, y = SoleXplorer.load(natopsloader)

model = symbolic_analysis(X, y);

show_measures(model)