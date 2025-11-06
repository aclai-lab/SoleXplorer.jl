using Test
using SoleXplorer
const SX = SoleXplorer

using MLJ
using DataFrames, Random
using Downloads, DelimitedFiles

# datasets to tests
datasets = [
    ("banknote", 1:4, 5),
    ("breast_cancer", 2:9, 1),
    ("car", 1:6, 7),
    ("cryotherapy", 1:6, 7),
    ("diabets", 1:8, 9),
    ("divorce", 1:54, 55),
    ("haberman", 1:3, 4),
    ("hayes-roth", 2:5, 6),
    ("heart", 1:13, 14),
    ("house-votes", 2:17, 1),
    ("htru", 1:8, 9),
    ("mammographic_masses", 1:5, 6),
    ("monks-1", 2:7, 1),
    ("monks-3", 2:7, 1),
    ("mushroom", 2:5, 1),
    ("occupancy", 3:7, 8),
    ("penguins", 3:7, 1),
    ("seeds", 1:7, 8),
    ("soybean-small", 1:35, 36),
    ("statlog", 1:13, 14),
    ("tictactoe", 1:9, 10),
    ("urinary-d1", 2:6, 7),
    ("urinary-d2", 2:6, 8),
    ("veichle_E", 1:18, 19),
]

# base URL for datasets on PasoStudio73 GitHub
base_url = "https://raw.githubusercontent.com/PasoStudio73/datasets/refs/heads/main/"

# download datasets
for (dataset_name, _, _) in datasets
    url = base_url * dataset_name * ".data"
    local_path = joinpath(@__DIR__, "datasets", dataset_name * ".data")  
    # create directory if it doesn't exist
    mkpath(dirname(local_path))
    # download file
    Downloads.download(url, local_path)
end

filepath = [joinpath(@__DIR__, "datasets/" * "$(d[1])" * ".data") for d in datasets]
data = [DelimitedFiles.readdlm(f, ',') for f in filepath]
dataX = [DataFrame(d[:, datasets[i][2]], :auto) for (i,d) in enumerate(data)]
datay = [d[:, datasets[i][3]] for (i,d) in enumerate(data)]

analysis_collection = [
    symbolic_analysis(
        X, y;
        model=XGBoostClassifier(max_depth=3, early_stopping_rounds=20),
        seed=11,
        valid_ratio=0.2,
        measures=(accuracy,)
    ) for (X, y) in zip(dataX, datay)
]

for modelc in analysis_collection
    @test modelc isa SX.ModelSet
end
