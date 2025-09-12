using ZipArchives, CSV, JLD2, DataFrames
using StatsBase, Random, MLJ
using AudioReader
using Audio911
using SoleXplorer

# ---------------------------------------------------------------------------- #
#                              audio utilities                                 #
# ---------------------------------------------------------------------------- #
# load wav files
function get_audio(archive, filepath)
    audio_data = zip_readentry(archive, filepath)
    # write to temporary file
    temp_file = tempname() * ".wav"
    write(temp_file, audio_data)
    # load with AudioReader
    audio = AudioReader.load(temp_file; sr=8000, norm=true, mono=true)
    # clean up temporary file
    rm(temp_file)
    return audio
end

# audio processing pipeline
function audio_pipeline(audio)
    win = Audio911.MovingWindow(window_size=256, window_step=128)
    type = (:hann, :periodic)
    frames = get_frames(audio; win, type)

    stft = get_stft(frames; spectrum_type=:magnitude)
    mel_spec = get_melspec(stft)
    # return mel_spec[1][1:13,:]
    return mel_spec
end

# ---------------------------------------------------------------------------- #
#                          load and setup dataset                              #
# ---------------------------------------------------------------------------- #
# path to already downloaded dataset
file="/home/paso/Documents/Datasets/Respiratory_DB.zip"

# open zip file
archive = read(file) |> ZipReader
entries = zip_names(archive)

# there are two identical sub folder in zip, get rid of one of these
respiratory_files = filter(entry -> contains(entry, "respiratory_sound_database"), entries)

# filter files
wav_files = filter(entry -> endswith(entry, ".wav"), respiratory_files)
csv_file  = filter(entry -> endswith(entry, ".csv"), respiratory_files)

# extract and read the CSV file
csv_data = CSV.read(zip_readentry(archive, first(csv_file)), DataFrame; header=false)

# create dictionary from DataFrame
patient_dict = Dict(csv_data[!, 1] .=> csv_data[!, 2])

# create DataFrame from wav_files
X = DataFrame(
    audio = [get_audio(archive, file) for file in wav_files]
)

# add patient condition from the dictionary
patient_id = [parse(Int, split(basename(file), "_")[1]) for file in wav_files]
y = [get(patient_dict, id, "Unknown") for id in patient_id]

# ---------------------------------------------------------------------------- #
#                              serialize result                                #
# ---------------------------------------------------------------------------- #
# save data using JLD2
# jldsave("respiratory_data.jld2"; X=X, y=y)

# ---------------------------------------------------------------------------- #
#      serialize specifically for Notte dei Ricercatori and JuliaCon 2025      #
# ---------------------------------------------------------------------------- #
# create DataFrame from X.audio including patient id
wav_df = DataFrame(
    audio = X.audio
)
# add patient condition from the dictionary
wav_df.condition = [get(patient_dict, id, "Unknown") for id in patient_id]

# filter healthy and pneumonia patients
healthy_patients = wav_df[wav_df.condition   .== "Healthy", :]
pneumonia_patients = wav_df[wav_df.condition .== "Pneumonia", :]

# select the smallest to balance classes
min_samples = min(nrow(healthy_patients), nrow(pneumonia_patients))

# merge datasets
merged_df = vcat(healthy_patients[1:min_samples,:], pneumonia_patients[1:min_samples,:])

# create a DataFrame with only the audio column
audio_only_df = select(merged_df, :audio)

# save data using JLD2
jldsave("respiratory_juliacon2025_wav.jld2"; X=audio_only_df, y=merged_df.condition)

# ---------------------------------------------------------------------------- #
#                                  load data                                   #
# ---------------------------------------------------------------------------- #
data          = JLD2.load("respiratory_juliacon2025_wav.jld2")
audio_ds      = data["X"]
conditions_ds = data["y"]

# ---------------------------------------------------------------------------- #
#                                   audio911                                   #
# ---------------------------------------------------------------------------- #
# collect audio features
raw_audio_features = [audio_pipeline(row.audio)[1] for row in eachrow(audio_ds)]
freqs = Int.(floor.(audio_pipeline(audio_ds[1,1])[3]))

# create vector where each index contains the mode of each row
row_modes = [[mean(i) for i in eachrow(matrix)] for matrix in raw_audio_features]

# reduce in matrix
audio_features = reduce(hcat, row_modes)'

# Create DataFrame with mel feature column names
audio_features_df = DataFrame(audio_features, ["$(freqs[i])hz" for i in 1:size(audio_features, 2)])

# ---------------------------------------------------------------------------- #
#                              serialize result                                #
# ---------------------------------------------------------------------------- #
# save data using JLD2
jldsave("respiratory_juliacon2025.jld2"; X=audio_features_df, y=conditions_ds)

# ---------------------------------------------------------------------------- #
#                                  test data                                   #
# ---------------------------------------------------------------------------- #
data  = JLD2.load("respiratory_juliacon2025.jld2")
Xc = data["X"]
# this is imperative: some algos accept only categorical value
# TODO automate in solexplorer if it's a CLabel ?
yc = MLJ.CategoricalArray{String,1,UInt32}(data["y"])

Xlight = Xc[:, 3:18]

# ---------------------------------------------------------------------------- #
#                                sole xplorer                                  #
# ---------------------------------------------------------------------------- #
dtc = symbolic_analysis(
    Xc, yc;
    model=DecisionTreeClassifier(),
    resample=StratifiedCV(nfolds=20, shuffle=true),
    rng=Xoshiro(12345),
    # extractor=InTreesRuleExtractor(),
    measures=(accuracy,)      
)

rfc = symbolic_analysis(
    Xlight, yc;
    model=RandomForestClassifier(n_trees=30),
    resample=StratifiedCV(nfolds=20, shuffle=true),
    rng=Xoshiro(12345),
    # extractor=InTreesRuleExtractor(),
    measures=(accuracy,)      
)

# ---------------------------------------------------------------------------- #
#                              serialize forest                                #
# ---------------------------------------------------------------------------- #
# save data using JLD2
jldsave("forest_juliacon2025.jld2"; X=rfc)

# ---------------------------------------------------------------------------- #
#                                  test data                                   #
# ---------------------------------------------------------------------------- #
data  = JLD2.load("forest_juliacon2025.jld2")
test_model = data["X"]

xgb = symbolic_analysis(
    Xlight, yc;
    model=XGBoostClassifier(early_stopping_rounds=20),
    resample=StratifiedCV(nfolds=20, shuffle=true),
    valid_ratio=0.2,
    rng=Xoshiro(12345),
    # extractor=InTreesRuleExtractor(),
    measures=(accuracy,)      
)