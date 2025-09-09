using ZipArchives, CSV, DataFrames
using AudioReader

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

########################################################################################################

audio_data = zip_readentry(archive, filepath)
# Create temporary buffer and read with AudioReader
temp_io = IOBuffer(audio_data)
return AudioReader.load(temp_io)
