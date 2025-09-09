using ZipArchives

file="/home/paso/Documents/Datasets/Respiratory_DB.zip"

archive = read(file) |> ZipReader
entries = zip_names(archive)

# there are two identical sub folder in zip, get rid of one of these
respiratory_files = filter(entry -> contains(entry, "respiratory_sound_database"), entries)
wav_files = filter(entry -> endswith(entry, ".wav"), respiratory_files)
csv_file  = filter(entry -> endswith(entry, ".csv"), respiratory_files)