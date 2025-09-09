using ZipArchives

file="/home/paso/Documents/Datasets/Respiratory_DB.zip"

data = read(file)
archive = ZipReader(data)

entries = zip_names(archive)

wav_files = filter(entry -> endswith(entry, ".wav"), entries)