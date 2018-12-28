import os

path = '/Users/chen/Desktop/ecg/www.physionet.org/physiobank/database/mitdb'

folder = os.fsencode(path)

filenames = []

for file in os.listdir(folder):
    filename = os.fsdecode(file)
    if filename.endswith( ('.atr', '.dat', '.hea') ): # whatever file types you're using...
        filenames.append(path + '/' +filename)

filenames.sort() # now you have the filenames and can do something with them
for name in filenames:
    print(name)


