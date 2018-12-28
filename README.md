# Time-series-machine-learning
## Data
Search data in link: https://www.physionet.org/cgi-bin/atm/ATM
![link](/imageforReadme/datasearch.png)
Select "MIT-BIH Arrhythmia Database (mitdb)" in "Input: Database" block. \
The link to the dataset is on the right side of the blue bar under the search block:
![datapagelink](/imageforReadme/downloaddatalink.png)
If you have "wget", you can download data by type 
```
wget -r -np http://www.physionet.org/physiobank/database/mitdb/
```
in terminal. \
Reminder: Delete the 102-0.atr file in the mitdb dataset for prevent from program failure
Type
```
python /Users/chen/Desktop/Time-series-machine-learning/onedimensionCGAN.py /path/to/mitdb
```
to run the CGAN for ECG data. \
The generated ecg would be in /cgansamplesingle.

