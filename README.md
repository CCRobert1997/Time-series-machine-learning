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
Run rddata.m file in matlab, referance to https://github.com/Aiwiscal/ECG-ML-DL-Algorithm-Matlab.
Type 
```
plot(TIME, M(:,1));grid on;xlabel('Time / s'); ylabel('Voltage / mV');
```
in matlab command line to show the image of the first channel of the selected sample. \

