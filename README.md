
# Watch-Based Hand Activity Recognition

python-src folder contains all the codes.

## pre-processing

### process_raw_data.py
sync the raw data which is recorded by the phone and watch

### ProprocessingGlobalAcc.ipynb
All the preprocessing:
 Load the linear acceleration in earth's coordination system (The convention from device's coordinate system to Earth's coordinate system is done by android App. The android App is written by myself
 Correct velocity drift
 Calculate displacenemnt from the starting point to the finishing point
 combine x&y axis

### Plot.ipynb
some analysis and ploting

### user_independent_split.py
User independently split the data collected into training and testing dataset

## train

### 0-6 BinaryModel.ipynb
Evaluate, training and Save the binary model
The trained binary model will be saved as binary_model.pkl

### final.py
Train the SVC, RF, CNN, DNN as the main model 
Get the overall results which is the result obtained by the whole system
The whole regocnition system combines the result of the main model and binary model.





