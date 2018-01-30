
# Dytechlab Data Project

dytechlab_data_project.py contains all the functions.

SGX publishes derivative data daily at the address below. \n
http://www.sgx.com/wps/portal/sgxweb/home/marketinfo/historical_data/derivatives/ \n
This programcould download the following files daily from the above website. \n
1)WEBPXTICK_DT-*.zip 2) TickData_structure.dat 3) TC_*.txt 4) TC_structure.dat \n
By default, it would automatically download the files which are available on
current website.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

urllib 

if you dont have urllib library, you can install it through the following command in terminal.

```
pip install urllib
```

### Arguments

Entering the following command for more details

```
Python dytechlab_data_project.py -h 
```

On non-martket days, it would not have any data. 

if the -date argument is specified as a non-market day, it would provide the warning:

```
01-30 16:37 root         WARNING  28 Jan 2018 is not a market day. Data would not be downloaded.
```

### Remarks

On the SGX website, only the data of the past 5 market days are available for free download. I could not be abe to download the historical data before the past 5 market days through this program.



### Recovery Plan

If the 
I have some ideas about recovery plan but I have not enought time to implement it.

