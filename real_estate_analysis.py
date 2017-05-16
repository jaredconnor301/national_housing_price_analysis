import quandl
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
%matplotlib inline
style.use("fivethirtyeight")

# -------------------------- Gathering Data -------------------------------

api_key = "UjMPYtm8xWdYKXHD8Fok"

#Gathering state abbreviation list
def states_list():
    
    states = pd.read_html("https://simple.wikipedia.org/wiki/List_of_U.S._states")
    
#Gathering state housing price data
def gather_initial_state_data():

    main_df = pd.DataFrame()

    for abrev in states[0][0][1:]:
        query = "FMAC/HPI_" + str(abrev)
        df = quandl.get(query, authtoken=api_key)
        df.columns = [abrev]

        if main_df.empty:
            main_df = df
            else:
                main_df = main_df.join(df)

    #Saving that data to a pickle
    pickle_out = open("states_housing_index.pickle", "wb")
    pickle.dump(main_df, pickle_out)
    pickle_out.close()
    

        
def gather_percent_change_data():
    main_df2 = pd.DataFrame()

    for abrev in states[0][0][1:]:
        query = "FMAC/HPI_" + str(abrev)
        df = quandl.get(query, authtoken=api_key)
        df = df.rename(columns={'Value': abrev})     
        df[abrev] = ((df[abrev] - df[abrev][0]) / df[abrev] [0]) * 100.00

        if main_df2.empty:
            main_df2 = df
        else:
            main_df2 = main_df2.join(df)
    
    #Save data to a pickle object
    pickle_out = open("percent_change_HPI.pickle", "wb")
    pickle.dump(main_df2, pickle_out)
    pickle_out.close()

    
#importing housing data from saved pickle of state price index 
def import_housing_data():
    
    #Import original dataset
    pickle_in = open("states_housing_index.pickle", "rb")
    HPI_data = pickle.load(pickle_in)

    #Import percent change dataset
    pickle_in = open("percent_change_HPI.pickle", "rb")
    HPI_percent_data = pickle.load(pickle_in)
    

#Gathers and creates series for national benchmark
def national_benchmark():
    
    HPI_bench = pd.DataFrame()
    HPI_bench = quandl.get("FMAC/HPI_USA", authtoken=api_key)
    HPI_bench.columns = ['United States']
    HPI_bench["United States"] = ((HPI_bench["United States"]-HPI_bench["United States"][0]) / HPI_bench["United States"][0]) * 100.0
    
    

# -------------------------- Analysis -------------------------------

#Plot original series
HPI_data.plot()
plt.legend().remove()
plt.show()
    

#Plot Percent Change:
plt.clf()

fig = plt.figure()
ax1 = plt.subplot2grid((1,1), (0,0))

HPI_percent_data.plot(ax=ax1)
HPI_bench.plot(color='k', ax=ax1, linewidth=7)

plt.legend().remove()
plt.show()
    
#Transform Data
HPI_data_SA1yr = HPI_data['OR'].resample('A').mean()

HPI_data['OR'].plot()
HPI_data_SA1yr.plot(color="k", linewidth=2)

plt.show()
    