import quandl
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn import svm, preprocessing, cross_validation

%matplotlib inline
style.use("fivethirtyeight")

# -------------------------- Gathering Data -------------------------------

api_key = "UjMPYtm8xWdYKXHD8Fok"

#Gathering state abbreviation list
def states_list():

    states = pd.read_html("https://simple.wikipedia.org/wiki/List_of_U.S._states", flavor="html5lib")


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
def HPI_benchmark():
    HPI_bench = pd.DataFrame()
    HPI_bench = quandl.get("FMAC/HPI_USA", authtoken=api_key)
    HPI_bench.columns = ['US_HPI']
    HPI_bench["US_HPI"] = ((HPI_bench["US_HPI"]-HPI_bench["US_HPI"][0]) / HPI_bench["US_HPI"][0]) * 100.0

    return HPI_bench

def gather_interest_rates():
    df = quandl.get("FMAC/MORTG", trim_start="1975-01-01", authtoken=api_key)
    df['Value'] = (df['Value'] - df['Value'][0]) / df['Value'][0] * 100.00
    df = df.resample("M").mean()

    return df

def mort_rate_30y():
    rate_30 = pd.DataFrame()
    rate_30 = quandl.get("FMAC/MORTG", trim_start='1975-01-01', authtoken=api_key)
    rate_30['Value'] = (rate_30['Value'] - rate_30['Value'][0]) / rate_30['Value'][0] * 100.00
    rate_30 = rate_30.resample('1D').mean()
    rate_30 = rate_30.resample('M').mean()
    rate_30.columns = ['rate_30']

    return rate_30

def GDP_data():
    df = pd.DataFrame()
    df = quandl.get("BEA/GDP", trim_start = '1975-01-01', authtoken=api_key)
    df.rename(columns={'GDP in billions of current dollars':'GDP'}, inplace=True)
    df['GDP'] = (df['GDP'] - df['GDP'][0]) / df['GDP'][0] * 100.00
    df = df.resample('M').mean()
    df.fillna(method='ffill', inplace=True)
    df = df['GDP']

    return df

def sp500_data():
    df = pd.DataFrame()
    df = quandl.get('YAHOO/INDEX_GSPC', trim_start='1975-01-01', authtoken=api_key, collapse='monthly')
    df['Adjusted Close'] = (df['Adjusted Close'] - df['Adjusted Close'][0]) / df['Adjusted Close'][0] * 100.00
    df.rename(columns={'Adjusted Close':'sp500'}, inplace=True)
    df = df['sp500']
    return df

def us_unemployment():
    df = quandl.get("ECPI/JOB_G", trim_start='1975-01-01', authtoken=api_key)
    df['Unemployment Rate'] = (df['Unemployment Rate'] - df['Unemployment Rate'][0]) / df['Unemployment Rate'][0] * 100.00
    df = df.resample('1D').mean()
    df = df.resample('M').mean()
    return df

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


# Forecasting
HPI = pd.read_pickle("HPI.pickle")
HPI = HPI.pct_change()
HPI.replace([np.inf, -np.inf], np.nan, inplace=True)
HPI.dropna(inplace=True)

HPI['US_HPI_forecast'] = HPI['US_HPI'].shift(-1)

def forecast_labels(current_HPI, future_HPI):
    if future_HPI > current_HPI:
        return 1
    else:
        return 0

HPI['label'] = list(map(forecast_labels, HPI['US_HPI'], HPI['US_HPI_forecast']))

X = np.array(HPI.drop(['label','US_HPI_forecast'], 1))
y = preprocessing.scale(X)
y = np.array(HPI['label'])
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
print(clf.score(X_test, y_test))
