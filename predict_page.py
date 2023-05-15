## -- REQUIRED LIBRARIES -- ##
import streamlit as st
import pickle

import pandas as pd
import numpy as np
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt

import re

from tqdm import tqdm

import requests
from bs4 import BeautifulSoup
import json
from lxml import objectify
from lxml import etree
from lxml import html
import lxml.html
import lxml.html.soupparser

import datetime
from datetime import datetime, date, time
from datetime import timedelta
from dateutil.relativedelta import relativedelta

import yfinance as yahooFinance

import sklearn
#import tensorflow as tf
#from tensorflow import keras

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

import xgboost
from xgboost import XGBRegressor

## -- -- ##

## -- U.S. Treasury Yield Curve Data -- ##

def extractEntries(sopa):
    """Assumes a bs4 object downloaded from the U.S. Treasury website.
    Returns a list with sections of the url with the yield curve data"""
    entries = sopa.find_all('content')
    tx = str(entries)
    tx = tx[1:]
    tx = tx.rstrip(tx[-1])
    tx = tx.split(",")
    return tx

def processEntries2(texto):
    """Assumes a list with sections of the url with the yield curve data.
    Returns a dict in with each key corresponds to a row"""
    entries = {}
    colPos = ["id","new_date","bc_1month","bc_2month","bc_3month","bc_4month","bc_6month",
            "bc_1year","bc_2year","bc_3year","bc_5year","bc_7year",
            "bc_10year","bc_20year","bc_30year"]
    for i in range(len(texto)):
        currEntry = texto[i]
        currEntrySplit = currEntry.split("\n")
        currEntryLen = len(currEntrySplit)
        subSetEntryList = currEntrySplit[2:(currEntryLen-3)]
        currRow = [pd.NA]*15
        for j in range(len(subSetEntryList)):
            item = re.findall('>(.+?)<', subSetEntryList[j])
            category = re.findall('d:(.+?)>', subSetEntryList[j])
            try:
                dataItem = item[0]
            except:
                pass
            try:
              extractCat = category[1].lower()
            except:
              pass
            try: 
              posInRow = colPos.index(extractCat)
            except:
              pass
            try:
              currRow[posInRow] = dataItem
            except:
              pass
        entries[i] = currRow
    return entries

def getYieldData2(yrs):
    """Assumes a list of years.
    Returns a pandas dataframe with the yield curve for the years in the list"""
    colNames = ["Id","Date","1-month","2-month","3-month","4-month","6-month","1-year","2-year","3-year","5-year","7-year","10-year","20-year","30-year"]
    treasuryYieldCurve = pd.DataFrame(columns=colNames)
    for i in tqdm(range(len(yrs))):
        currURL = 'https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value={}'.format(yrs[i])
        try:
            r = requests.get(currURL)
        except:
            print(i,r.status_code)
        soup = BeautifulSoup(r.text, features="lxml")
        txt = extractEntries(soup)
        data = processEntries2(txt)
        df = pd.DataFrame.from_dict(data, orient='index',columns=colNames)
        treasuryYieldCurve = pd.concat([treasuryYieldCurve, df], ignore_index=True, axis=0)
    return treasuryYieldCurve

def tblFormater(yldData):
    """Assumes a pandas dataframe with the yield curve data for a given number of years.
    Returns the pandas dataframe with correct data types."""
    #print("start")
    yldData["Id"] = yldData["Id"].apply(lambda x: int(x) if pd.notnull(x) else x)
    yldData["Date"] = yldData["Date"].apply(lambda x: str(x).replace("T"," ") if pd.notnull(x) else x)
    yldData["Date"] = yldData["Date"].apply(lambda x: datetime.strptime(x,"%Y-%m-%d %H:%M:%S") if pd.notnull(x) else x)
    yldData["1-month"] = yldData["1-month"].apply(lambda x: float(x) if pd.notnull(x) else x)
    yldData["2-month"] = yldData["2-month"].apply(lambda x: float(x) if pd.notnull(x) else x)
    yldData["3-month"] = yldData["3-month"].apply(lambda x: float(x) if pd.notnull(x) else x)
    yldData["4-month"] = yldData["4-month"].apply(lambda x: float(x) if pd.notnull(x) else x)
    yldData["6-month"] = yldData["6-month"].apply(lambda x: float(x) if pd.notnull(x) else x)
    yldData["1-year"] = yldData["1-year"].apply(lambda x: float(x) if pd.notnull(x) else x)
    yldData["2-year"] = yldData["2-year"].apply(lambda x: float(x) if pd.notnull(x) else x)
    yldData["3-year"] = yldData["3-year"].apply(lambda x: float(x) if pd.notnull(x) else x)
    yldData["5-year"] = yldData["5-year"].apply(lambda x: float(x) if pd.notnull(x) else x)
    yldData["7-year"] = yldData["7-year"].apply(lambda x: float(x) if pd.notnull(x) else x)
    yldData["10-year"] = yldData["10-year"].apply(lambda x: float(x) if pd.notnull(x) else x)
    yldData["20-year"] = yldData["20-year"].apply(lambda x: float(x) if pd.notnull(x) else x)
    yldData["30-year"] = yldData["30-year"].apply(lambda x: float(x) if pd.notnull(x) else x)
    return yldData

yieldCurveCurrent = getYieldData2([datetime.now().year])
yieldCurveCurrent2 = tblFormater(yieldCurveCurrent)

## -- Stock Data -- ##

# Top 25 stocks traded in the U.S. plus publickly traded stocks of defense companies
stocks = ['MSFT','AMZN','TSLA','GOOGL','GOOG','BRK-B','UNH','JNJ','XOM','JPM',
         'META','V','PG','NVDA','HD','CVX','LLY','MA','ABBV','PFE','MRK','PEP','BAC','KO','LMT','NOC','GD','BA','RTX']

def stckFormater(tbl):
  histTable = tbl.reset_index()
  histTable['Date'] = histTable['Date'].apply(lambda x: str(x)[:19] if pd.notnull(x) else x)
  histTable['Date'] = histTable['Date'].apply(lambda x: datetime.strptime(str(x),"%Y-%m-%d %H:%M:%S") if pd.notnull(x) else x)
  return histTable

def stckPull(stcks,startDate = datetime(2022, 1, 1),endDate = datetime.now()):
  """Assumes a list of stock tickers.
  Returns a pandas dataframe with the daily closing price for each stock."""
  currStockName = stcks[0]
  currStock = yahooFinance.Ticker(currStockName)
  currStockHist = currStock.history(start=startDate, end=endDate)
  currStockFormated = stckFormater(currStockHist)
  currStock2 = currStockFormated[['Date','Close']]
  stocksTable = currStock2.rename(columns={"Close": currStockName})
  for i in range(1,len(stcks)):
    currStockName = stcks[i]
    currStock = yahooFinance.Ticker(currStockName)
    currStockHist = currStock.history(start=startDate, end=endDate)
    currStockFormated = stckFormater(currStockHist)
    currStock2 = currStockFormated[['Date','Close']]
    currStockTable = currStock2.rename(columns={"Close": currStockName})
    stocksTable = pd.merge(stocksTable,currStockTable,on='Date',how='outer')
  return stocksTable.sort_values(by=['Date'])

stocksData = stckPull(stocks)

## -- U.S. Bureau of Labor Statistics Data -- ##

def getBLS(start=str(datetime(2022, 1, 1).year),end=str(datetime.now().year)): 
    """Assumes a start year and an end year. Both strings.
    Defaults: year=current year minus ten years, end=current year.
    System-allowed range is 9 years.
    Returns the following series from the U.S. Bureau of Labor Statistics:
    CPI, Import/Export Price Index, National Employment"""
    #CUUR0000SA0L1E = Consumer Price Index - All Urban Consumers
    #EIUCOCANMANU = Import/Export Price Indexes
    #CEU0800000003 = National Employment, Hours, and Earnings
    #CXUMENBOYSLB0101M = Consumer Expenditure Survey - Annual Publication thus EXCLUDED
    headers = {'Content-type': 'application/json'}
    data = json.dumps({"seriesid": ['CUUR0000SA0L1E','EIUCOCANMANU','CEU0800000003'],"startyear":start, "endyear":end})
    p = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data, headers=headers)
    json_data = json.loads(p.text)

    msg = json_data['message']
    for item in msg:
        print(item)
    
    colNames = ["seriesId","year","period","periodName","value"]
    blsData = pd.DataFrame(columns=colNames)
    
    for series in json_data['Results']['series']:
        seriesId = series['seriesID']
        for item in series['data']:
            year = item['year']
            period = item['period']
            periodName = item['periodName']
            value = item['value']
            row = [[seriesId,year,period,periodName,value]]
            temp_df = pd.DataFrame(row,columns=colNames)
            blsData = pd.concat([blsData,temp_df], ignore_index=True, axis=0)

    return blsData

blsData = getBLS()

## -- Federal Reserve Data -- ##

series_id = ['DFF','RRPONTSYD','SP500','DCOILWTICO','SOFR','DJIA','NASDAQCOM']

def getFRED(nombreSerie):
    """Assumes a series valid with the St. Louis FRED API.
    Returns a pandas data frame with the series values/observations."""
    apiKey = '9180dde91a32bac5c7699bbf994870bc'
    file_type = 'json'
    seriesName = nombreSerie

    urlSeriesObservations = 'https://api.stlouisfed.org/fred/series/observations?series_id={}&api_key={}&file_type={}'.format(nombreSerie,apiKey,file_type)
    r = requests.get(urlSeriesObservations)
    json_data = json.loads(r.text)
    
    colNames = ['Date',seriesName]
    df = pd.DataFrame(columns=colNames)

    for item in json_data['observations']:
        currDate = item['date']
        currDate = datetime.strptime(currDate,"%Y-%m-%d")
        currValue = item['value']
        row = [[currDate,currValue]]
        temp_df = pd.DataFrame(row,columns=colNames)
        df = pd.concat([df,temp_df], ignore_index=True, axis=0)
    
    return df

def multiSeriesFRED(seriesList):
    """Assumes a list of series, valid with the St. Louis FRED API.
    Returns a pandas dataframe with the series merged by date."""
    df = pd.merge(getFRED(seriesList[0]),getFRED(seriesList[1]),on='Date',how='outer')
    for i in range(2,len(seriesList)):
        temp_df = getFRED(seriesList[i])
        df = pd.merge(df,temp_df,on='Date',how='outer')
    return df

fredData = multiSeriesFRED(series_id)

## -- Data Processing -- ##

mergedEconData = pd.merge(yieldCurveCurrent2,stocksData,on="Date",how="left")
mergedEconData = pd.merge(mergedEconData,fredData,on="Date",how="left")
blsData['month'] = pd.NA
blsData['seriesName'] = pd.NA
seriesDict = {'CUUR0000SA0L1E':'CPI','EIUCOCANMANU':'Import_Export_Indx','CEU0800000003':'ntnlEmployment'}
for i in range(len(blsData)):
    month = int(re.sub('[a-zA-Z]','',blsData.iloc[i,2]))
    blsData.iloc[i,5] = month
    blsData.iloc[i,6] = seriesDict.get(blsData.iloc[i,0])
mergedEconData['CPI'] = pd.NA
mergedEconData['Import_Export_Indx'] = pd.NA
mergedEconData['ntnlEmployment'] = pd.NA
for i in range(len(mergedEconData)):
    mergedEconData.iloc[i,1] = mergedEconData.iloc[i,1].date()
for i in range(len(blsData)):
    blsData.iloc[i,1] = int(blsData.iloc[i,1])
colsDict = {'CPI':51,'Import_Export_Indx':52,'ntnlEmployment':53}
for i in tqdm(range(len(mergedEconData))):
    obsMonth = mergedEconData.iloc[i,1].month
    obsYear = mergedEconData.iloc[i,1].year   
    for j in range(len(blsData)):
        currYear = blsData.iloc[j,1]
        currMonth = blsData.iloc[j,5]
        if (obsMonth==currMonth) and (obsYear==currYear):
            colPos = colsDict.get(blsData.iloc[j,6])
            mergedEconData.iloc[i,colPos] = blsData.iloc[j,4]
yLabels = mergedEconData[["Date","1-month","2-month","3-month","4-month","6-month",
                 "1-year","2-year","3-year","5-year","7-year",
                 "10-year","20-year","30-year"]].copy()
xLabels = mergedEconData[['Date','MSFT', 'AMZN', 'TSLA', 'GOOGL', 'GOOG', 'BRK-B', 'UNH',
       'JNJ', 'XOM', 'JPM', 'META', 'V', 'PG', 'NVDA', 'HD', 'CVX', 'LLY',
       'MA', 'ABBV', 'PFE', 'MRK', 'PEP', 'BAC', 'KO', 'LMT', 'NOC', 'GD',
       'BA', 'RTX', 'DFF', 'RRPONTSYD', 'SP500', 'SOFR', 'DJIA',
       'NASDAQCOM', 'CPI', 'Import_Export_Indx', 'ntnlEmployment']].copy()
lastBLSDataRow = 0
lastCpiVal = 0
lastImpExVal = 0
lastEmplVal = 0
blsUpToDate = False
for i in reversed(range(len(xLabels) + 0)) :
    if not(pd.isnull(xLabels.iloc[i,38])) and not(pd.isnull(xLabels.iloc[i,37])) and not(pd.isnull(xLabels.iloc[i,36])):
        lastBLSDataRow = i
        lastCpiVal = xLabels.iloc[i,36]
        lastImpExVal = xLabels.iloc[i,37]
        lastEmplVal = xLabels.iloc[i,38]
        break

if lastCpiVal == 0:
    blsUpToDate = True

if blsUpToDate == False:
    for i in range((lastBLSDataRow+1),len(xLabels)):
        xLabels.iloc[i,36] = lastCpiVal 
        xLabels.iloc[i,37] = lastImpExVal 
        xLabels.iloc[i,38] = lastEmplVal

for i in range(len(xLabels.columns)):
    if pd.isnull(xLabels.iloc[(len(xLabels)-1),i]):
        xLabels.iloc[(len(xLabels)-1),i] = xLabels.iloc[(len(xLabels)-2),i]

for i in range(len(xLabels)):
    if pd.isnull(xLabels.iloc[i,33]):
        xLabels.iloc[i,33] = 0.0
    if pd.isnull(xLabels.iloc[i,3]):
        xLabels.iloc[i,3] = 0.0
    if pd.isnull(xLabels.iloc[i,12]):
        xLabels.iloc[i,12] = 0.0
    if pd.isnull(xLabels.iloc[i,19]):
        xLabels.iloc[i,19] = 0.0
    if pd.isnull(xLabels.iloc[i,32]):
        xLabels.iloc[i,32] = 0.0
    if pd.isnull(xLabels.iloc[i,34]):
        xLabels.iloc[i,34] = 0.0
    if pd.isnull(xLabels.iloc[i,11]):
        xLabels.iloc[i,11] = 0.0    
    if xLabels.iloc[i,31]==".":
        xLabels.iloc[i,31] = 0.0

dte = datetime.now() - relativedelta(years=1)
dte2 = date(dte.year, dte.month, dte.day)

yLabels2 = yLabels[yLabels['Date']>dte2] 
# Dropping 2-month and 4-month columns
yLabels2 = yLabels2[['Date', '1-month', '3-month','6-month', '1-year',
       '2-year', '3-year', '5-year', '7-year', '10-year', '20-year',
       '30-year']]

xLabels2 = xLabels[xLabels['Date']>dte2] 

X = xLabels2[['MSFT', 'AMZN', 'TSLA', 'GOOGL', 'GOOG', 'BRK-B', 'UNH', 'JNJ',
       'XOM', 'JPM', 'META', 'V', 'PG', 'NVDA', 'HD', 'CVX', 'LLY', 'MA',
       'ABBV', 'PFE', 'MRK', 'PEP', 'BAC', 'KO', 'LMT', 'NOC', 'GD', 'BA',
       'RTX', 'DFF', 'RRPONTSYD', 'SP500', 'SOFR', 'DJIA', 'NASDAQCOM', 'CPI',
       'Import_Export_Indx', 'ntnlEmployment']]

Y = yLabels2[['1-month', '3-month', '6-month', '1-year', '2-year', '3-year',
       '5-year', '7-year', '10-year', '20-year', '30-year']]

todayYvalues = yLabels2.iloc[(len(yLabels2)-1),:]
todayYvalues = todayYvalues[1:]
for i in range(len(todayYvalues)):
    todayYvalues[i] = np.float64(todayYvalues[i])

todayXvalues = xLabels2.iloc[(len(xLabels2)-1),:]
todayXvalues = todayXvalues[1:]
for i in range(len(todayXvalues)):
    todayXvalues[i] = np.float64(todayXvalues[i])

Yseries = Y.iloc[:len(Y)-1,:].copy()
Xseries = X.iloc[:len(Y)-1,:].copy()

#Handles missing values codified as "." - Raplaces with 0
for i in range(len(Xseries)):
    for j in range(29,len(Xseries.columns)):
        if Xseries.iloc[i,j] == ".":
            Xseries.iloc[i,j] = 0

#Handles missing values codified as np.nan or pd.NA - Raplaces with 0
for i in range(len(Xseries)):
    for j in range(len(Xseries.columns)):
        if pd.isnull(Xseries.iloc[i,j]):
            Xseries.iloc[i,j] = 0

#Transform all observations to np.float64 type 
Xseries = Xseries.astype(np.float64)

#Handles missing values codified as np.nan or pd.NA - Raplaces with previous observation value
for i in range(len(Yseries)):
    for j in range(len(Yseries.columns)):
        if pd.isnull(Yseries.iloc[i,j]):
            Yseries.iloc[i,j] = Yseries.iloc[i-1,j]

#Transform all observations to np.float64 type
Yseries = Yseries.astype(np.float64) 

lastDate = yLabels2.tail(1).iloc[0,0]

yPlotVals = yLabels2.tail(11).head(10)

## -- Plots --##

lastDateX = xLabels2.tail(1).iloc[0,0]
xPlotVals = xLabels2.tail(91).head(90)

## --- Stocks --- ##

stocksPlot = xPlotVals.iloc[:,0:30]
color = cm.rainbow(np.linspace(0, 1, len(stocksPlot.columns)))
selStocks = [1,2,4,6,7,8,10,12,15,16,17,18,19,25,26,27,28]

fig2 = plt.figure()

for i in range(1,len(selStocks)):
    c = color[i]
    plt.plot(stocksPlot[stocksPlot.columns[0]],
             stocksPlot[stocksPlot.columns[selStocks[i]]],
             linestyle='solid',marker='.',label='{}'.format(stocksPlot.columns[selStocks[i]]),color=c)

plt.legend(loc="upper right", frameon=True,
          bbox_to_anchor=(1.35, 1.0))
plt.xticks(rotation = 45)
plt.title("Last 90 Days of Selected Best-Performing Stocks")
plt.grid()
#plt.show()

## --- Bureau of Labor Statistics Data --- ##

blsTable = xPlotVals.iloc[:,36:39]
blsTbl = blsTable.tail(1)

## --- U.S. Federal Reserve Data --- ##

fedVals = xPlotVals[["Date","DFF","RRPONTSYD","SP500","SOFR","DJIA","NASDAQCOM"]]
fedPlot = fedVals.tail(31).head(30)  

color = cm.rainbow(np.linspace(0, 1, len(fedPlot.columns)))
fedLabels = ["notUsedVal","Federal Funds Effective Rate",
"Overnight Reverse Repurchase Agreements",
"S&P 500",
"Secured Overnight Financing Rate",
"Dow Jones Industrial Average",
"NASDAQ Composite Index"]

fig3 = plt.figure()
plt.plot(fedPlot[fedPlot.columns[0]],
         fedPlot[fedPlot.columns[1]],
         linestyle='solid',marker='.',label='{}'.format(fedLabels[1]),
         color=color[0])
plt.xticks(rotation = 45)
plt.title("Last 30 Days of {} - Source U.S. Federal Reserve".format(fedLabels[1]))
plt.grid()

fig4 = plt.figure()
plt.plot(fedPlot[fedPlot.columns[0]],
         fedPlot[fedPlot.columns[2]],
         linestyle='solid',marker='.',label='{}'.format(fedLabels[2]),
         color=color[1])
plt.xticks(rotation = 45)
plt.title("Last 30 Days of {} - Source U.S. Federal Reserve".format(fedLabels[2]))
plt.grid()

fig5 = plt.figure()
plt.plot(fedPlot[fedPlot.columns[0]],
         fedPlot[fedPlot.columns[3]],
         linestyle='solid',marker='.',label='{}'.format(fedLabels[3]),
         color=color[2])
plt.xticks(rotation = 45)
plt.title("Last 30 Days of {} - Source U.S. Federal Reserve".format(fedLabels[3]))
plt.grid()

fig6 = plt.figure()
plt.plot(fedPlot[fedPlot.columns[0]],
         fedPlot[fedPlot.columns[4]],
         linestyle='solid',marker='.',label='{}'.format(fedLabels[4]),
         color=color[3])
plt.xticks(rotation = 45)
plt.title("Last 30 Days of {} - Source U.S. Federal Reserve".format(fedLabels[4]))
plt.grid()

fig7 = plt.figure()
plt.plot(fedPlot[fedPlot.columns[0]],
         fedPlot[fedPlot.columns[5]],
         linestyle='solid',marker='.',label='{}'.format(fedLabels[5]),
         color=color[4])
plt.xticks(rotation = 45)
plt.title("Last 30 Days of {} - Source U.S. Federal Reserve".format(fedLabels[5]))
plt.grid()

fig8 = plt.figure()
plt.plot(fedPlot[fedPlot.columns[0]],
         fedPlot[fedPlot.columns[6]],
         linestyle='solid',marker='.',label='{}'.format(fedLabels[6]),
         color=color[5])
plt.xticks(rotation = 45)
plt.title("Last 30 Days of {} - Source U.S. Federal Reserve".format(fedLabels[6]))
plt.grid()

#plt.show()

## --- Yield Curve --- ##

color = cm.rainbow(np.linspace(0, 1, len(yPlotVals.columns)))

fig1 = plt.figure()

for i in range(1,len(yPlotVals.columns)):
    c = color[i]
    plt.plot(yPlotVals[yPlotVals.columns[0]],
             yPlotVals[yPlotVals.columns[i]],
             linestyle='solid',marker='o',label='{}'.format(yPlotVals.columns[i]),color=c)
plt.legend(loc="upper right", frameon=True,
          bbox_to_anchor=(1.35, 1.0))
plt.xticks(rotation = 45)
plt.title("Last 10 Days of U.S. Treasury Yield Curve")
plt.grid()
#plt.show()

## -- -- ##

## -- Loading Model -- ##

###### -- PICKLED MODELS ARE NOT WORKING -- #####
#def load_model():
#    with open('xgboostModelYieldCurve4.pkl','rb') as file:
#        retrievedData = pickle.load(file)
#    return retrievedData

#modelData = load_model()
#retrievedModel = modelData['model'] 
###### -- PICKLED MODELS ARE NOT WORKING -- #####

## -- fitting the model with only one year of data -- ##

Yseries2 = Yseries.copy()
Xseries2 = Xseries.copy()
Yseries2 = Yseries2.astype('float32')
Xseries2 = Xseries2.astype('float32')

bestModel = MultiOutputRegressor(XGBRegressor(subsample = 0.5, n_estimators = 100, max_depth = 3,
                              learning_rate = 0.3, colsample_bytree = 0.5, colsample_bylevel = 0.8999999999999999,seed = 20))

bestModel.fit(Xseries2,Yseries2)

## -- -- ##

todayXvalues = xLabels2.loc[[xLabels2.index[len(xLabels2.index)-1]]]
todayXvalues = todayXvalues.iloc[:,1:]
todayXvalues = todayXvalues.astype(np.float64)

ynew2 = bestModel.predict(todayXvalues)

predictVals = pd.DataFrame(data = ynew2,columns=todayYvalues.index.values)
actualVals = pd.DataFrame(data = todayYvalues.values.reshape(1,11),columns=todayYvalues.index)

yesterdayVals = yPlotVals.iloc[len(yPlotVals)-1,:]
yesterdayDate = yesterdayVals[0]

fig9 = plt.figure()
fig9.set_figwidth(15)
fig9.set_figheight(6)

color = cm.rainbow(np.linspace(0, 1, len(yPlotVals.columns)))
for i in range(1,len(yPlotVals.columns)):
    c = color[i]
    plt.plot(yPlotVals[yPlotVals.columns[0]],
             yPlotVals[yPlotVals.columns[i]],
             linestyle='solid',marker='o',label='{}'.format(yPlotVals.columns[i]),color=c)

for i in range(len(ynew2[0])):
    c = color[i]
    plt.plot(lastDate, ynew2[0][i], color=c,markeredgecolor="black",markersize=10,marker="*")

for i in range(len(todayYvalues.values)):
    c = color[i]
    plt.plot(lastDate,todayYvalues.values[i],color=c,markeredgecolor="black",markersize=10,marker="X")

for i in range(len(todayYvalues.values)):
    c = color[i]
    plt.arrow(yesterdayDate, yesterdayVals[i+1], 1, (todayYvalues.values[i]-yesterdayVals[i+1]), 
              color=c,linestyle="--")

for i in range(len(ynew2[0])):
    c = color[i]
    plt.arrow(yesterdayDate, yesterdayVals[i+1], 1, (ynew2[0][i]-yesterdayVals[i+1]), 
              color='black',linestyle="--")

plt.legend(loc="upper right", frameon=True,
          bbox_to_anchor=(1.15, 1.0))
plt.xticks(rotation = 45)
plt.title("Last 10 Days of U.S. Treasury Yield Curve Overlayed with Predicted Value [Star] and Actual Value [Cross]")
plt.grid()
#plt.show()

## -- Page Loading with Streamlit-- ##

def show_predict_page():
    st.title("U.S. Treasury Yield Curve Prediction with XGBoost Model")
    
    st.write("""### U.S. Treasury Yield Curve - Predicted v. Actual Value""")
    
    st.write("""#### Table1: Predicted Values""")
    st.dataframe(data=predictVals)
    st.write("""#### Table2: Actual Values""")
    st.dataframe(data=actualVals)

    st.pyplot(fig=fig9)

    st.title("Selected Data Used to Generate the Prediction")

    st.write("""### Selection of Most Important Stocks in the U.S. Economy - 90 days Prior Prediction""")

    st.pyplot(fig=fig2)

    st.write("""### Latest indicators from the U.S. Bureau of Labor Statistics""")

    st.dataframe(data=blsTbl)
    
    st.write("""### Selected indicators from the U.S. Federal Reserve""")

    st.pyplot(fig=fig3)
    st.pyplot(fig=fig4)
    st.pyplot(fig=fig5)
    st.pyplot(fig=fig6)
    st.pyplot(fig=fig7)
    st.pyplot(fig=fig8)

    st.write("""### U.S. Treasury Yield Curve 10 days Prior Prediction""")

    st.pyplot(fig=fig1)

show_predict_page()