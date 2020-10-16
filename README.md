[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![GitHub watchers](https://img.shields.io/badge/Watchers-1-blue)](https://github.com/Anghille/covid_19_analysis/watchers)
[![Pull-Requests Welcome](https://img.shields.io/badge/Pull%20Request-Welcome-blue)](https://github.com/Anghille/covid_19_analysis/pulls)

[![python](https://img.shields.io/badge/Made%20with-Python-blue)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-v0.0.1-blue)](https://github.com/Anghille/covid_19_analysis#versioning)
[![References](https://img.shields.io/badge/References-blue)](https://github.com/Anghille/covid_19_analysis#references)

# Time Series Forecasting of Covid-19 Transmission Using LSTM Networks (WIP)

## Introduction

The Coronavirus, also known as Covid-19 or SARS-Cov-2, is an infectious disease caused by a virus belonging to the coronavirus family. First originated in the Huabei Province un China in december 2019, the Coronavirus spread around the world within a few weeks with most of Europe, North America, Middle-East and Asia contaminated by March 11th.
  
<br><br>
## Goal  

1. Implement each algorithm by hand to understand the model fully and showcase his functioning mechanism. 
2. Predict Covid Cases (first for France, then other countries)

## Datasets  

I used the data from Novel Corona Virus 2019 Datasets (https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset). Those datasets are an aggregation of data coming from multiple sources (government, google own data, medical report...).

It contains mutliples datasets:
* covid_19_data
* time_series_covid_19_confirmed
* time_series_covid_19_deaths
* time_series_covid_19_recovered
* Other that wasn't put in use here..
  
  <br><br>
## Coronavirus choropleth maps  

A choropleth map is a type of thematic map where areas or regions are shaded in proportion to a given data variable. The data that I used to create the following visualizations is the Novel Corona Virus 2019 dataset from Kaggle.

<p align="center">
<img src="/files/chorepleth.png" alt="Evolution Covid Cases" width=800>
</p>

Static choropleth maps are most useful when you want to compare a desired variable by region. For example, if you wanted to compare the crime rate of each state in the US at a given moment, you could visualize it with a static choropleth.    

An animated or dynamic choropleth map is similar to a static choropleth map, except that you can compare a variable by region,over time. This adds a third dimension of information and is what makes these visualizations so interesting and powerful.    
  
<br><br>  
## Evolution of cases and recovered cases graphs for each country

Show the evolution of recovered and confirmed cases, by country. In the script, change the `pays=[]` parameter with name of desired countries to display them. 
ex : 
```python
lineplot_timeseries(covid_confirmed, covid_recovered, pays=["France","Italy","Germany","Algeria","Austria","Australia","Russia","Mexico","Argentina","Peru","Colombia"], width_coef=1, height_coef=1.5)
```
<p align="center">
<img src="/files/evolution_covid.png" alt="Evolution Covid Cases" width=700 height=500 >
</p>
  
<br><br>  
## TimeSeries Analysis

A large part of real-world datasets are temporal. Data collected over regular intervals of time is called **time-series (TS)** data and each data point is equally spaced over time. This last property is less true in real-life but this assumption is usually used. TS prediction is the method of **forecasting upcoming trends/patterns** of the given dataset that contains temporal features. In order to forecast COVID-19 transmission, we need input data as temporal components. A time series (TS) data can be break downed into **trend**, **seasonality** and **error**.  

* **Trend** - presence of pattern that repeats on regular intervals of time due to external factors like lockdown, mandatory social distancing, etc.
* **easonality** - presence of variations that occur at specific regular intervals less than a year, such as weekly, monthly, or quarterly, etc.  
In many real-world scenarios, either of trend or seasonality are absent. After finding the nature of TS, various forecasting methods have to be applied on given TS.


Given a TS, we can broadly classify into 2 categories: stationary and non-stationary.

* **Stationary TS** - does not depend on the time components like trend, seasonality effects. Mean and variances are constant with respect to time. Stationary TS is easier to analyze and results skilful forecasting.
* **Non-stationary TS** - has trend, seasonality effects in it and changes with respect to time. Mean, variance, standard deviation also changes with respect to time.

### ADF test

As stated by *Chimula & Zhang* (2020), in order to check the nature (stationarity and non-stationarity) of our COVID-19 dataset, we have to perform the **Augmented Dickey Fuller (ADF) test** (*Cheung & Lai, 1995*) on the input data. ADF is the standard unit root test to find the impact of trends on the data . If the *p-value* is **less than 0.05**, it rejects the null hypothesis i.e. it does not have a unit root and it is called stationary series. If the *p-value* is **greater than 0.05** the input data has unit root so it is regarded as non-stationary series.
  
  
<br><br><br> 
## Versioning
### v0.0.1

Added Dynamic Chorepleth *graphic*  
Added infected/recovered per Country *graphic*  
Added DataSet *cleaning*  
  
  
<br><br><br>  
## References

1. CDC COVID-19 Response Team, CDC COVID-19 Response Team, Bialek, S., Boundy, E., Bowen, V., Chow, N., Cohn, A., Dowling, N., Ellington, S., Gierke, R., Hall, A., MacNeil, J., Patel, P., Peacock, G., Pilishvili, T., Razzaghi, H., Reed, N., Ritchey, M., & Sauber-Schatz, E. (2020). Severe Outcomes Among Patients with Coronavirus Disease 2019 (COVID-19)—United States, *February 12–March 16, 2020. MMWR. Morbidity and Mortality Weekly Report, 69(12)*, 343‑346.

2. Cheung, Y.-W., & Lai, K. S. (1995). Lag Order and Critical Values of the Augmented Dickey–Fuller Test. *Journal of Business & Economic Statistics, 13(3)*, 277‑280.

3. Chimmula, V. K. R., & Zhang, L. (2020). Time series forecasting of COVID-19 transmission in Canada using LSTM networks. *Chaos, Solitons and Fractals: Nonlinear Science, and Nonequilibrium and Complex Phenomena, 135*, 1‑6.

4. World Health Organization. (2020). Coronavirus disease 2019 (COVID-19): situation report, 82.

5. Yansun Xu, Weaver, J. B., Healy, D. M., & Jian Lu. (1994). Wavelet transform domain filters : A spatially selective noise filtration technique. *IEEE Transactions on Image Processing, 3(6)*, 747‑758.
