[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![GitHub watchers](https://img.shields.io/badge/Watchers-1-blue)](https://github.com/Anghille/covid_19_analysis/watchers)
[![Pull-Requests Welcome](https://img.shields.io/badge/Pull%20Request-Welcome-blue)](https://github.com/Anghille/covid_19_analysis/pulls)

[![python](https://img.shields.io/badge/Made%20with-Python-blue)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-v0.0.1-blue)](https://github.com/Anghille/covid_19_analysis#versioning)
[![References](https://img.shields.io/badge/References-blue)](https://github.com/Anghille/covid_19_analysis#references)

# Time Series Forecasting of Covid-19 Transmission Using LSTM Networks (WIP)

## Introduction

The Coronavirus, also known as Covid-19 or SARS-Cov-2, is an infectious disease caused by a virus belonging to the coronavirus family. First originated in the Huabei Province un China in december 2019, the Coronavirus spread around the world within a few weeks with most of Europe, North America, Middle-East and Asia contaminated by March 11th.

## Datasets  


I used the data from Novel Corona Virus 2019 Datasets (https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset). Those datasets are an aggregation of data coming from multiple sources (government, google own data, medical report...).

It contains mutliples datasets:
* covid_19_data
* time_series_covid_19_confirmed
* time_series_covid_19_deaths
* time_series_covid_19_recovered
* Other that wasn't put in use here..

## Coronavirus choropleth maps  


A choropleth map is a type of thematic map where areas or regions are shaded in proportion to a given data variable. The data that I used to create the following visualizations is the Novel Corona Virus 2019 dataset from Kaggle.  

![chorepleth](/files/chorepleth.png)

Static choropleth maps are most useful when you want to compare a desired variable by region. For example, if you wanted to compare the crime rate of each state in the US at a given moment, you could visualize it with a static choropleth.  

An animated or dynamic choropleth map is similar to a static choropleth map, except that you can compare a variable by region,over time. This adds a third dimension of information and is what makes these visualizations so interesting and powerful.  

## Evolution of cases and recovered cases graphs for each country

This code let you choose the country you want to display. It shows one or multiple graphs with covid confirmed and recovered cases by country. Changing (last line of the script) the "pays" list (pays=["France"]) to other country name or empty list will display those countries (pays=[] shows all countries)

```python
# Get the data from link above (kaggle novel covid 19 data) and import it
covid_data = pd.read_csv("../covid/Data/covid_19_data.csv")
covid_recovered = pd.read_csv("../covid/Data/time_series_covid_19_recovered.csv")
covid_deaths = pd.read_csv("../covid/Data/time_series_covid_19_deaths.csv")
covid_confirmed = pd.read_csv("../covid/Data/time_series_covid_19_confirmed.csv")

# Prepare the data
def cleaning_data(dataframe):
    dataframe.rename(columns={"Province/State":"State","Country/Region":"Country"}, inplace=True)
    dataframe = dataframe.groupby("Country").sum()
    dataframe = dataframe.iloc[:,2:].T.reset_index()
    dataframe = dataframe.set_index(pd.to_datetime(dataframe["index"])).drop(columns="index")

    return dataframe    
covid_recovered = cleaning_data(covid_recovered)
covid_confirmed = cleaning_data(covid_confirmed)

# Plot the evolution of the country recovered and confirmed covid cases
def lineplot_timeseries(df,df2, title="Evolution of COVID-19 confirmed and recovered cases", legend="Type of Cases", pays=[], width_coef=1, height_coef=1):
    if not pays:
        pays = df.columns
    else: 
        pays = pays
    
    subplot_rows = int(len(pays)//4) 
    if subplot_rows <= 1:
        if len(pays) <= 4:
            subplot_rows = 1
        else: 
            subplot_rows = 2
    if len(pays) < 2:
        subplot_cols = 1
    elif len(pays) < 3:
        subplot_cols = 2
    elif len(pays) < 4:
        subplot_cols = 3
    else: 
        subplot_cols = 4
    
    
    fig = make_subplots(rows=subplot_rows, cols=subplot_cols, subplot_titles = pays, shared_xaxes=True)

    col = 1
    row = 1
    show_legend_switch = 0
    for column in pays:
        if col <= subplot_cols:
            if show_legend_switch == 0:
                fig.append_trace(go.Scatter(
                    x = df[column].index,
                    y = df[column],
                    name="Confirmed",
                    mode="lines",
                    line=go.scatter.Line(color="red"),
                    legendgroup='group1'), row=row, col=col)
                fig.append_trace(go.Scatter(
                    x = df2[column].index,
                    y = df2[column],
                    name="Recovered",
                    mode="lines",
                    line=go.scatter.Line(color="blue"),
                    legendgroup='group2'), row=row, col=col)
                fig.update_xaxes(
                    row=row, col=col,
                    showline=True,
                    showgrid=False,
                    automargin=True,
                    showticklabels=True,
                    linecolor='rgb(204, 204, 204)',
                    linewidth=2,
                    ticks='outside',
                    tickfont=dict(
                        family='Arial',
                        size=12,
                        color='rgb(82, 82, 82)',)
                )
                fig.update_yaxes(
                    row=row, col=col,
                    showline=True,
                    showgrid=False,
                    automargin=True,
                    showticklabels=True,
                    linecolor='rgb(204, 204, 204)',
                    linewidth=2,
                    ticks='outside',
                    tickfont=dict(
                        family='Arial',
                        size=12,
                        color='rgb(82, 82, 82)',)
                )
                col+=1
                show_legend_switch = 1
            else: 
                fig.append_trace(go.Scatter(
                    x = df[column].index,
                    y = df[column],
                    name="Confirmed",
                    mode="lines",
                    line=go.scatter.Line(color="red"),
                    legendgroup='group1',
                    showlegend=False), row=row, col=col)
                fig.append_trace(go.Scatter(
                    x = df2[column].index,
                    y = df2[column],
                    name="Recovered",
                    mode="lines",
                    line=go.scatter.Line(color="blue"),
                    legendgroup='group2',
                    showlegend=False), row=row, col=col)
                fig.update_xaxes(
                    row=row, col=col,
                    automargin=True,
                    showline=True,
                    showgrid=False,
                    showticklabels=True,
                    linecolor='rgb(204, 204, 204)',
                    linewidth=2,
                    ticks='outside',
                    tickfont=dict(
                        family='Arial',
                        size=12,
                        color='rgb(82, 82, 82)',)
                )
                fig.update_yaxes(
                    row=row, col=col,
                    showline=True,
                    automargin=True,
                    showgrid=False,
                    showticklabels=True,
                    linecolor='rgb(204, 204, 204)',
                    linewidth=2,
                    ticks='outside',
                    tickfont=dict(
                        family='Arial',
                        size=12,
                        color='rgb(82, 82, 82)',)
                )
                col+=1
            
        elif (col > subplot_cols) & (row < subplot_rows):
            row+=1
            col=1
            fig.append_trace(go.Scatter(
                x = df[column].index,
                y = df[column],
                name="Confirmed",
                mode="lines",
                line=go.scatter.Line(color="red"),
                legendgroup='group1',
                    showlegend=False), row=row, col=col)
            fig.append_trace(go.Scatter(
                x = df2[column].index,
                y = df2[column],
                name="Recovered",
                mode="lines",
                line=go.scatter.Line(color="blue"),
                legendgroup='group2',
                    showlegend=False), row=row, col=col)
            fig.update_xaxes(
                row=row, col=col,
                showline=True,
                showgrid=False,
                automargin=True,
                showticklabels=True,
                linecolor='rgb(204, 204, 204)',
                linewidth=2,
                ticks='outside',
                tickfont=dict(
                    family='Arial',
                    size=12,
                    color='rgb(82, 82, 82)',)
            )
            fig.update_yaxes(
                row=row, col=col,
                showline=True,
                showgrid=False,
                automargin=True,
                showticklabels=True,
                linecolor='rgb(204, 204, 204)',
                linewidth=2,
                ticks='outside',
                tickfont=dict(
                    family='Arial',
                    size=12,
                    color='rgb(82, 82, 82)',)
            )
            col+=1

    fig.update_layout(
        autosize=False,
        margin=dict(
            autoexpand=True,
            l=100,
            r=20,
            t=110,
        ),
        title=title,
        legend_title="Type of cases",
        plot_bgcolor='white',
        height=((3860/34034)*len(pays)**2 + ((100-35*(3860/34034))/5)*len(pays) + (400-((100-35*(3860/34034))/5)-(3860/34034)))*height_coef,
        width=(((-785/17017)*len(pays)**2 + (25095/2431)*len(pays) + (16842121/17017)))*width_coef, 
    )
    fig.show()
    
    # Change the "pays" list and add other country of need (ex: pays=["Italy","Germany","Austria"]). An empty list show ALL THE COUNTRIES
    lineplot_timeseries(covid_confirmed, covid_recovered, pays=["France"], width_coef=1, height_coef=1.5)
```

![Covid Evolution](/files/covid_evolution.png)

## TimeSeries Analysis

A large part of real-world datasets are temporal. Data collected over regular intervals of time is called time-series (TS) data and each data point is equally spaced over time. This last property is less true in real-life but this assumption is usually used. TS prediction is the method of forecasting upcoming trends/patterns of the given dataset that contains temporal features. In order to forecast COVID-19 transmission, we need input data as temporal components. A time series (TS) data can be break downed into trend, seasonality and error.

Trend - presence of pattern that repeats on regular intervals of time due to external factors like lockdown, mandatory social distancing, etc.
Seasonality - presence of variations that occur at specific regular intervals less than a year, such as weekly, monthly, or quarterly, etc.
In many real-world scenarios, either of trend or seasonality are absent. After finding the nature of TS, various forecasting methods have to be applied on given TS.


Given a TS, we can broadly classify into 2 categories: stationary and non-stationary.

Stationary TS: does not depend on the time components like trend, seasonality effects. Mean and variances are constant with respect to time. Stationary TS is easier to analyze and results skilful forecasting.
Non-stationary TS: has trend, seasonality effects in it and changes with respect to time. Mean, variance, standard deviation also changes with respect to time.

### ADF test

As stated by Chimula & Zhang (2020), in order to check the nature (stationarity and non-stationarity) of our COVID-19 dataset, we have to perform the Augmented Dickey Fuller (ADF) test (Cheung & Lai, 1995) on the input data. ADF is the standard unit root test to find the impact of trends on the data . If the p-value is less than 0.05, it rejects the null hypothesis i.e. it does not have a unit root and it is called stationary series. If the p-value is greater than 0.05 the input data has unit root so it is regarded as non-stationary series.

## Versioning
### v0.0.1

Added Dynamic Chorepleth *graphic*  
Added infected/recovered per Country *graphic*  
Added DataSet *cleaning*  



## References

1. CDC COVID-19 Response Team, CDC COVID-19 Response Team, Bialek, S., Boundy, E., Bowen, V., Chow, N., Cohn, A., Dowling, N., Ellington, S., Gierke, R., Hall, A., MacNeil, J., Patel, P., Peacock, G., Pilishvili, T., Razzaghi, H., Reed, N., Ritchey, M., & Sauber-Schatz, E. (2020). Severe Outcomes Among Patients with Coronavirus Disease 2019 (COVID-19)—United States, February 12–March 16, 2020. MMWR. Morbidity and Mortality Weekly Report, 69(12), 343‑346.

2. Cheung, Y.-W., & Lai, K. S. (1995). Lag Order and Critical Values of the Augmented Dickey–Fuller Test. Journal of Business & Economic Statistics, 13(3), 277‑280.

3. Chimmula, V. K. R., & Zhang, L. (2020). Time series forecasting of COVID-19 transmission in Canada using LSTM networks. Chaos, Solitons and Fractals: Nonlinear Science, and Nonequilibrium and Complex Phenomena, 135, 1‑6.

4. World Health Organization. (2020). Coronavirus disease 2019 (COVID-19): situation report, 82.

5. Yansun Xu, Weaver, J. B., Healy, D. M., & Jian Lu. (1994). Wavelet transform domain filters : A spatially selective noise filtration technique. IEEE Transactions on Image Processing, 3(6), 747‑758.
