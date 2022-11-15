from statsmodels.tsa.seasonal import seasonal_decompose

"""pripraveni casu - cena BTC"""
dates_list = [datetime.strptime(date, "%Y-%m-%d %H:%M:%S") for date in dataset["time"]]
dates_list_round_minute = [date.replace(second=0) for date in dates_list]
BTC_dnes[BTC_dnes["cas"].isin(dates_list_round_minute)]["open"]


decompose = seasonal_decompose(BTC_dnes['open'],model='additive', period=7)
decompose.plot()
plt.show()