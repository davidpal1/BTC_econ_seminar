from statsmodels.tsa.seasonal import seasonal_decompose

"""pripraveni casu - cena BTC"""

dataset["minutes"]= [datetime.strptime(date, "%Y-%m-%d %H:%M:%S") for date in dataset["time"]]
dataset["min_down"] = [date.replace(second=0) for date in dataset["minutes"]]
dataset["BTC_open"] = [float(BTC_dnes[BTC_dnes["cas"]==date]["open"]) for date in dataset["min_down"]]


"""
for date in dataset["minutes"]

"""
import pyflux as pf

model = pf.ARIMAX(data=dataset, formula='BTC_open~1+sum_of_satoshi+number_of_txs',
                  ar=1, ma=1, family=pf.Normal())
x = model.fit("MLE")
x.summary()


