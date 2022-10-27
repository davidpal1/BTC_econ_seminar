import os
import time
from datetime import date, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger
from strenum import StrEnum

#casy
from datetime import datetime

API_KEY = "202001ZjMvj8R3BF"
DATA_PATH = "data"
ONE_BTC = 100000000  # Satoshis

DATE_FROM = date(2022, 9, 1)
DATE_TO = date(2022,9,10)
TRANSACTION_TRASHOLD = ONE_BTC * 100
TXIN_TRASHOLD = ONE_BTC * 0
TXOUT_TRASHOLD = ONE_BTC * 0


class FileType(StrEnum):
    """
    Transactions contain only basic data, information about addresses are stored in separate types of files.
    """

    transactions = "transactions"
    inputs = "inputs"
    outputs = "outputs"


class TypeConfig:
    """
    Configuration of btc and usd thresholds and columns that should be filtered from loaded dataframe.
    """

    def __init__(
        self, btc_threshold: Optional[int] = None, usd_threshold: Optional[float] = None, columns: Optional[list] = None
    ):
        self.btc_threshold = btc_threshold
        self.usd_threshold = usd_threshold
        self.columns = columns


class TransactionExtractor:
    """
    Class for downloading data files and loading them filtered.
    """

    _file_type: Optional[FileType]
    _date_from: date
    _date_to: date

    _configs: Dict[str, TypeConfig] = dict()
    _result_dfs: Dict[str, pd.DataFrame] = dict()
    _day_transactions_map: Dict[date, pd.DataFrame] = dict()

    def __init__(
        self,
        file_type: Optional[FileType] = None,
        date_from: date = DATE_FROM,
        date_to: date = DATE_TO,
        *,
        tx_config: Optional[TypeConfig] = None,
        txin_config: Optional[TypeConfig] = None,
        txout_config: Optional[TypeConfig] = None,
    ):
        """

        :param file_type: File type to work with. If None, all types are considered.
        :param date_from: Date from which should be data processed (including).
        :param date_to: Date to which should be data processed (excluding).
        :param tx_config: Configuration of transactions filtering.
        :param txin_config: Configuration of transaction inputs filtering.
        :param txout_config: Configuration of transaction outputs filtering.
        """
        self._file_type = file_type
        self._date_from = date_from
        self._date_to = date_to

        self._configs[FileType.transactions] = (
            TypeConfig(
                btc_threshold=TRANSACTION_TRASHOLD,
                usd_threshold=None,
                columns=["hash", "time", "input_total", "output_total", "input_total_usd", "output_total_usd"],
            )
            if tx_config is None
            else tx_config
        )

        self._configs[FileType.inputs] = (
            TypeConfig(
                btc_threshold=TXIN_TRASHOLD,
                usd_threshold=None,
                columns=[
                    "recipient",
                    "value",
                    "spending_value_usd",
                    "spending_transaction_hash",
                    "is_from_coinbase",
                ],
            )
            if txin_config is None
            else txin_config
        )

        self._configs[FileType.outputs] = (
            TypeConfig(
                btc_threshold=TXOUT_TRASHOLD,
                usd_threshold=None,
                columns=["recipient", "value", "value_usd", "transaction_hash"],
            )
            if txout_config is None
            else txout_config
        )

        logger.debug(f"Initialized instance with type {self._file_type} from date {self._date_from} to {self._date_to}")

        if self._date_from >= self._date_to:
            logger.error("Date from is greater than or equal to date from!")
            raise Exception

    def _load_missing_type(self, file_type: str):
        logger.info(f"Started loading {file_type}.")
        start_time = time.time()
        downloading_date = self._date_from

        dir_path = os.path.join(os.getcwd(), DATA_PATH, file_type)
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
            logger.info(f"Creating data directory {dir_path}")

        file_names = []
        while downloading_date < self._date_to:
            file_names.append(self._get_file_name(file_type, downloading_date))
            downloading_date = downloading_date + timedelta(days=1)

        already_downloaded = os.listdir(dir_path)
        to_download = [item for item in file_names if item not in already_downloaded]

        for file_name in to_download:
            self._download_file(file_type, file_name)

        logger.info(f"Downloaded {len(to_download)} files in {(time.time() - start_time):.2f} seconds.")

    @staticmethod
    def _get_file_name(file_type: str, file_date: date) -> str:
        return f"blockchair_bitcoin_{file_type}_{file_date.strftime('%Y%m%d')}.tsv.gz"

    def _download_file(self, file_type: str, file_name: str):
        cmd = f"curl -k https://gz.blockchair.com/bitcoin/{file_type}/{file_name}?key={API_KEY} -o {os.path.join(DATA_PATH, file_type, file_name)}"
        logger.debug(f"Downloading file {file_name}.")
        os.system(cmd)

    def download_missing_files(self):
        """
        Checks existence of all necessary files in specific data folder. Downloads all missing.
        """
        logger.info("Started looking for missing files.")

        if self._file_type:
            self._load_missing_type(self._file_type)
        else:
            for file_type in FileType:
                self._load_missing_type(file_type)

        logger.info(f"All data between {self._date_from} and {self._date_to} are downloaded.")

    def load_filtered_data(self):
        """
        Reads data files one by one, filters them and concatenates results in a common dataframe.
        """
        if self._file_type:
            self._load_type(self._file_type)
        else:
            # transactions FileType should be processed first
            self._load_type(FileType.transactions)
            self._load_type(FileType.inputs)
            self._load_type(FileType.outputs)

    def _load_type(self, file_type: FileType):
        logger.info(f"Started processing {file_type}.")
        start_time = time.time()
        df_list = []
        type_config = self._configs[file_type]
        processing_date = self._date_from

        selected_columns = type_config.columns.copy() if type_config.columns else None
        if selected_columns is not None and file_type == FileType.transactions:
            selected_columns += ["is_coinbase"]

        while processing_date < self._date_to:
            file_name = self._get_file_name(file_type, processing_date)
            tsv_start_time = time.time()

            file_path = os.path.join(DATA_PATH, file_type, file_name)
            try:
                aux_df = pd.read_csv(file_path, sep="\t", usecols=selected_columns)
            except FileNotFoundError:
                logger.error(f"Could not find file {file_path}. Download missing files first!")
                return
            except Exception as e:
                logger.error(f"Could not read file {file_path}. Try downloading it again. Exception message:\n{e.args}")
                return

            logger.debug(
                f"Loaded file {file_name} with {len(aux_df)} rows in {(time.time() - tsv_start_time):.2f} seconds."
            )
            if type_config.btc_threshold:
                btc_value_column_name = "output_total" if file_type == FileType.transactions else "value"
                aux_df = aux_df[aux_df[btc_value_column_name] > type_config.btc_threshold]

            if type_config.usd_threshold:
                usd_value_columns = {
                    FileType.transactions: "output_total_usd",
                    FileType.inputs: "spending_value_usd",
                    FileType.outputs: "value_usd",
                }
                usd_value_column_name = usd_value_columns[file_type]
                aux_df = aux_df[aux_df[usd_value_column_name] > type_config.usd_threshold]

            if file_type == FileType.transactions:
                aux_df = aux_df[aux_df["is_coinbase"] == 0]

            if type_config.columns:
                aux_df = aux_df[type_config.columns]

            if file_type == FileType.transactions:
                self._day_transactions_map[processing_date] = aux_df["hash"]
            else:
                transaction_hash_column = (
                    "transaction_hash" if file_type == FileType.outputs else "spending_transaction_hash"
                )
                if processing_date in self._day_transactions_map:
                    aux_df = aux_df[aux_df[transaction_hash_column].isin(self._day_transactions_map[processing_date])]

            df_list.append(aux_df)
            processing_date = processing_date + timedelta(days=1)

        result_df = pd.concat(df_list)
        self._result_dfs[file_type] = result_df

        logger.info(f"Finished processing {file_type} in {(time.time() - start_time):.2f} seconds.")

    def get_resulting_df(self, file_type: Optional[FileType] = None) -> pd.DataFrame:
        """
        :param file_type: File type that we are interested in. Default is transactions.
        :return: Returns filtered dataframe stored in the class instance.
        """
        implied_file_type = FileType.transactions
        if not file_type:
            if self._file_type:
                implied_file_type = self._file_type
        else:
            implied_file_type = file_type

        if implied_file_type not in self._result_dfs:
            logger.warning(f"{implied_file_type} dataframe not loaded.")
            return pd.DataFrame()
        else:
            return self._result_dfs[implied_file_type]


###############
# example usage
###############

# loading example of data snippet for one day without any filtering
blank_config = TypeConfig(btc_threshold=None, usd_threshold=None, columns=None)
tex = TransactionExtractor(date_from=date(2022, 9, 1), date_to=date(2022, 9, 2),
                           tx_config=blank_config, txin_config=blank_config, txout_config=blank_config)
tex.download_missing_files()
tex.load_filtered_data()
tx = tex.get_resulting_df(FileType.transactions)
txin = tex.get_resulting_df(FileType.inputs)
txout = tex.get_resulting_df(FileType.outputs)



# example of loading and filtering just transactions from this year
tex = TransactionExtractor(FileType.transactions, date_from=date(2022, 9, 1))
tex.download_missing_files()
tex.load_filtered_data()

tx = tex.get_resulting_df()

# Dodatečný code
"""
tx:
block_id - číslo bloku transakce
hash - hash transakce
time - validace bloku
size - ???
weight - ???
version - ????
lock_time - ????
is_coinbase - ???
has_witness - ????
input_count - ??  počet UTXO skládajících se na transakci??
output_count - ?? odhoz + fee + návrat??? ??
input_total - ?? Celková částka transakce (Shitoshi) ??
input_total_usd - ?? Coelková částka v dolarech ?? 
output_total - ???
output_total_usd - ???
fee - ???
fee - usd ???
fee_per_kb(_usd) - ??
fee_per_kwu(_usd) - ???
ccd_total - ??

txin:
block_id - odkud beru UTXO
transaction_hash - hash transakce ze ktere beru UTXO
index  - ??? index transakce v bloku ???
time - čas transakce UTXO
value - hodnota UTXO v satoshis
recipient - komu posílám transakci
type - ????
script_hash - ???
is_from_coinbase - ???
is_spendable - ???
spending_block_id - blok ve kterém utrácím UTXO
spending_transaction_hash - hash transakce ve které utrácím 
spending_index  - ???
spending_time - čas ve kterém utrácím (nová transakce)
spending_value_usd - ???
spending_sequence - ???
spending_signature_hex - ???
spending_witness  - ???
lifespan - čas [vteřiny] po který byl UTXO "živý"
cdd - hodnota UTXO * čas [dny] "života" UTXO

txout:
block_id - block transakce
transaction_hash - hash transakce
index - ???
time - čas validace bloku
value(_usd) - ???
recipient - ???
type - ???
script_hex - ???
is_from_coinbase ???
is_spendable ???

"""
#Adresa binance
binance="bc1qm34lsc65zpw79lxes69zkqmk6ee3ewf0j77s3h"
sum(txin["recipient"]=="bc1qm34lsc65zpw79lxes69zkqmk6ee3ewf0j77s3h")

len(txin["spending_transaction_hash"].unique())


tx["recipient"]=np.repeat("A2",len(tx))
txin2=txin[["recipient","spending_transaction_hash"]]
txin2.drop_duplicates()

#txout spendable - ukazka
tx["is_spendable"].unique()
tx["time"].unique()

spend_out=txout[txout["is_spendable"]!=-1]
#většinou typ nulldata ale na konci je multsig
#value pro nulldata má 0 ale i 1 (index 1072114)

idx = txout.groupby(['transaction_hash'])['value'].transform(max) == txout['value']
t1=txout.loc[txout.groupby(["transaction_hash"])["value"].idxmax()]

#toto ale upravit
mtxout=txout.groupby('transaction_hash').idxmax("value")
dtxout=txout.groupby('transaction_hash').max("value")


pocet_transakci=len(tx["hash"])
np.repeat(0,pocet_transakci)

for i in tx["hash"]:

dtxout["tx["hash"]]


#maximalni transakce
dtxout=txout.groupby('block_id').max("value")
dttxout=txout.groupby('time').max("value")
plt.yscale('log')
dtxout["value"].plot()

#prumerne transakce
avout=txout.groupby('block_id').mean("value")
plt.yscale('linear')

avout["value"].plot()



btc_max=array(dtxout["value"]/10**8)
import matplotlib.pyplot as plt
plt.plot(y=dtxout["value"]/10**8
         ,x=np.array([1,len(dtxout["value"])]) )
plt.show()


"""
Vypreparovali jsme časy jednnotlivých bloků a zaokrouhlili je dolů na minuty
abychom je měli ve stejném formátu jak minutové ceny BTC
"""
casy=tx["time"].unique()

#string=> datetime format
dates_list = [datetime.strptime(date, "%Y-%m-%d %H:%M:%S") for date in casy]
# vteriny XX=>00
dates_list_round_minute = [date.replace(second=01) for date in dates_list]

"""
Nacteni minutovych cen BTC
mezi daty cas_start a cas_end
"""
BTC_cena = pd.read_csv('BTC_1min.txt',header=None)
BTC_cena.columns = ["cas", "open", "high", "low","closed","vol"]

BTC_cena['cas'] = pd.to_datetime(
                          BTC_cena["cas"],
                          format="%Y-%m-%d %H:%M:%S")

cas_start=datetime(year=2022,month=9,day=1,hour=0,minute=0,second=0)
cas_end=datetime(year=2022,month=9,day=1,hour=23,minute=59,second=59)

BTC_dnes=BTC_cena.loc[BTC_cena["cas"] >cas_start]
BTC_dnes=BTC_dnes.loc[BTC_dnes["cas"] <cas_end]
