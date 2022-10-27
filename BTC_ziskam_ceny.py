import os
import time
from datetime import date, timedelta,datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger
from strenum import StrEnum
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

#vyresetovani indexu
BTC_dnes=BTC_dnes.reset_index()
