library(readxl)
library(dplyr)
library(gdata)
Big1000 <- read_excel("C:/Users/david/Desktop/CUNIk/5_semestr/Econ/Big1000.xlsx",
                      col_names = FALSE)

Big1000[,2]
BB <- Big1000[complete.cases(Big1000[,1]),]
#BB = tabulka s 1000 nejvetšími waletkami

Big1000[,2]
jmena<- Big1000 %>% pull(2)

wal <- startsWith(jmena,"wallet",trim=TRUE)
jmena[wal] # jmena waletek

