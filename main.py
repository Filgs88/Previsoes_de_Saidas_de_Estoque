from datetime import datetime

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

codigo_material = int(input("Qual o código que você quer analisar: "))
data_inicial = input("A partir de qual data deve começar a analise? (dd/mm/yyyy): ")
data_inicial = datetime.strptime(data_inicial, "%d/%m/%Y")

dataset_req = pd.read_excel(
    "//192.168.1.88/Users/PC/OneDrive - MSFT/PCM/01. PCMI/25. SQLs/Requisição.xlsx"
)

dataset_req = dataset_req[
    (dataset_req["COD_MATERIAL"] == codigo_material)
    & (dataset_req["DATAREQUISICAO"] > data_inicial)
].copy()

dataset_req["PROXIMA_DATA"] = dataset_req["DATAREQUISICAO"].shift(-1)
dataset_req["NR_DIAS_PROXIMA"] = (
    dataset_req["PROXIMA_DATA"] - dataset_req["DATAREQUISICAO"]
).dt.days

df_model = dataset_req.dropna(subset=["NR_DIAS_PROXIMA"]).copy()

df_model["ANO"] = df_model["DATAREQUISICAO"].dt.year
df_model["MES"] = df_model["DATAREQUISICAO"].dt.month
df_model["DIA_DO_ANO"] = df_model["DATAREQUISICAO"].dt.dayofyear
df_model["DIA_SEMANA"] = df_model["DATAREQUISICAO"].dt.dayofweek

df_model["NR_DESDE_ANTERIOR"] = df_model["DATAREQUISICAO"].diff().dt.days
df_model["NR_DESDE_ANTERIOR"] = df_model["NR_DESDE_ANTERIOR"].fillna(
    df_model["NR_DESDE_ANTERIOR"].median()
)

df_encoded = pd.get_dummies(df_model, columns=["COD_MATERIAL"])

x = df_encoded.drop(
    columns=[
        "QUANTIDADE",
        "DATAREQUISICAO",
        "PROXIMA_DATA",
        "NR_DIAS_PROXIMA",
        "DATARETIRADA",
    ]
)
y = df_encoded["NR_DIAS_PROXIMA"]

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

modelo_dias = RandomForestRegressor(n_estimators=200, random_state=42)
modelo_dias.fit(X_train, y_train)

y_pred = modelo_dias.predict(X_test)
print("MAE (dias):", mean_absolute_error(y_test, y_pred))
