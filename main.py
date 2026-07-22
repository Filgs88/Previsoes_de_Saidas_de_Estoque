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
dataset_est = pd.read_excel(
    "//192.168.1.88/Users/PC/OneDrive - MSFT/PCM/01. PCMI/25. SQLs/MateiralEstoqueGeral.xlsx"
)

dataset_req = dataset_req[
    (dataset_req["COD_MATERIAL"] == codigo_material)
    & (dataset_req["DATAREQUISICAO"] > data_inicial)
].copy()

dataset_req = dataset_req[["QUANTIDADE", "DATAREQUISICAO", "COD_MATERIAL"]].copy()

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

x = df_encoded.drop(columns=["DATAREQUISICAO", "PROXIMA_DATA", "NR_DIAS_PROXIMA"])
y = df_encoded["NR_DIAS_PROXIMA"]

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

modelo_dias = RandomForestRegressor(n_estimators=200, random_state=42)
modelo_dias.fit(X_train, y_train)

y_pred = modelo_dias.predict(X_test)
print("Tempo médio de Retirada do material. (dias):", round(mean_absolute_error(y_test, y_pred), 3))


ultima_linha = df_model.sort_values("DATAREQUISICAO").iloc[[-1]].copy()

linha_material = dataset_est[(dataset_est["COD_MATERIAL"] == codigo_material)]
qtd_estoque = linha_material["QUANTIDADE"].values[0]
qtd_media = df_model["QUANTIDADE"].mean()

data_atual = ultima_linha["DATAREQUISICAO"].values[0]
nr_desde_anterior_atual = ultima_linha["NR_DESDE_ANTERIOR"].values[0]

datas_previstas = []

while True:
    nova_linha = pd.DataFrame(
        {
            "ANO": [pd.Timestamp(data_atual).year],
            "MES": [pd.Timestamp(data_atual).month],
            "DIA_DO_ANO": [pd.Timestamp(data_atual).dayofyear],
            "DIA_SEMANA": [pd.Timestamp(data_atual).dayofweek],
            "NR_DESDE_ANTERIOR": [nr_desde_anterior_atual],
        }
    )

    for col in x.columns:
        if col.startswith("COD_MATERIAL_") and col not in nova_linha.columns:
            nova_linha[col] = 0
    col_material = f"COD_MATERIAL_{codigo_material}"
    if col_material in x.columns:
        nova_linha[col_material] = 1

    nova_linha = nova_linha.reindex(columns=x.columns, fill_value=0)

    dias_previstos = modelo_dias.predict(nova_linha)[0]

    proxima_data = pd.Timestamp(data_atual) + pd.Timedelta(days=round(dias_previstos))

    qtd_estoque = qtd_estoque - qtd_media

    if qtd_estoque <= 0:
        break

    datas_previstas.append(proxima_data)

    nr_desde_anterior_atual = dias_previstos
    data_atual = proxima_data

print(f"\nSaídas previstas do material {codigo_material}:")
print(f"\nA quantidade média de de saída é de {round(qtd_media, 3)}")
print("\nAs próximas datas de retirada serão (será):")
for data in datas_previstas:
    print(data.strftime("%d/%m/%Y"))
