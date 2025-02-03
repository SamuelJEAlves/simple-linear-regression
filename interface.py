import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(page_title="Previsão de Custo de Casas", layout="centered")
st.title("🏡 Previsão de Custo de uma Casa a partir da Área (m²)")

# Carregamento dos dados
dados = pd.read_csv("dados/precos_casas.csv", sep=",")

X = dados[['Area (m²)']]
y = dados['Preco (R$)']

modelo = LinearRegression().fit(X, y)

# Layout organizado em abas
tabs = st.tabs(["📊 Dados", "📈 Análise", "🔢 Previsão"])

with tabs[0]:
    st.subheader("📋 Amostra dos Dados")
    st.dataframe(dados.head(10), use_container_width=True)
    st.write(f"Total de registros: {dados.shape[0]}")

with tabs[1]:
    st.subheader("📉 Gráfico de Dispersão")
    fig, ax = plt.subplots()
    ax.scatter(X, y, color='blue', label="Dados reais")
    ax.plot(X, modelo.predict(X), color='red', linewidth=2, label="Linha de Regressão")
    ax.set_xlabel("Área (m²)")
    ax.set_ylabel("Preço (R$)")
    ax.legend()
    st.pyplot(fig)

with tabs[2]:
    st.subheader("💰 Estimativa de Preço")
    novo_valor = st.number_input("Insira a área da casa em m²", min_value=1.0, max_value=999999.0, step=0.01)
    if st.button("📊 Calcular Preço"):
        dados_novo_valor = pd.DataFrame([[novo_valor]], columns=['Area (m²)'])
        prev = modelo.predict(dados_novo_valor)
        st.success(f"💲 Preço Estimado: R$ {prev[0]:,.2f}")
