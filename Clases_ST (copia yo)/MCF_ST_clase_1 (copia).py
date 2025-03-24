import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import kurtosis, skew, shapiro ,norm


# TITULO
st.title("Visualización de Rendimientos de Acciones")
# ENCABEZADO
st.header("Streamlit clase 1 ")

# Podemos escribir en el strimlit asi
# st.write('hola')

# Definimos la funcion para obtener los datos y hacemos que streamlit 
# los guarde en el cache para mejor rendimiento
@st.cache_data
def obtener_datos(stocks):
    df = yf.download(stocks, period="1y")['Close']
    return df

# Definimos la funcion para obtener los rendimientos y eliminamos los
# valores vacios, tambien guardamos en cache para mejor rendimiento
@st.cache_data
def calcular_rendimientos(df):
    return df.pct_change().dropna()

# Lista de acciones de ejemplo
stocks_lista = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']

# Muestra un mensaje de carga mientras se ejecutan las funciones
with st.spinner("Descargando datos..."):
    # Obtiene los precios de cierre de las acciones
    df_precios = obtener_datos(stocks_lista)
    # Calcula los rendimientos
    df_rendimientos = calcular_rendimientos(df_precios)

# Selector de acción
# Muestra un menú desplegable para seleccionar una acción.
stock_seleccionado = st.selectbox("Selecciona una acción", stocks_lista)

# Calculos y graficos para la acción seleccionada
if stock_seleccionado:
    ### Apartado de metricas de rendimiento ###
    st.subheader(f"Métricas de Rendimiento: {stock_seleccionado}")
    
    # Calculo de las metricas
    rendimiento_medio = df_rendimientos[stock_seleccionado].mean() # La media
    Kurtosis = kurtosis(df_rendimientos[stock_seleccionado]) #El nivel de curtosis de la distribucion
    skew = skew(df_rendimientos[stock_seleccionado]) # El nivel de asimetria de la distribucion

    # Divide la sección en 3 columnas
    col1, col2, col3= st.columns(3)
    # Muestra las metricas en las columnas en formato de tarjetas
    col1.metric("Rendimiento Medio Diario", f"{rendimiento_medio:.4%}")
    col2.metric("Kurtosis", f"{Kurtosis:.4}")
    col3.metric("Skew", f"{skew:.2}")

    ### Gráfico de rendimientos diarios ###
    st.subheader(f"Gráfico de Rendimientos: {stock_seleccionado}") 

    fig, ax = plt.subplots(figsize=(13, 5)) # Crea el grafico
    # La primera es para las fechas, la segunda para la columna de la acción y el tercero para el nombre
    ax.plot(df_rendimientos.index, df_rendimientos[stock_seleccionado], label=stock_seleccionado)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.7) #Linea en cero de color rojo
    ax.legend()
    ax.set_title(f"Rendimientos de {stock_seleccionado}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Rendimiento Diario")
    st.pyplot(fig) #Muestra el grafico en streamlit

    ### Histograma de rendimientos ###
    st.subheader("Distribución de Rendimientos")

    fig, ax = plt.subplots(figsize=(10, 5))
    # Especifica los rendimientos de la acción, el numero de columnas y color
    ax.hist(df_rendimientos[stock_seleccionado], bins=30, alpha=0.7, color='blue', edgecolor='black')
    # Muestra la linea del promedio de rendimientos
    ax.axvline(rendimiento_medio, color='red', linestyle='dashed', linewidth=2, label=f"Promedio: {rendimiento_medio:.4%}")
    ax.legend()
    ax.set_title("Histograma de Rendimientos")
    ax.set_xlabel("Rendimiento Diario")
    ax.set_ylabel("Frecuencia")
    st.pyplot(fig) #Muestra el grafico en streamlit

    ### Prueba si los rendimientos siguen una distribución Normal ###
    st.subheader("Test de Normalidad (Shapiro-Wilk)")
    stat, p = shapiro(df_rendimientos[stock_seleccionado])

    st.write(f"**Shapiro-Wilk Test Statistic:** {stat:.4f}") #Devuelve la estadistica
    st.write(f"**P-value:** {p:.4f}") #Devuelve el p value

    # Interpretación del test
    alpha = 0.05
    if p > alpha:
        st.success("La distribución parece ser normal (No se rechaza H0)") #Exito
    else:
        st.error("La distribución NO es normal (Se rechaza H0)") #Error

    ### Comparación de los rendimientos con la distribucion normal teorica ###
    st.subheader("Q-Q Plot de Rendimientos")

    fig, ax = plt.subplots(figsize=(6, 6))
    stats.probplot(df_rendimientos[stock_seleccionado], dist="norm", plot=ax)
    ax.set_title("Q-Q Plot de los Rendimientos")
    st.pyplot(fig) #Muestra el grafico en streamlit

    ### Metricas de riesgo ###

    # VaR Parametrico (suponiendo la distribucion normal) #
    mean = np.mean(df_rendimientos[stock_seleccionado])
    stdev = np.std(df_rendimientos[stock_seleccionado])
    # Calculo del percentil 5% de la distribucion Normal estimando parametros
    VaR_95 = (norm.ppf(1-0.95,mean,stdev)) 

    # VaR Historico #
    # Percentil 5% de los datos de los rendimientos
    hVaR_95 = (df_rendimientos[stock_seleccionado].quantile(0.05)) 

    # Monte Carlo #
    n_sims = 100000 # Numero de simulaciones
    sim_returns = np.random.normal(mean, stdev, n_sims) #Muestra aleatoria de rendimientos simulados
    MCVaR_95 = np.percentile(sim_returns, 5) #Percentil 5% de los rendimientos simulados

    # CVaR Calcula la pérdida esperada antes de el VaR #
    CVaR_95 = (df_rendimientos[stock_seleccionado][df_rendimientos[stock_seleccionado] <= hVaR_95].mean())

    ### Apartado para el VaR y el CVaR ###
    st.subheader("Metricas de riesgo")
    
    #Dividde la seccion en 4 columnas y mostramos cada valor en tarjetas
    col4, col5, col6, col7= st.columns(4)
    col4.metric("95% VaR Parametrico", f"{VaR_95:.4%}")
    col5.metric("95% VaR Historico", f"{hVaR_95:.4%}")
    col6.metric("Monte Carlo VaR", f"{MCVaR_95:.4%}")
    col7.metric("95% CVaR", f"{CVaR_95:.4%}")

    st.subheader("Grafica metricas de riesgo")

    # Crear la figura y el eje
    fig, ax = plt.subplots(figsize=(13, 5))

    # Generar histograma
    n, bins, patches = ax.hist(df_rendimientos[stock_seleccionado], bins=50, color='blue', alpha=0.7, label='Rendimientos')

    # Identificar y colorear de rojo las barras a la izquierda de hVaR_95
    for bin_left, bin_right, patch in zip(bins, bins[1:], patches):
        if bin_left < hVaR_95:
            patch.set_facecolor('red')

    # Marcar las líneas de VaR y CVaR
    ax.axvline(x=VaR_95, color='skyblue', linestyle='--', label='VaR 95% (Paramétrico)')
    ax.axvline(x=MCVaR_95, color='grey', linestyle='--', label='VaR 95% (Monte Carlo)')
    ax.axvline(x=hVaR_95, color='green', linestyle=':', label='VaR 95% (Histórico)')
    ax.axvline(x=CVaR_95, color='purple', linestyle='-.', label='CVaR 95%')

    # Configurar etiquetas y leyenda
    ax.set_title("Histograma de Rendimientos con VaR y CVaR")
    ax.set_xlabel("Rendimiento Diario")
    ax.set_ylabel("Frecuencia")
    ax.legend()

    # Mostrar la figura en Streamlit
    st.pyplot(fig)  
