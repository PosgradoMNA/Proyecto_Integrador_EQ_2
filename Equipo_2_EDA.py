# 1. Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Cargar los datos
datos = pd.read_csv('datos.csv')

# 3. Análisis inicial de los datos
# Ver las primeras filas del dataset
print("Primeras filas del dataset:")
print(datos.head())

# Información general del dataset
print("\nInformación del dataset:")
print(datos.info())

# Estadísticas descriptivas básicas
print("\nEstadísticas descriptivas:")
print(datos.describe())

# 4. Análisis univariante
# Para variables numéricas
def analisis_numerico(datos):
    numericas = datos.select_dtypes(include=[np.number]).columns
    for columna in numericas:
        plt.figure(figsize=(10, 6))
        # Histograma
        plt.subplot(1, 2, 1)
        sns.histplot(datos[columna], kde=True)
        plt.title(f'Histograma de {columna}')
        
        # Box plot
        plt.subplot(1, 2, 2)
        sns.boxplot(y=datos[columna])
        plt.title(f'Box Plot de {columna}')
        plt.tight_layout()
        plt.show()

# Para variables categóricas
def analisis_categorico(datos):
    categoricas = datos.select_dtypes(include=['object']).columns
    for columna in categoricas:
        plt.figure(figsize=(10, 6))
        # Gráfico de barras
        sns.countplot(data=datos, x=columna)
        plt.title(f'Distribución de {columna}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# 5. Análisis bivariante
def analisis_bivariante(datos):
    # Matriz de correlación para variables numéricas
    numericas = datos.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numericas.corr(), annot=True, cmap='coolwarm')
    plt.title('Matriz de Correlación')
    plt.tight_layout()
    plt.show()
    
    # Para variables categóricas vs numéricas
    numericas = datos.select_dtypes(include=[np.number]).columns
    categoricas = datos.select_dtypes(include=['object']).columns
    
    for num in numericas:
        for cat in categoricas:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=datos, x=cat, y=num)
            plt.title(f'{cat} vs {num}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

# 6. Ejecutar los análisis
print("\nIniciando análisis univariante...")
analisis_numerico(datos)
analisis_categorico(datos)

print("\nIniciando análisis bivariante...")
analisis_bivariante(datos)

# 7. Verificar valores faltantes
print("\nValores faltantes por columna:")
print(datos.isnull().sum())

# 8. Guardar un resumen en un archivo
with open('resumen_eda.txt', 'w') as f:
    f.write("Resumen del Análisis Exploratorio de Datos\n")
    f.write("=========================================\n\n")
    f.write("Información del Dataset:\n")
    datos.info(buf=f)
    f.write("\n\nEstadísticas Descriptivas:\n")
    f.write(datos.describe().to_string())
    f.write("\n\nValores Faltantes:\n")
    f.write(datos.isnull().sum().to_string())
