# Employee Attrition Prediction
Proyecto de ciencia de datos para predecir rotación de empleados (Attrition) a partir de variables demográficas, laborales y de satisfacción, siguiendo un flujo de trabajo completo:

- Análisis exploratorio de datos (EDA)
- Construcción de variables de negocio (satisfacción compuesta, arraigo, burnout, etc.)
- Entrenamiento y evaluación de modelos de Machine Learning
- Trazabilidad de experimentos con MLflow
- API de inferencia con FastAPI
- Containerización con Docker


---

### 1. Requisitos
- Python 3.9+ (desarrollado con Python 3.11)
- pip instalado
- (Opcional pero recomendado) Virtualenv / venv
- JupyterNotebook instalado
- Las dependencias del proyecto están definidas en requirements.txt
(ej.: pandas, scikit-learn, xgboost, mlflow, fastapi, etc.).

### 2.Estructura del proyecto

proyecto_first/
├── data/
│   └── attrition_dataset.pkl       # Dataset de empleados
├── entrega_obligatoria/
│   └── best_model.pkl       # Mejor modelo
│   └── entrega_final.ipynb       # Notebbok de entrega final, conteiene todo el eda, feature_eng y models
├── mlruns/                          # Se crea automáticamente al correr el pipeline
├── notebooks/
│   ├── eda.ipynb                   # Análisis exploratorio de datos
│   ├── feature_eng.ipynb           # Exploración y validación de features
│   ├── models.ipynb                # Experimentación y selección de modelos
│   └── best_model.pkl              # Pipeline entrenado exportado
├── src/
│   ├── data/
│   │   └── loader.py               # Lectura del dataset
│   ├── features/
│   │   └── engineering.py          # Feature engineering (8 variables derivadas)
│   ├── models/
│   │   ├── train.py                # Entrenamiento, evaluación y tuning
│   │   └── predict.py              # Inferencia
│   └── preprocessing/
│       └── pipeline.py             # Construcción de pipelines de sklearn
├── api/
│   ├── main.py                     # API FastAPI para predicción en tiempo real
│   └── schemas.py                  # Schemas de entrada/salida (Pydantic)
├── main.py                         # Script principal: orquesta todo el flujo
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
└── requirements.txt

Comentario: Se crea la carpeta entrega_obligatoria que conteine los dos archivos que se piden enetregar. Entrega_final.ipynb es una union de los tres notebooks (eda, feature_eng y models). Best_model.plk es el mismo en ambas carpetas.



### 3. Flujo de trabajo
En una primera etapa se realizó el análisis exploratorio de datos en notebooks, para iterar rapido sobre ideas,identificando variables sin variabilidad, distribución del target, valores faltantes y correlaciones con Attrition.

A partir del EDA se desarrollaron 8 variables de negocio en feature_eng.ipynb, cada una con justificación conceptual y validación empírica: índice de satisfacción compuesto, ratio de antigüedad, estancamiento en promociones, ingreso relativo al nivel, estabilidad con el manager, densidad de experiencia, presión de sobrecarga laboral y score de arraigo.

Luego, en models.ipynb se llevó a cabo la experimentación con modelos de Machine Learning. Se entrenó una Regresión Logística como baseline interpretable, seguida de un XGBoost con parámetros por defecto y un XGBoost con tuning de hiperparámetros via Optuna. La evaluación combinó hold-out y 5-Fold Stratified Cross-Validation con un Pipeline completo de sklearn para garantizar la ausencia de data leakage en cada fold. Se utilizó SHAP para analizar la importancia y dirección de las features, y se realizó un análisis de errores diferenciando el costo de negocio entre falsos positivos y falsos negativos. El modelo ganador se reentrenó con la totalidad de los datos y se exportó como best_model.pkl.

Una vez seleccionado el modelo ganador, el flujo completo se migró a código modular y reproducible dentro de src/, centralizando la orquestación en main.py. Los experimentos y modelos se registran automáticamente en MLflow, promoviendo el mejor modelo como @champion en el Model Registry. Para la inferencia en tiempo real se desarrolló una API con FastAPI que carga el modelo directamente desde el registry y expone un endpoint de predicción. Finalmente, todo el entorno se containerizó con Docker para garantizar reproducibilidad y facilitar el despliegue.


### 4. Instalación y ejecución del proyecto

Opción A — Entorno virtual

Desde la raíz del proyecto (proyecto_first/):

- 1) Crear entorno virtual (se utilizó Python 3.11)
- 2) Activar entorno virtual
- 3) Instalar dependencias
- 4) Correr el pipeline de entrenamiento
        python main.py (Esto entrena los modelos, los registra en MLflow y promueve el mejor como @champion. Debe ejecutarse antes de levantar la API)

5) Levantar la API
    uvicorn api.main:app --reload

Opcion B - Docker (Recomendado)

Requiere tener Docker y Docker Compose instalados

- 1) Construir y levantar los servicios
        docker-compose up --build

Opción B — Docker
Requiere tener Docker y Docker Compose instalados. No requiere crear entorno virtual ni instalar dependencias manualmente.

1) Construir y levantar los servicios
docker-compose up --build
Esto levanta 3 contenedores

| Servicio | Descripción | Puerto |
|---|---|---|
| `train` | Corre `main.py`: entrena los modelos y los registra en MLflow | — |
| `api` | Levanta la API FastAPI para inferencia en tiempo real | `8000` |
| `mlflow` | Levanta la UI de MLflow para visualizar experimentos | `5000` |


Los servicios api y mlflow esperan a que train finalice correctamente antes de iniciarse. Los tres comparten un volumen común (mlflow_data) donde se persisten el modelo y la base de datos de experimentos.

- 2) Verificar que la API está corriendo
curl http://localhost:8000/health

- 3) Ver experimentos en MLflow
http://localhost:5000

- 4) API de inferencia
La API expone dos endpoints:

- GET /health — verificación del estado
- POST /predict — predicción de Attrition para un empleado. POST http://localhost:8000/predict

Se le pasa un JSON con los datos crudos del empleado. La API aplica el feature engineering internamente antes de realizar la predicción, por lo que no es necesario calcular ninguna variable derivada.

Request ex:
{
  "hrs": 8.5,
  "absences": 10,
  "JobInvolvement": 3.0,
  "PerformanceRating": 3,
  "EnvironmentSatisfaction": 2.0,
  "JobSatisfaction": 3.0,
  "WorkLifeBalance": 2.0,
  "Age": 32.0,
  "BusinessTravel": "Travel_Rarely",
  "Department": "Research & Development",
  "DistanceFromHome": 5,
  "Education": 3,
  "EducationField": "Medical",
  "Gender": "Male",
  "JobLevel": 2,
  "JobRole": "Research Scientist",
  "MaritalStatus": "Single",
  "MonthlyIncome": 55000.0,
  "NumCompaniesWorked": 2.0,
  "PercentSalaryHike": 14,
  "StockOptionLevel": 1,
  "TotalWorkingYears": 8.0,
  "TrainingTimesLastYear": 3,
  "YearsAtCompany": 4,
  "YearsSinceLastPromotion": 1,
  "YearsWithCurrManager": 2
}
Response
{
  "prediction": 1,
  "prediction_label": "Yes",
  "probability": 0.9888
}

#### Aclaracion: 
- Para reducir el tiempo de construcción, el tuning de hiperparámetros con Optuna está configurado con 3 trials (n_trials=3). 


- 5) Bajar los servicios
docker-compose down


#### 5. Conclusión 

Si bien el desafío priorizaba modelos sencillos con buena fundamentación metodológica, se optó por maximizar la performance predictiva cuando los datos lo justificaban. El modelo final superó al baseline lineal en 17 puntos porcentuales de AUC-ROC, con resultados estables en la evaluación con 5-Fold Cross-Validation.

Más allá del análisis, el objetivo del proyecto fue construir una solución de punta a punta: desde el EDA hasta el modelo en producción. El pipeline está completamente automatizado, los experimentos son trazables via MLflow y la API de inferencia puede levantarse desde cualquier máquina con un solo comando, sin necesidad de configurar entornos ni instalar dependencias manualmente gracias a la containerización con Docker.