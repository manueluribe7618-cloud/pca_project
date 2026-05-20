# Proyecto 16 · PCA de indicadores socioeconómicos
**Álgebra Lineal I con apoyo computacional · Universidad de La Sabana · 2026-1**

---

## Estructura del proyecto

```
pca_project/
│
├── data/
│   └── indicadores_socioeconomicos_municipios.csv   ← base de datos real (100 municipios)
│
├── src/
│   ├── 01_exploracion.py        ← correlaciones y distribuciones
│   ├── 02_pca_desde_cero.py     ← PCA manual (álgebra lineal pura)
│   ├── 03_visualizacion.py      ← biplot, círculo, ranking
│   └── 04_validacion_critica.py ← validación sklearn + análisis crítico
│
├── resultados/                  ← gráficas generadas (.png) + CSVs
│   ├── 01_correlacion.png
│   ├── 01_boxplots.png
│   ├── 02_scree_plot.png
│   ├── 02_cargas_heatmap.png
│   ├── 03_biplot_pc1_pc2.png
│   ├── 03_biplot_pc1_pc3.png
│   ├── 03_ranking_municipios.png
│   ├── 03_circulo_correlaciones.png
│   ├── 04_reconstruccion.png
│   ├── 04_errores_reconstruccion.png
│   ├── scores_pca.csv
│   └── cargas_pca.csv
│
├── requirements.txt
└── README.md
```

---

## Fuentes de datos oficiales

El dataset cubre **100 municipios colombianos** seleccionados para representar toda la heterogeneidad del país: desde municipios con muy altos índices de desarrollo (Sabaneta, Envigado, Chía) hasta municipios en condiciones de alta vulnerabilidad (Quibdó, Tumaco, Mitú, Uribia). Las variables provienen de fuentes públicas oficiales:

| Variable              | Fuente oficial                                          | Año de referencia |
|-----------------------|---------------------------------------------------------|-------------------|
| `pobreza_pct`         | DANE – Medida de Pobreza Multidimensional Municipal (CNPV 2018) | 2018 |
| `desempleo_pct`       | DANE – Gran Encuesta Integrada de Hogares (GEIH)        | 2021–2022 |
| `educacion_anos`      | DANE – Censo Nacional de Población y Vivienda (CNPV)    | 2018 |
| `ingreso_capita`      | DNP – TerriData / DANE – GEIH                           | 2021 |
| `cobertura_salud`     | Ministerio de Salud – Registros SGSSS / TerriData DNP   | 2021 |
| `mortalidad_infantil` | DANE – Estadísticas vitales (defunciones y nacimientos) | 2020–2021 |
| `acceso_internet`     | DANE – Encuesta de Calidad de Vida (ECV) / MinTIC       | 2021 |
| `indice_gini`         | DANE – GEIH / DNP-TerriData                             | 2021 |

### Referencias bibliográficas

- **DANE** (2020). *Medida de Pobreza Multidimensional Municipal de fuente censal*. Departamento Administrativo Nacional de Estadísticas. https://www.dane.gov.co/index.php/estadisticas-por-tema/pobreza-y-condiciones-de-vida/pobreza-y-desigualdad/medida-de-pobreza-multidimensional-de-fuente-censal

- **DANE** (2022). *Gran Encuesta Integrada de Hogares – GEIH*. Mercado laboral e ingresos. https://www.dane.gov.co/index.php/estadisticas-por-tema/mercado-laboral/empleo-y-desempleo

- **DANE** (2019). *Censo Nacional de Población y Vivienda – CNPV 2018*. Resultados educación y hogares. https://www.dane.gov.co/index.php/estadisticas-por-tema/demografia-y-poblacion/censo-nacional-de-poblacion-y-vivenda-2018

- **DNP** (2023). *TerriData – Sistema de Estadísticas Territoriales*. Departamento Nacional de Planeación. https://terridata.dnp.gov.co/

- **Ministerio de Salud y Protección Social** (2022). *Análisis de Situación de Salud – ASIS Colombia 2022*. https://www.minsalud.gov.co/

- **MinTIC** (2022). *Boletín trimestral de las TIC*. Acceso a internet por municipios. https://mintic.gov.co/

---

## Selección y limpieza de la base de datos

### Criterio de selección de municipios
Se incluyeron 100 municipios distribuidos en los 32 departamentos del país, priorizando:
1. Las capitales departamentales (32 municipios).
2. Municipios intermedios con datos completos en todas las dimensiones.
3. Municipios con condiciones extremas (alta y baja vulnerabilidad) para garantizar varianza suficiente para el PCA.

### Proceso de limpieza
- Se verificó la ausencia de valores nulos en las 8 variables de análisis.
- Las unidades fueron homogenizadas: porcentajes en escala 0–100, ingreso per cápita en pesos colombianos corrientes, mortalidad infantil por 1.000 nacidos vivos, Gini en escala 0–1.
- Se estandarizaron los nombres de municipios según el DIVIPOLA del DANE.
- Se excluyeron municipios con información incompleta en más de 2 variables.

---

## Variables del dataset

| Variable              | Descripción                                                    | Unidad          |
|-----------------------|----------------------------------------------------------------|-----------------|
| `municipio`           | Nombre del municipio (nomenclatura DIVIPOLA-DANE)              | –               |
| `departamento`        | Departamento de pertenencia                                    | –               |
| `pobreza_pct`         | Incidencia de pobreza multidimensional                         | % hogares       |
| `desempleo_pct`       | Tasa de desempleo                                              | % PEA           |
| `educacion_anos`      | Años promedio de escolaridad (población ≥25 años)              | Años            |
| `ingreso_capita`      | Ingreso per cápita mensual                                     | COP corrientes  |
| `cobertura_salud`     | Cobertura en salud (afiliados al SGSSS)                        | % población     |
| `mortalidad_infantil` | Tasa de mortalidad infantil                                    | Por 1.000 NV    |
| `acceso_internet`     | Hogares con acceso a internet                                  | % hogares       |
| `indice_gini`         | Coeficiente de Gini (desigualdad de ingresos)                  | 0–1             |

---

## Cómo correr el proyecto

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Ejecutar los scripts en orden
```bash
cd src/
python 01_exploracion.py
python 02_pca_desde_cero.py
python 03_visualizacion.py
python 04_validacion_critica.py
```

> **Nota:** los scripts usan la columna `municipio` para etiquetas; la columna `departamento` puede usarse en `03_visualizacion.py` para colorear por región si se desea una capa adicional de análisis.

---

## Teoría matemática implementada

### 1. Estandarización (Z-score)
$$X_c = \frac{X - \mu}{\sigma}$$

Necesaria porque las variables tienen escalas muy distintas (el ingreso está en millones de pesos, el Gini en 0–1).

### 2. Matriz de covarianza
$$C = \frac{1}{n-1} X_c^T X_c \quad \in \mathbb{R}^{p \times p}$$

### 3. Descomposición espectral
$$C \mathbf{v}_k = \lambda_k \mathbf{v}_k$$

Donde $\lambda_k$ son los autovalores y $\mathbf{v}_k$ los autovectores (componentes principales).

### 4. Varianza explicada
$$\text{Var. explicada}_k = \frac{\lambda_k}{\sum_{j=1}^{p} \lambda_j} \times 100\%$$

### 5. Proyección (cambio de base)
$$Y = X_c \, V_k \quad \in \mathbb{R}^{n \times k}$$

---

## Preguntas del enunciado

1. **¿Qué porcentaje de variación explican las primeras componentes?**
   → Ver scree plot (`02_scree_plot.png`) y output de `02_pca_desde_cero.py`

2. **¿Cómo se interpreta un autovector principal?**
   → Ver heatmap de cargas (`02_cargas_heatmap.png`) y resumen impreso en `02_pca_desde_cero.py`

3. **¿Qué grupos o patrones emergen tras la proyección?**
   → Ver biplot (`03_biplot_pc1_pc2.png`) y ranking (`03_ranking_municipios.png`)

---

## Rúbrica (100 pts)

| Criterio                    | Puntos | Cubierto en                        |
|-----------------------------|--------|------------------------------------|
| Contexto y selección datos  | 15     | Este README + fuentes DANE/DNP     |
| Teoría matemática           | 20     | `02_pca_desde_cero.py`             |
| Implementación              | 20     | Todos los scripts                  |
| Resultados y análisis       | 20     | `03_visualizacion.py`              |
| Análisis crítico            | 15     | `04_validacion_critica.py`         |
| Presentación                | 10     | Informe + este README              |
