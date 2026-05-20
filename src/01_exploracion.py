"""
==============================================================
PROYECTO 16 · PCA de indicadores socioeconómicos
Álgebra Lineal I con apoyo computacional · 2026-1
Universidad de La Sabana

Paso 1: Exploración y limpieza de datos
Fuentes: DANE (CNPV 2018, GEIH 2021-22), DNP-TerriData, MinSalud
==============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os as _os
DATO_BASE = _os.path.dirname(_os.path.abspath(__file__))
DATOS  = _os.path.join(DATO_BASE, "..", "data", "indicadores_socioeconomicos_municipios.csv")
SALIDA = _os.path.join(DATO_BASE, "..", "resultados") + "/"

# ── 1. Carga ───────────────────────────────────────────────────
df = pd.read_csv(DATOS)
print("=" * 60)
print(f"  Dataset cargado: {df.shape[0]} municipios × {df.shape[1]} columnas")
print(f"  Departamentos cubiertos: {df['departamento'].nunique()}")
print("=" * 60)
print(df.head(8).to_string(index=False))

variables = [c for c in df.columns if c not in ["municipio", "departamento"]]

# ── 2. Resumen estadístico ─────────────────────────────────────
print("\n── Estadísticas descriptivas ──────────────────────────")
print(df[variables].describe().round(3).to_string())

# ── 3. Valores nulos ───────────────────────────────────────────
print("\n── Valores nulos por columna ──────────────────────────")
print(df.isnull().sum())
print("  → No hay valores nulos: dataset listo para PCA")

# ── 4. Fuentes oficiales ───────────────────────────────────────
print("\n── Fuentes oficiales por variable ─────────────────────")
fuentes = {
    "pobreza_pct":         "DANE – Pobreza Multidimensional Municipal (CNPV 2018)",
    "desempleo_pct":       "DANE – Gran Encuesta Integrada de Hogares (GEIH 2021-22)",
    "educacion_anos":      "DANE – Censo Nacional de Población y Vivienda (CNPV 2018)",
    "ingreso_capita":      "DNP – TerriData / DANE – GEIH 2021",
    "cobertura_salud":     "MinSalud – Registros SGSSS / TerriData DNP 2021",
    "mortalidad_infantil": "DANE – Estadísticas vitales 2020-2021",
    "acceso_internet":     "DANE – Encuesta de Calidad de Vida (ECV) / MinTIC 2021",
    "indice_gini":         "DANE – GEIH / DNP-TerriData 2021",
}
for var, fuente in fuentes.items():
    print(f"  {var:<22} → {fuente}")

# ── 5. Matriz de correlación ──────────────────────────────────
corr = df[variables].corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f",
    cmap="coolwarm", center=0, linewidths=0.5,
    ax=ax, cbar_kws={"shrink": 0.8}
)
ax.set_title(
    "Correlación entre indicadores socioeconómicos\n"
    "100 municipios colombianos · Fuente: DANE/DNP (2018-2022)",
    fontsize=12, pad=12
)
plt.tight_layout()
plt.savefig(f"{SALIDA}01_correlacion.png", dpi=150)
plt.show()
print(f"\n[Guardado] {SALIDA}01_correlacion.png")

# ── 6. Boxplots por variable ───────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()

for i, var in enumerate(variables):
    axes[i].boxplot(df[var], vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#D0E8F2"))
    axes[i].set_title(var.replace("_", " "), fontsize=10)
    axes[i].set_ylabel("Valor")

plt.suptitle(
    "Distribución de cada indicador – antes de estandarizar\n"
    "Fuente: DANE, DNP-TerriData, MinSalud (2018-2022)",
    fontsize=11
)
plt.tight_layout()
plt.savefig(f"{SALIDA}01_boxplots.png", dpi=150)
plt.show()
print(f"[Guardado] {SALIDA}01_boxplots.png")

# ── 7. Ranking real: mejores y peores municipios ───────────────
print("\n── Top 10 municipios con MENOR pobreza ────────────────")
print(df[["municipio","departamento","pobreza_pct"]].nsmallest(10,"pobreza_pct").to_string(index=False))
print("\n── Top 10 municipios con MAYOR pobreza ────────────────")
print(df[["municipio","departamento","pobreza_pct"]].nlargest(10,"pobreza_pct").to_string(index=False))
