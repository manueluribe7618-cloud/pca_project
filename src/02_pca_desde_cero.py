"""
==============================================================
PROYECTO 16 · PCA de indicadores socioeconómicos
Álgebra Lineal I con apoyo computacional · 2026-1
Universidad de La Sabana

Paso 2: PCA desde cero (teoría matricial explícita)
Fuentes: DANE (CNPV 2018, GEIH 2021-22), DNP-TerriData, MinSalud
==============================================================

Teoría implementada:
  1. Estandarización:     Xc = (X - μ) / σ
  2. Matriz covarianza:   C  = (1/(n-1)) · Xc^T · Xc   ∈ R^(p×p)
  3. Descomposición esp.: C · v_k = λ_k · v_k
  4. Varianza explicada:  VE_k = λ_k / Σλ_j × 100%
  5. Proyección:          Y = Xc · V_k                  ∈ R^(n×k)
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
df         = pd.read_csv(DATOS)
municipios = df["municipio"].values
deptos     = df["departamento"].values
variables  = [c for c in df.columns if c not in ["municipio", "departamento"]]
X          = df[variables].values       # matriz n×p
n, p       = X.shape
print(f"Datos: n={n} municipios, p={p} variables")
print(f"Variables: {variables}\n")

# ══════════════════════════════════════════════════════════════
# PASO 2A · ESTANDARIZACIÓN (Z-score)
# Necesaria porque las unidades son heterogéneas:
# ingreso en millones COP, Gini en 0-1, pobreza en %
# ══════════════════════════════════════════════════════════════
mu    = X.mean(axis=0)         # μ ∈ R^p calcula un promedio de cada variable
sigma = X.std(axis=0, ddof=1)  # desviacion estandar de cada variable
Xc    = (X - mu) / sigma       # Aqui se hace una estandarizacion

print("── Verificación de estandarización ──────────────────────")
print(f"  Medias de Xc  (deben ser ≈ 0): {Xc.mean(axis=0).round(10)}")
print(f"  Desv. de Xc   (deben ser ≈ 1): {Xc.std(axis=0, ddof=1).round(10)}\n")

# ══════════════════════════════════════════════════════════════
# PASO 2B · MATRIZ DE COVARIANZA
# C = (1/(n-1)) · Xc^T · Xc   →  simétrica semidefinida positiva
# Al trabajar con datos estandarizados, C es también la
# matriz de correlación de Pearson entre las variables.
# ══════════════════════════════════════════════════════════════
C    = (1 / (n - 1)) * (Xc.T @ Xc) # Como se relacion las variables entre si
C_np = np.cov(Xc, rowvar=False)
assert np.allclose(C, C_np, atol=1e-10), "Error: matrices de covarianza difieren"

print("── Matriz de covarianza C (p×p) ──────────────────────────")
df_C = pd.DataFrame(C, index=variables, columns=variables)
print(df_C.round(4).to_string())
print(f"\n  Dimensión : {C.shape}")
print(f"  Simétrica : {np.allclose(C, C.T)}")
print(f"  Autovalores todos ≥ 0 (semidefinida positiva): "
      f"{(np.linalg.eigvalsh(C) >= -1e-10).all()}\n")

# ══════════════════════════════════════════════════════════════
# PASO 2C · DESCOMPOSICIÓN ESPECTRAL
# np.linalg.eigh: garantiza autovalores reales y autovectores
# ortonormales porque C es simétrica.
# ATENCIÓN: eigh devuelve en orden ASCENDENTE → invertimos.
# ══════════════════════════════════════════════════════════════
vals_raw, vecs_raw = np.linalg.eigh(C)

idx  = np.argsort(vals_raw)[::-1]   # orden descendente
vals = vals_raw[idx]                 # autovalores λ_1 ≥ λ_2 ≥ ... ≥ λ_p
vecs = vecs_raw[:, idx]             # columna k = autovector de PC_k

var_exp  = vals / vals.sum() * 100
var_acum = np.cumsum(var_exp)

print("── Autovalores y varianza explicada ──────────────────────")
for i in range(p):
    barra = "█" * int(var_exp[i] / 2)
    print(f"  PC{i+1}: λ={vals[i]:7.4f}  |  {var_exp[i]:5.1f}%  acum={var_acum[i]:5.1f}%  {barra}")

# Criterio de Kaiser: conservar componentes con λ > 1
k_kaiser = int((vals > 1).sum())
# Criterio 80%: mínimo de componentes que expliquen ≥ 80%
k_80     = int(np.argmax(var_acum >= 80) + 1)
k        = k_80  # usamos criterio 80%

print(f"\n  Criterio Kaiser  (λ > 1)     : k = {k_kaiser}")
print(f"  Criterio 80% var. acumulada  : k = {k_80}")
print(f"  → Se usan k = {k} componentes\n")

# Ortonormalidad de autovectores: V^T · V = I
VtV = vecs.T @ vecs
print(f"  Verificación V^T·V = I  (max desv.): {np.abs(VtV - np.eye(p)).max():.2e}\n")

# ══════════════════════════════════════════════════════════════
# PASO 2D · PROYECCIÓN (cambio de base)
# Y = Xc · V_k   →   cada fila es un municipio en la nueva base
# ══════════════════════════════════════════════════════════════
Vk = vecs[:, :k]       # matriz de cargas  p×k
Y  = Xc @ Vk           # scores (proyección) n×k

print("── Primeros 5 scores (proyección al nuevo espacio) ───────")
df_Y = pd.DataFrame(Y, columns=[f"PC{i+1}" for i in range(k)])
df_Y.insert(0, "departamento", deptos)
df_Y.insert(0, "municipio", municipios)
print(df_Y.head(8).to_string(index=False))

# Guardar resultados
df_Y.to_csv(f"{SALIDA}scores_pca.csv", index=False)
df_cargas = pd.DataFrame(vecs[:, :k], index=variables,
                         columns=[f"PC{i+1}" for i in range(k)])
df_cargas.to_csv(f"{SALIDA}cargas_pca.csv")
print(f"\n[Guardado] scores_pca.csv  y  cargas_pca.csv")

# ══════════════════════════════════════════════════════════════
# GRÁFICA 1 · SCREE PLOT
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
componentes = [f"PC{i+1}" for i in range(p)]

bars = ax.bar(componentes, var_exp, color="#4A90D9", alpha=0.75, label="Varianza explicada")
ax2  = ax.twinx()
ax2.plot(componentes, var_acum, "o-", color="#E05A3A", linewidth=2, label="Varianza acumulada")
ax2.axhline(80, color="gray", linestyle="--", linewidth=1)
ax2.text(p - 0.5, 81.5, "80%", color="gray", fontsize=10, ha="right")

for bar, ve in zip(bars, var_exp):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{ve:.1f}%", ha="center", fontsize=9)

# Marcar componentes seleccionados
for i in range(k):
    bars[i].set_edgecolor("#C0392B")
    bars[i].set_linewidth(2)

ax.set_xlabel("Componente principal", fontsize=11)
ax.set_ylabel("Varianza explicada (%)", fontsize=11, color="#4A90D9")
ax2.set_ylabel("Varianza acumulada (%)", fontsize=11, color="#E05A3A")
ax2.set_ylim(0, 110)
ax.set_title(
    f"Scree plot · Varianza explicada por componente\n"
    f"Borde rojo = componentes seleccionados (k={k}, acumulado={var_acum[k-1]:.1f}%)",
    fontsize=12
)
lineas1, lab1 = ax.get_legend_handles_labels()
lineas2, lab2 = ax2.get_legend_handles_labels()
ax.legend(lineas1 + lineas2, lab1 + lab2, loc="center right")
plt.tight_layout()
plt.savefig(f"{SALIDA}02_scree_plot.png", dpi=150)
plt.show()
print(f"[Guardado] {SALIDA}02_scree_plot.png")

# ══════════════════════════════════════════════════════════════
# GRÁFICA 2 · HEATMAP DE CARGAS
# ══════════════════════════════════════════════════════════════
# Mostramos TODAS las PCs (no solo las k seleccionadas)
# para ver la estructura completa de la descomposición
fig, ax = plt.subplots(figsize=(12, 5))
df_todas_cargas = pd.DataFrame(vecs, index=variables,
                               columns=[f"PC{i+1}" for i in range(p)])

sns.heatmap(df_todas_cargas, annot=True, fmt=".3f", cmap="RdBu_r",
            center=0, linewidths=0.5, ax=ax,
            cbar_kws={"label": "Carga (loading)"})

# Resaltar las k columnas seleccionadas
for j in range(k):
    ax.add_patch(plt.Rectangle((j, 0), 1, p,
                 fill=False, edgecolor="#C0392B", lw=2.5))

ax.set_title(
    f"Cargas de cada variable en los {p} componentes principales\n"
    f"Recuadro rojo = k={k} componentes seleccionados",
    fontsize=12
)
ax.set_xlabel("Componente principal")
ax.set_ylabel("Variable (fuente: DANE/DNP/MinSalud)")
plt.tight_layout()
plt.savefig(f"{SALIDA}02_cargas_heatmap.png", dpi=150)
plt.show()
print(f"[Guardado] {SALIDA}02_cargas_heatmap.png")

# ── Interpretación textual de PC1 ─────────────────────────────
print("\n── Interpretación de PC1 ─────────────────────────────────")
print("  PC1 captura el eje principal de desarrollo/vulnerabilidad:")
cargas_ord = sorted(zip(variables, vecs[:, 0]), key=lambda x: abs(x[1]), reverse=True)
for var, carga in cargas_ord:
    signo = "▲ mayor vulnerabilidad" if carga > 0 else "▼ mayor desarrollo"
    barra = "█" * int(abs(carga) * 20)
    print(f"    {var:<25}  {carga:+.4f}  {barra}  ({signo})")
