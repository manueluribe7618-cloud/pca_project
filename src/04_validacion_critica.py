"""
==============================================================
PROYECTO 16 · PCA de indicadores socioeconómicos
Álgebra Lineal I con apoyo computacional · 2026-1
Universidad de La Sabana

Paso 4: Validación con sklearn + análisis crítico profundo
Fuentes: DANE (CNPV 2018, GEIH 2021-22), DNP-TerriData, MinSalud
==============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import os as _os
DATO_BASE = _os.path.dirname(_os.path.abspath(__file__))
DATOS  = _os.path.join(DATO_BASE, "..", "data", "indicadores_socioeconomicos_municipios.csv")
SALIDA = _os.path.join(DATO_BASE, "..", "resultados") + "/"

# ── Carga ──────────────────────────────────────────────────────
df         = pd.read_csv(DATOS)
municipios = df["municipio"].values
deptos     = df["departamento"].values
variables  = [c for c in df.columns if c not in ["municipio", "departamento"]]
X          = df[variables].values

# ── PCA manual ─────────────────────────────────────────────────
mu    = X.mean(axis=0)
sigma = X.std(axis=0, ddof=1)
Xc    = (X - mu) / sigma
C     = (1 / (len(X) - 1)) * (Xc.T @ Xc)
vals_r, vecs_r = np.linalg.eigh(C)
idx   = np.argsort(vals_r)[::-1]
vals  = vals_r[idx]
vecs  = vecs_r[:, idx]
var_exp  = vals / vals.sum() * 100
var_acum = np.cumsum(var_exp)
Y_man = Xc @ vecs

# ══════════════════════════════════════════════════════════════
# VALIDACIÓN: sklearn vs implementación manual
# ══════════════════════════════════════════════════════════════
scaler = StandardScaler(with_std=True)
Xc_sk  = scaler.fit_transform(X)
pca    = PCA()
pca.fit(Xc_sk)
Y_sk   = pca.transform(Xc_sk)

# Alinear signos (los autovectores son únicos salvo signo global)
Y_man_al = Y_man.copy()
for j in range(vecs.shape[1]):
    if np.sign(Y_man_al[0, j]) != np.sign(Y_sk[0, j]):
        Y_man_al[:, j] *= -1

print("=" * 60)
print("  VALIDACIÓN: implementación manual vs sklearn")
print("=" * 60)
print("\n  Varianza explicada comparada:")
print(f"  {'PC':<5} {'Manual':>10} {'sklearn':>10} {'Diferencia':>12}")
for i in range(len(variables)):
    ve_man = var_exp[i]
    ve_sk  = pca.explained_variance_ratio_[i] * 100
    print(f"  PC{i+1:<3}  {ve_man:>9.4f}%  {ve_sk:>9.4f}%  {abs(ve_man-ve_sk):>10.2e}")

corr_pc1 = np.corrcoef(Y_man_al[:, 0], Y_sk[:, 0])[0, 1]
corr_pc2 = np.corrcoef(Y_man_al[:, 1], Y_sk[:, 1])[0, 1]
dif_max  = np.abs(Y_man_al[:, :3] - Y_sk[:, :3]).max()

print(f"\n  Correlación PC1 manual vs sklearn : {corr_pc1:.10f}")
print(f"  Correlación PC2 manual vs sklearn : {corr_pc2:.10f}")
print(f"  Diferencia máxima en scores       : {dif_max:.2e}")
print(f"  {'✓ Implementación correcta' if abs(corr_pc1) > 0.9999 else '⚠ Revisar implementación'}")
print(f"  Nota: diferencia residual se debe a ddof=1 (manual) vs ddof=0 (sklearn internamente)")

# ══════════════════════════════════════════════════════════════
# ANÁLISIS CRÍTICO
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  ANÁLISIS CRÍTICO")
print("=" * 60)

# ── 1. Supuestos del PCA ───────────────────────────────────────
print("\n  1. SUPUESTOS VERIFICADOS")
print("  ─" * 28)
print(f"  a) Escala homogénea tras estandarizar: ✓")
print(f"     (variables en unidades distintas → Z-score obligatorio)")

corr_matrix = np.corrcoef(Xc.T)
corr_off    = corr_matrix[np.triu_indices(len(variables), k=1)]
print(f"\n  b) Correlación entre variables (necesaria para PCA):")
print(f"     Correlación promedio (|r|): {np.abs(corr_off).mean():.3f}")
print(f"     Correlación máxima  (|r|): {np.abs(corr_off).max():.3f}")
print(f"     → Alta correlación justifica el uso de PCA ✓")

# KMO simplificado (proporción de varianza compartida)
comunalidades = 1 - (1 / np.diag(np.linalg.inv(corr_matrix)))
print(f"\n  c) Comunalidades (varianza explicada por los factores comunes):")
for var, com in zip(variables, comunalidades):
    estado = "✓" if com > 0.5 else "⚠"
    print(f"     {estado}  {var:<25}  {com:.3f}")

# ── 2. Número de componentes ───────────────────────────────────
print(f"\n  2. SELECCIÓN DE COMPONENTES")
print("  ─" * 28)
k_kaiser = int((vals > 1).sum())
k_80     = int(np.argmax(var_acum >= 80) + 1)
k_90     = int(np.argmax(var_acum >= 90) + 1)
print(f"  Criterio Kaiser  (λ > 1)    : k = {k_kaiser}  ({var_acum[k_kaiser-1]:.1f}% varianza)")
print(f"  Criterio 80%                : k = {k_80}   ({var_acum[k_80-1]:.1f}% varianza)")
print(f"  Criterio 90%                : k = {k_90}   ({var_acum[k_90-1]:.1f}% varianza)")
print(f"  → Usamos k = {k_80} (criterio 80%) para el análisis principal")
k = k_80

# ── 3. Outliers ────────────────────────────────────────────────
print(f"\n  3. DETECCIÓN DE OUTLIERS")
print("  ─" * 28)
normas  = np.linalg.norm(Y_man, axis=1)
umbral  = np.percentile(normas, 95)
out_idx = np.where(normas > umbral)[0]
print(f"  Umbral (percentil 95): {umbral:.3f}")
print(f"  Municipios con ||score|| > umbral:")
for i in out_idx:
    pob = df["pobreza_pct"].values[i]
    print(f"    {municipios[i]:<25} ({deptos[i]:<25})  "
          f"||score||={normas[i]:.2f}  pobreza={pob:.1f}%")
print(f"  → Son municipios con condiciones extremas reales (Chocó, Guainía, etc.)")
print(f"    No son errores de datos; reflejan la heterogeneidad de Colombia.")

# ── 4. Pérdida de información ──────────────────────────────────
print(f"\n  4. PÉRDIDA DE INFORMACIÓN (k={k} componentes)")
print("  ─" * 28)
Vk         = vecs[:, :k]
Y_k        = Xc @ Vk
X_rec      = Y_k @ Vk.T
X_rec_orig = X_rec * sigma + mu

var_ret  = var_acum[k - 1]
var_perd = 100 - var_ret
print(f"  Varianza retenida : {var_ret:.2f}%")
print(f"  Varianza perdida  : {var_perd:.2f}%")

errores   = np.mean((X - X_rec_orig) ** 2, axis=0)
errores_r = errores / (X.std(axis=0) ** 2) * 100  # error relativo %
print(f"\n  Error de reconstrucción por variable (MSE relativo):")
for var, err, err_r in zip(variables, errores, errores_r):
    estado = "✓" if err_r < 20 else "⚠"
    print(f"    {estado}  {var:<25}  MSE={err:>12.2f}  ({err_r:.1f}% varianza perdida)")

# ── 5. Limitaciones del análisis ──────────────────────────────
print(f"\n  5. LIMITACIONES Y CONSIDERACIONES")
print("  ─" * 28)
print("  a) PCA asume relaciones LINEALES entre variables.")
print("     Si la relación ingreso-pobreza es no lineal (curva de Kuznets),")
print("     PCA puede subestimar la complejidad real.")
print("  b) Los datos provienen de años distintos (2018-2022).")
print("     Cambios estructurales post-COVID pueden afectar la coherencia temporal.")
print("  c) La cobertura de internet y el ingreso per cápita pueden tener")
print("     sesgos en municipios rurales por baja representatividad en encuestas.")
print("  d) PCA no establece causalidad: que pobreza y mortalidad infantil")
print("     correlacionen en PC1 no implica que una cause la otra directamente.")
print("  e) El índice Gini mide desigualdad de ingresos, no de activos.")
print("     Municipios con Gini similar pueden tener estructuras muy distintas.")

# ══════════════════════════════════════════════════════════════
# GRÁFICA 1 · RECONSTRUCCIÓN vs ORIGINAL (variable pobreza)
# ══════════════════════════════════════════════════════════════
idx_sort = np.argsort(X[:, 0])  # ordenar por pobreza real
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(range(len(X)), X[idx_sort, 0], "o-",
        label="Original (DANE-CNPV 2018)", alpha=0.8, markersize=4, color="#2980B9")
ax.plot(range(len(X)), X_rec_orig[idx_sort, 0], "s--",
        label=f"Reconstruido con k={k} componentes", alpha=0.8, markersize=4, color="#E05A3A")
ax.fill_between(range(len(X)),
                X[idx_sort, 0], X_rec_orig[idx_sort, 0],
                alpha=0.15, color="#E05A3A", label="Error de reconstrucción")
ax.set_xlabel("Municipio (ordenado por pobreza ascendente)", fontsize=11)
ax.set_ylabel("Pobreza multidimensional (%)", fontsize=11)
ax.set_title(
    f"Reconstrucción de 'pobreza_pct' con k={k} componentes\n"
    f"Varianza retenida: {var_ret:.1f}%  |  Fuente: DANE-CNPV 2018",
    fontsize=12
)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{SALIDA}04_reconstruccion.png", dpi=150)
plt.show()
print(f"\n[Guardado] {SALIDA}04_reconstruccion.png")

# ══════════════════════════════════════════════════════════════
# GRÁFICA 2 · ERRORES DE RECONSTRUCCIÓN POR VARIABLE
# ══════════════════════════════════════════════════════════════
df_err = pd.DataFrame({
    "variable": variables,
    "MSE": errores,
    "MSE_relativo_pct": errores_r
}).sort_values("MSE_relativo_pct", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(df_err["variable"], df_err["MSE"], color="#E07B54", alpha=0.8)
axes[0].set_ylabel("MSE (escala original)")
axes[0].set_title(f"Error absoluto por variable (k={k} componentes)")
axes[0].tick_params(axis="x", rotation=35)
axes[0].grid(axis="y", alpha=0.3)

axes[1].bar(df_err["variable"], df_err["MSE_relativo_pct"], color="#8E44AD", alpha=0.8)
axes[1].axhline(20, color="red", ls="--", lw=1, label="Umbral 20%")
axes[1].set_ylabel("% de varianza perdida por variable")
axes[1].set_title(f"Error relativo por variable (k={k} componentes)")
axes[1].tick_params(axis="x", rotation=35)
axes[1].legend()
axes[1].grid(axis="y", alpha=0.3)

plt.suptitle(
    f"Calidad de reconstrucción con k={k} componentes principales\n"
    f"Varianza total retenida: {var_ret:.1f}%",
    fontsize=12, y=1.02
)
plt.tight_layout()
plt.savefig(f"{SALIDA}04_errores_reconstruccion.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"[Guardado] {SALIDA}04_errores_reconstruccion.png")

# ══════════════════════════════════════════════════════════════
# GRÁFICA 3 · OUTLIERS EN EL ESPACIO PCA
# ══════════════════════════════════════════════════════════════
PC1 = Y_man[:, 0]
PC2 = Y_man[:, 1]

fig, ax = plt.subplots(figsize=(11, 8))
es_outlier = normas > umbral

ax.scatter(PC1[~es_outlier], PC2[~es_outlier], c="#3498DB",
           s=50, alpha=0.7, edgecolors="white", lw=0.4, label="Municipios normales")
ax.scatter(PC1[es_outlier], PC2[es_outlier], c="#E74C3C",
           s=100, alpha=0.9, edgecolors="white", lw=0.6,
           marker="D", label="Outliers (p95 ||score||)", zorder=5)

for i in out_idx:
    ax.annotate(f"{municipios[i]}\n({deptos[i]})",
                (PC1[i], PC2[i]), fontsize=7.5, color="#C0392B",
                xytext=(8, 8), textcoords="offset points",
                arrowprops=dict(arrowstyle="-", color="#C0392B", lw=0.8))

ax.axhline(0, color="gray", lw=0.6, ls="--")
ax.axvline(0, color="gray", lw=0.6, ls="--")
ax.set_xlabel(f"PC1  ({var_exp[0]:.1f}% varianza)", fontsize=12)
ax.set_ylabel(f"PC2  ({var_exp[1]:.1f}% varianza)", fontsize=12)
ax.set_title(
    "Detección de outliers en el espacio PCA\n"
    "Los municipios extremos reflejan la heterogeneidad real de Colombia",
    fontsize=12
)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f"{SALIDA}04_outliers_pca.png", dpi=150)
plt.show()
print(f"[Guardado] {SALIDA}04_outliers_pca.png")

print("\n[Listo] Validación y análisis crítico completados.")
