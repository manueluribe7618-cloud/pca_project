"""
==============================================================
PROYECTO 16 · PCA de indicadores socioeconómicos
Álgebra Lineal I con apoyo computacional · 2026-1
Universidad de La Sabana

Paso 3: Visualización e interpretación de resultados
Fuente: DANE (CNPV 2018, GEIH 2021-22), DNP-TerriData, MinSalud
==============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import os as _os
DATO_BASE = _os.path.dirname(_os.path.abspath(__file__))
DATOS  = _os.path.join(DATO_BASE, "..", "data", "indicadores_socioeconomicos_municipios.csv")
SALIDA = _os.path.join(DATO_BASE, "..", "resultados") + "/"

# ── Carga y PCA ────────────────────────────────────────────────
df         = pd.read_csv(DATOS)
municipios = df["municipio"].values
deptos     = df["departamento"].values
variables  = [c for c in df.columns if c not in ["municipio", "departamento"]]
X          = df[variables].values
n, p       = X.shape

mu    = X.mean(axis=0)
sigma = X.std(axis=0, ddof=1)
Xc    = (X - mu) / sigma

C              = (1 / (n - 1)) * (Xc.T @ Xc)
vals_r, vecs_r = np.linalg.eigh(C)
idx            = np.argsort(vals_r)[::-1]
vals           = vals_r[idx]
vecs           = vecs_r[:, idx]
var_exp        = vals / vals.sum() * 100
var_acum       = np.cumsum(var_exp)

Y   = Xc @ vecs
PC1 = Y[:, 0]
PC2 = Y[:, 1]
PC3 = Y[:, 2]

# ── Clasificación por región geográfica ────────────────────────
REGIONES = {
    "Andina":    ["Antioquia","Bogotá D.C.","Boyacá","Caldas","Cundinamarca",
                  "Huila","Nariño","Norte de Santander","Quindío","Risaralda",
                  "Santander","Tolima"],
    "Caribe":    ["Atlántico","Bolívar","Cesar","Córdoba","La Guajira",
                  "Magdalena","Sucre"],
    "Pacífica":  ["Cauca","Chocó","Valle del Cauca"],
    "Orinoquía": ["Arauca","Casanare","Meta","Vichada"],
    "Amazonía":  ["Amazonas","Caquetá","Guainía","Guaviare","Putumayo","Vaupés"],
}
COLORES_REGION = {
    "Andina":    "#2980B9",
    "Caribe":    "#E67E22",
    "Pacífica":  "#27AE60",
    "Orinoquía": "#8E44AD",
    "Amazonía":  "#C0392B",
}

def get_region(depto):
    for region, lista in REGIONES.items():
        if depto in lista:
            return region
    return "Otra"

regiones_mun = np.array([get_region(d) for d in deptos])

df_scores = pd.DataFrame({
    "municipio": municipios, "departamento": deptos, "region": regiones_mun,
    "PC1": PC1, "PC2": PC2, "PC3": PC3,
    "pobreza_pct": df["pobreza_pct"].values,
    "mortalidad_infantil": df["mortalidad_infantil"].values,
})

# ══════════════════════════════════════════════════════════════
# GRÁFICA 1 · BIPLOT coloreado por REGIÓN (no por pobreza)
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 10))

for region in REGIONES:
    mask = regiones_mun == region
    if mask.sum() == 0:
        continue
    color = COLORES_REGION[region]
    ax.scatter(PC1[mask], PC2[mask], c=color, s=70, alpha=0.85,
               edgecolors="white", linewidths=0.5, label=region, zorder=3)
    if mask.sum() >= 3:
        cx, cy = PC1[mask].mean(), PC2[mask].mean()
        sx, sy = PC1[mask].std(),  PC2[mask].std()
        elipse = mpatches.Ellipse((cx, cy), 2*sx, 2*sy,
                                   fill=True, facecolor=color, alpha=0.10,
                                   edgecolor=color, linewidth=1.4,
                                   linestyle="--", zorder=2)
        ax.add_patch(elipse)
        ax.text(cx, cy + sy + 0.18, region, ha="center",
                fontsize=9, color=color, fontweight="bold")

escala = 2.8
for j, var in enumerate(variables):
    ax.annotate("", xy=(vecs[j,0]*escala, vecs[j,1]*escala), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.8))
    ox = 0.10 if vecs[j,0] >= 0 else -0.12
    oy = 0.08 if vecs[j,1] >= 0 else -0.10
    ax.text(vecs[j,0]*escala+ox, vecs[j,1]*escala+oy,
            var.replace("_", "\n"), color="#2C3E50", fontsize=8,
            fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

ax.axhline(0, color="gray", lw=0.6, ls="--")
ax.axvline(0, color="gray", lw=0.6, ls="--")
ax.set_xlabel(f"PC1  ({var_exp[0]:.1f}% varianza explicada)", fontsize=12)
ax.set_ylabel(f"PC2  ({var_exp[1]:.1f}% varianza explicada)", fontsize=12)
ax.set_title("Biplot PCA · 100 municipios colombianos por región geográfica\n"
             "Elipses = dispersión regional  |  Vectores = dirección de cada indicador",
             fontsize=12)
ax.legend(title="Región geográfica", loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig(f"{SALIDA}03_biplot_regiones.png", dpi=150)
plt.show()
print(f"[Guardado] {SALIDA}03_biplot_regiones.png")


# ══════════════════════════════════════════════════════════════
# GRÁFICA 2 · HEATMAP DE SCORES PROMEDIO POR DEPARTAMENTO
# (gráfica completamente nueva — no existía)
# ══════════════════════════════════════════════════════════════
scores_depto = (df_scores.groupby("departamento")[["PC1","PC2","PC3"]]
                .mean().sort_values("PC1"))

fig, ax = plt.subplots(figsize=(10, 11))
sns.heatmap(scores_depto, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            linewidths=0.4, ax=ax,
            cbar_kws={"label": "Score promedio en el componente", "shrink": 0.7})
ax.set_title("Score PCA promedio por departamento\n"
             "(ordenado de mejor a peor desarrollo según PC1 — Fuente: DANE/DNP)",
             fontsize=12, pad=14)
ax.set_xlabel("Componente principal")
ax.set_ylabel("Departamento")
plt.tight_layout()
plt.savefig(f"{SALIDA}03_scores_por_departamento.png", dpi=150)
plt.show()
print(f"[Guardado] {SALIDA}03_scores_por_departamento.png")


# ══════════════════════════════════════════════════════════════
# GRÁFICA 3 · RANKING MEJORADO: score + datos reales de pobreza
# ══════════════════════════════════════════════════════════════
idx_ord = np.argsort(PC1)
top10   = idx_ord[:10]
bot10   = idx_ord[-10:][::-1]

fig, axes = plt.subplots(1, 2, figsize=(17, 6))

for ax, grupo, titulo, color_bar in zip(
    axes,
    [top10, bot10],
    ["Top 10 · mayor desarrollo relativo", "Bottom 10 · mayor vulnerabilidad"],
    ["#1A7A4A", "#A93226"]
):
    etiquetas = [f"{municipios[i]} ({deptos[i]})" for i in grupo]
    pc_vals   = [PC1[i] for i in grupo]
    pobrezas  = [df["pobreza_pct"].values[i] for i in grupo]
    mortal    = [df["mortalidad_infantil"].values[i] for i in grupo]

    bars = ax.barh(etiquetas, pc_vals, color=color_bar, alpha=0.78, zorder=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.set_xlabel("Score PC1", fontsize=11)
    ax.set_title(titulo, fontsize=11, pad=10)

    for bar, v, pob, mort in zip(bars, pc_vals, pobrezas, mortal):
        ha  = "left"  if v >= 0 else "right"
        off = 0.08    if v >= 0 else -0.08
        ax.text(v + off, bar.get_y() + bar.get_height()/2,
                f"pob={pob:.0f}%  mort={mort:.1f}‰",
                va="center", fontsize=8, ha=ha, color="#2C3E50")

plt.suptitle("Ranking de municipios colombianos según PC1\n"
             "Fuente: DANE (CNPV 2018, GEIH 2021-22), DNP-TerriData",
             fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(f"{SALIDA}03_ranking_municipios.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"[Guardado] {SALIDA}03_ranking_municipios.png")


# ══════════════════════════════════════════════════════════════
# GRÁFICA 4 · BOXPLOT DE PC1 POR REGIÓN
# (gráfica completamente nueva — no existía)
# ══════════════════════════════════════════════════════════════
orden_region = (df_scores.groupby("region")["PC1"].median()
                .sort_values().index.tolist())

fig, ax = plt.subplots(figsize=(10, 6))
data_box    = [df_scores.loc[df_scores["region"]==r, "PC1"].values for r in orden_region]
colores_box = [COLORES_REGION.get(r, "#888") for r in orden_region]

bp = ax.boxplot(data_box, patch_artist=True, vert=True,
                labels=orden_region, widths=0.5)
for patch, color in zip(bp["boxes"], colores_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.65)
for median in bp["medians"]:
    median.set_color("black")
    median.set_linewidth(2)

ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
ax.set_ylabel("Score PC1  (negativo = mayor desarrollo)", fontsize=11)
ax.set_xlabel("Región geográfica de Colombia", fontsize=11)
ax.set_title("Distribución del Score PC1 por región geográfica\n"
             "Cuanto más negativo, mejores indicadores socioeconómicos",
             fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{SALIDA}03_pc1_por_region.png", dpi=150)
plt.show()
print(f"[Guardado] {SALIDA}03_pc1_por_region.png")


# ══════════════════════════════════════════════════════════════
# GRÁFICA 5 · CÍRCULO DE CORRELACIONES DOBLE (PC1-PC2 y PC1-PC3)
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, pc_x, pc_y, nom_x, nom_y in [
    (axes[0], 0, 1, f"PC1 ({var_exp[0]:.1f}%)", f"PC2 ({var_exp[1]:.1f}%)"),
    (axes[1], 0, 2, f"PC1 ({var_exp[0]:.1f}%)", f"PC3 ({var_exp[2]:.1f}%)"),
]:
    ax.add_patch(plt.Circle((0,0), 1, color="#BDC3C7", fill=False, lw=1.5, ls="--"))
    ax.add_patch(plt.Circle((0,0), 0.5, color="#F2F3F4", fill=True))

    cx_arr = vecs[:, pc_x]
    cy_arr = vecs[:, pc_y]

    for j, var in enumerate(variables):
        cx, cy = cx_arr[j], cy_arr[j]
        col = "#C0392B" if cx > 0 else "#2980B9"
        ax.annotate("", xy=(cx, cy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2, mutation_scale=12))
        ax.text(cx*1.18, cy*1.18,
                f"{var.replace('_',' ')}\n({cx:+.2f}, {cy:+.2f})",
                ha="center", fontsize=7.5, color=col, fontweight="bold",
                bbox=dict(fc="white", alpha=0.75, ec="none", pad=1))

    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel(nom_x, fontsize=10)
    ax.set_ylabel(nom_y, fontsize=10)
    ax.set_title(f"Círculo de correlaciones\n{nom_x} vs {nom_y}", fontsize=10)
    ax.set_aspect("equal")

plt.suptitle("Cargas de cada variable en los componentes principales\n"
             "Valores entre paréntesis: (carga en X, carga en Y)", fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(f"{SALIDA}03_circulo_correlaciones.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"[Guardado] {SALIDA}03_circulo_correlaciones.png")


# ══════════════════════════════════════════════════════════════
# GRÁFICA 6 · SCATTER MATRIX DE LOS 3 PRIMEROS PCs
# (completamente nueva — no existía)
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
pcs = [PC1, PC2, PC3]
pc_labels = [f"PC1 ({var_exp[0]:.1f}%)",
             f"PC2 ({var_exp[1]:.1f}%)",
             f"PC3 ({var_exp[2]:.1f}%)"]

for i in range(3):
    for j in range(3):
        ax = axes[i][j]
        if i == j:
            for region in REGIONES:
                mask = regiones_mun == region
                if mask.sum() == 0: continue
                ax.hist(pcs[i][mask], bins=10, alpha=0.5,
                        color=COLORES_REGION[region], density=True)
            ax.set_xlabel(pc_labels[i], fontsize=8)
        else:
            for region in REGIONES:
                mask = regiones_mun == region
                if mask.sum() == 0: continue
                ax.scatter(pcs[j][mask], pcs[i][mask],
                           c=COLORES_REGION[region], s=25, alpha=0.7,
                           edgecolors="white", linewidths=0.3)
            ax.set_xlabel(pc_labels[j], fontsize=8)
            ax.set_ylabel(pc_labels[i], fontsize=8)
            ax.axhline(0, color="gray", lw=0.4, ls="--")
            ax.axvline(0, color="gray", lw=0.4, ls="--")
        ax.tick_params(labelsize=7)

patches = [mpatches.Patch(color=COLORES_REGION[r], label=r) for r in REGIONES]
fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=8,
           title="Región geográfica", title_fontsize=9, bbox_to_anchor=(0.5, -0.02))
plt.suptitle("Matriz de dispersión · Primeros 3 componentes principales\n"
             "100 municipios colombianos · DANE/DNP 2018-2022", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(f"{SALIDA}03_scatter_matrix_pcs.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"[Guardado] {SALIDA}03_scatter_matrix_pcs.png")


# ══════════════════════════════════════════════════════════════
# RESUMEN IMPRESO
# ══════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("INTERPRETACIÓN DE LOS COMPONENTES")
print("="*65)
for k in range(3):
    print(f"\n  PC{k+1}  ({var_exp[k]:.1f}% varianza  |  acumulado {var_acum[k]:.1f}%):")
    cargas_ord = sorted(zip(variables, vecs[:, k]),
                        key=lambda x: abs(x[1]), reverse=True)
    for var, carga in cargas_ord:
        barra = "█" * int(abs(carga) * 20)
        signo = "▲" if carga > 0 else "▼"
        print(f"    {signo}  {var:<25}  {carga:+.4f}  {barra}")

print("\n" + "="*65)
print("SCORE PC1 PROMEDIO POR REGIÓN")
print("="*65)
orden_print = df_scores.groupby("region")["PC1"].median().sort_values().index
for region in orden_print:
    mask  = regiones_mun == region
    media = PC1[mask].mean()
    n_mun = mask.sum()
    print(f"  {region:<12}  n={n_mun:2d}  PC1_prom={media:+.3f}")
