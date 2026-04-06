# %% [markdown]
# # Projet Data Science — Prostate cancer dataset
# 
# - Vérification des données,
# - Statistiques descriptives,
# - Transformation logarithmique,
# - ACP,
# - Régression linéaire simple,
# - Best subset selection,

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import statsmodels.api as sm
import statsmodels.formula.api as smf

# -----------------------------
# Plot configuration
# -----------------------------
plt.rcParams.update({
    "figure.figsize": (8, 5),
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

DATA_PATH = "data/prostate.txt"
TARGET = "lpsa"
ORIGINAL_COLUMNS = ["vol", "wht", "age", "bh", "pc", "psa"]
LOG_COLS = ["vol", "wht", "bh", "pc", "psa"]
PREDICTORS_LOG = ["lvol", "lwht", "age", "lbh", "lpc"]

# %% [markdown]
# ## 1. Chargement et contrôle du jeu de données

# %%
prostate = pd.read_csv(DATA_PATH, sep=r"\s+")

print("Aperçu :")
print(prostate.head())

print(f"Nombre d'observations : {prostate.shape[0]}")
print(f"Nombre de variables    : {prostate.shape[1]}")
print("Valeurs manquantes :")
print(prostate.isna().sum().to_frame("missing"))

missing_cols = set(ORIGINAL_COLUMNS) - set(prostate.columns)
if missing_cols:
    raise ValueError(f"Colonnes attendues manquantes : {missing_cols}")

# %% [markdown]
# ## 2. Statistiques descriptives et graphiques exploratoires

# %%
desc = prostate.describe().T

desc["variance"] = prostate.var(numeric_only=True, ddof=1)
desc["missing"] = prostate.isna().sum()

print(desc)

# %%
# Boxplots lisibles variable par variable
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.ravel()

for ax, col in zip(axes, ORIGINAL_COLUMNS):
    ax.boxplot(prostate[col].dropna(), vert=True)
    ax.set_title(f"Boxplot — {col}")
    ax.set_ylabel(col)

plt.suptitle("Distribution des variables originales", y=1.02)
plt.tight_layout()
plt.show()

# %%
# Scatterplots de psa contre les autres variables
predictors_original = ["vol", "wht", "age", "bh", "pc"]
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.ravel()

for i, col in enumerate(predictors_original):
    x = prostate[col]
    y = prostate["psa"]
    axes[i].scatter(x, y, alpha=0.75)
    corr = prostate[[col, "psa"]].corr().iloc[0, 1]
    axes[i].set_title(f"psa vs {col} (r = {corr:.2f})")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("psa")

axes[-1].axis("off")
plt.suptitle("Relation entre psa et chaque prédicteur (échelle originale)", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Transformation logarithmique
# 
# L'énoncé demande de transformer logarithmiquement toutes les variables sauf `age`. On ajoute une vérification de sécurité pour éviter les erreurs si une variable à log-transformer contient une valeur non positive.

# %%
for col in LOG_COLS:
    if (prostate[col] <= 0).any():
        bad_n = int((prostate[col] <= 0).sum())
        raise ValueError(
            f"La colonne '{col}' contient {bad_n} valeur(s) <= 0 : log impossible sans traitement spécifique."
        )

prostate_log = prostate.copy()
for col in LOG_COLS:
    prostate_log[f"l{col}"] = np.log(prostate_log[col])

prostate_log = prostate_log[["lvol", "lwht", "age", "lbh", "lpc", "lpsa"]].copy()

print("Statistiques descriptives après transformation :")
print(prostate_log.describe().T)

print("Variances après transformation :")
print(prostate_log.var(ddof=1).to_frame("variance"))

# %%
# Scatterplots après transformation
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.ravel()

for i, col in enumerate(PREDICTORS_LOG):
    x = prostate_log[col]
    y = prostate_log["lpsa"]
    axes[i].scatter(x, y, alpha=0.75)
    corr = prostate_log[[col, "lpsa"]].corr().iloc[0, 1]
    axes[i].set_title(f"lpsa vs {col} (r = {corr:.2f})")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("lpsa")

axes[-1].axis("off")
plt.suptitle("Relation entre lpsa et les prédicteurs transformés", y=1.02)
plt.tight_layout()
plt.show()

# %%
# Matrice de corrélation plus utile qu'un pairplot très chargé pour le rapport
corr = prostate_log.corr()
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr, aspect="auto")
ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr.index)))
ax.set_yticklabels(corr.index)
ax.set_title("Matrice de corrélation — variables transformées")

for i in range(corr.shape[0]):
    for j in range(corr.shape[1]):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)

fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. ACP (PCA)
# 
# Comme les variances des variables transformées restent différentes, l'ACP est réalisée **sur variables standardisées**.

# %%
X_pca = prostate_log[["lvol", "lwht", "age", "lbh", "lpc", "lpsa"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

pca = PCA()
X_scores = pca.fit_transform(X_scaled)

loadings = pd.DataFrame(
    pca.components_.T,
    index=X_pca.columns,
    columns=[f"PC{i+1}" for i in range(X_pca.shape[1])]
)

pve = pca.explained_variance_ratio_
cum_pve = np.cumsum(pve)

pve_df = pd.DataFrame({
    "Component": [f"PC{i+1}" for i in range(len(pve))],
    "PVE": pve,
    "Cumulative_PVE": cum_pve,
})

print("Loadings :")
print(loadings.round(3))

print("Variance expliquée :")
print(pve_df.round(3))

# %%
# Scree plot + courbe cumulée
components = np.arange(1, len(pve) + 1)
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(components, pve, alpha=0.8)
ax.plot(components, cum_pve, marker="o")
ax.set_xticks(components)
ax.set_xlabel("Composante principale")
ax.set_ylabel("Proportion de variance expliquée")
ax.set_title("Scree plot et variance expliquée cumulée")

for x, y in zip(components, pve):
    ax.text(x, y + 0.01, f"{y:.2f}", ha="center", fontsize=9)

plt.tight_layout()
plt.show()

# %%
# Scores plot PC1 / PC2
scores_df = pd.DataFrame(X_scores[:, :2], columns=["PC1", "PC2"])

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(scores_df["PC1"], scores_df["PC2"], alpha=0.75)
ax.axhline(0, linewidth=1)
ax.axvline(0, linewidth=1)
ax.set_title("Nuage des individus sur le plan factoriel (PC1, PC2)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
plt.tight_layout()
plt.show()

# %%
# Cercle des corrélations
# Corr(variable_j, PC_k) = loading_jk * sqrt(eigenvalue_k) pour ACP sur données standardisées
corr_circle = pd.DataFrame(
    pca.components_.T[:, :2] * np.sqrt(pca.explained_variance_[:2]),
    index=X_pca.columns,
    columns=["PC1", "PC2"]
)

fig, ax = plt.subplots(figsize=(8, 8))
unit_circle = plt.Circle((0, 0), 1, fill=False)
ax.add_patch(unit_circle)

for var in corr_circle.index:
    x, y = corr_circle.loc[var, ["PC1", "PC2"]]
    ax.arrow(0, 0, x, y, alpha=0.8, head_width=0.03, length_includes_head=True)
    ax.text(1.08 * x, 1.08 * y, var, ha="center", va="center")

ax.axhline(0, linewidth=1)
ax.axvline(0, linewidth=1)
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("Cercle des corrélations")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Rappel théorique sur \(R^2\)

# %%
print("Interprétation : R² mesure la proportion de la variance de Y expliquée par le modèle.")
print("En régression linéaire avec constante, 0 <= R² <= 1.")
print("Avec deux prédicteurs X1 et X2 :")
print("R² = (r1² + r2² - 2*r1*r2*r12) / (1 - r12²)")
print("où r1 = corr(X1, Y), r2 = corr(X2, Y), r12 = corr(X1, X2).")
print("Si X1 et X2 sont non corrélés (r12 = 0), alors R² = r1² + r2².")

# %% [markdown]
# ## 6. Régression linéaire simple

# %%
corr_with_target = prostate_log.corr()[TARGET].drop(TARGET).sort_values(ascending=False)
print("Corrélations avec lpsa :")
print(corr_with_target.to_frame("corr_lpsa"))

best_single_predictor = corr_with_target.index[0]
print(f"Variable la plus corrélée à lpsa : {best_single_predictor}")
print(f"Coefficient de corrélation        : {corr_with_target.iloc[0]:.3f}")

# %%
formula_simple = f"{TARGET} ~ {best_single_predictor}"
model_simple = smf.ols(formula_simple, data=prostate_log).fit()

print(model_simple.summary())

# %%
beta0 = model_simple.params["Intercept"]
beta1 = model_simple.params[best_single_predictor]
ci_beta1 = model_simple.conf_int(alpha=0.05).loc[best_single_predictor]

t_value = model_simple.tvalues[best_single_predictor]
p_value = model_simple.pvalues[best_single_predictor]
r2_simple = model_simple.rsquared

print(f"Modèle estimé : lpsa = {beta0:.3f} + {beta1:.3f} × {best_single_predictor}")
print(f"IC 95% de beta1 : [{ci_beta1[0]:.3f}, {ci_beta1[1]:.3f}]")
print(f"Statistique t   : {t_value:.3f}")
print(f"p-value         : {p_value:.6f}")
print(f"R²              : {r2_simple:.3f}")

# %%
# Diagnostic graphique du modèle simple
fitted_vals = model_simple.fittedvalues
residuals = model_simple.resid

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(fitted_vals, residuals, alpha=0.8)
axes[0].axhline(0, linestyle="--")
axes[0].set_title("Résidus vs valeurs ajustées")
axes[0].set_xlabel("Valeurs ajustées")
axes[0].set_ylabel("Résidus")

sm.qqplot(residuals, line="45", ax=axes[1])
axes[1].set_title("Q-Q plot des résidus")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Best subset selection pour la régression multiple

# %%
candidate_features = ["lvol", "lwht", "age", "lbh", "lpc"]
results = []

for k in range(1, len(candidate_features) + 1):
    for combo in combinations(candidate_features, k):
        predictors = list(combo)
        formula = TARGET + " ~ " + " + ".join(predictors)
        model = smf.ols(formula, data=prostate_log).fit()

        results.append({
            "n_features": k,
            "features": predictors,
            "R2": model.rsquared,
            "Adj_R2": model.rsquared_adj,
            "AIC": model.aic,
            "BIC": model.bic,
            "model": model,
        })

results_df = pd.DataFrame(results).sort_values(by="Adj_R2", ascending=False).reset_index(drop=True)
print(results_df[["n_features", "features", "R2", "Adj_R2", "AIC", "BIC"]].head(10))

# %%
# Meilleur modèle selon l'Adjusted R²
best_row = results_df.iloc[0]
best_model = best_row["model"]

print("Meilleur modèle :")
print(f"- Nombre de variables : {best_row['n_features']}")
print(f"- Variables retenues : {best_row['features']}")
print(f"- R²                 : {best_row['R2']:.3f}")
print(f"- R² ajusté          : {best_row['Adj_R2']:.3f}")
print(f"- AIC                : {best_row['AIC']:.2f}")
print(f"- BIC                : {best_row['BIC']:.2f}")

print(best_model.summary())

# %%
# Comparaison synthétique du meilleur modèle par taille
best_by_size = (
    results_df.sort_values(["n_features", "Adj_R2"], ascending=[True, False])
              .groupby("n_features", as_index=False)
              .first()
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(best_by_size["n_features"], best_by_size["Adj_R2"], marker="o")
ax.set_xticks(best_by_size["n_features"])
ax.set_xlabel("Nombre de variables")
ax.set_ylabel("Adjusted R² du meilleur modèle")
ax.set_title("Best subset selection — meilleur modèle par taille")
plt.tight_layout()
plt.show()

# %%
print("Tests de pente nulle pour le meilleur modèle :")
coef_table = pd.DataFrame({
    "coef": best_model.params,
    "t": best_model.tvalues,
    "p_value": best_model.pvalues,
    "CI_low": best_model.conf_int()[0],
    "CI_high": best_model.conf_int()[1],
})
print(coef_table.round(4))

# %% [markdown]
# ## 8. Prédiction pour un nouveau patient

# %%
new_patient_raw = {
    "vol": 7.2,
    "wht": 22,
    "bh": 1.5,
    "pc": 0.26,
    "age": 67,
}

new_patient = {
    "lvol": np.log(new_patient_raw["vol"]),
    "lwht": np.log(new_patient_raw["wht"]),
    "lbh": np.log(new_patient_raw["bh"]),
    "lpc": np.log(new_patient_raw["pc"]),
    "age": new_patient_raw["age"],
}

new_patient_df = pd.DataFrame([new_patient])
pred_lpsa = best_model.predict(new_patient_df).iloc[0]
pred_psa = np.exp(pred_lpsa)

print(f"Prédiction lpsa : {pred_lpsa:.3f}")
print(f"Prédiction psa  : {pred_psa:.3f}")


