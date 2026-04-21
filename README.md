#  Analyse du PSA chez des patients atteints d’un cancer de la prostate

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Project](https://img.shields.io/badge/Project-Data%20Science-orange)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

---

## 🧾 Description du projet
Ce projet de Data Science a pour objectif d’analyser les facteurs influençant le taux de **PSA (Prostate-Specific Antigen)** chez des patients atteints d’un cancer de la prostate.
L’étude repose sur un jeu de données médical réel (`prostate.txt`) contenant plusieurs variables cliniques et biologiques :

- volume tumoral (`vol`)
- poids de la prostate (`wht`)
- âge (`age`)
- hyperplasie bénigne (`bh`)
- pénétration capsulaire (`pc`)
- taux de PSA (`psa`)

Le projet suit une démarche complète d’analyse statistique :

1. Analyse descriptive des données
2. Transformation logarithmique
3. Analyse en composantes principales (ACP)
4. Régression linéaire simple et multiple
5. Sélection de variables (Best Subset Selection)
6. Prédiction pour un nouveau patient

---

## 📁 Structure du projet

```text
.
├── .venv / __pycache__             # Fichiers techniques
├── data/                           # Jeu de données (txt/csv)
├── output/                         # Graphiques générés (ACP, régression, etc.)
├── notebooks/                      # Notebook principal (analyse complète)
├── src/                            # Script Python
├── Subject.pdf                     # Sujet du projet
├── Report.pdf                      # ⚠️ Rapport final
└── requirements.txt                # Dépendances
```

👉 Le fichier Report.pdf est l’élément central du projet.

Il contient toutes les réponses aux questions du sujet, les interprétations statistiques, les conclusions de l’analyse et les figures commentées.

---

## ⚙️ Prérequis

Avant d'utiliser ce projet, assurez-vous d’avoir installé :

- Python 3.8 ou supérieur
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- statsmodels
- scipy

---

## ▶️ Installation

Clonez le dépôt :

```bash
git clone <ton_repo>
cd <ton_repo>
```

Créez un environnement virtuel :

```bash
python -m venv env
```

Activez-le :

```bash
env\Scripts\activate # Windows
source env/bin/activate # Mac / Linux 
```

Installez les dépendances :

```bash
pip install -r requirements.txt
```

Utilisation Notebook (recommandé)
```bash
jupyter notebook project_implementation.ipynb
```

---


## 📌 Résultats principaux
- lvol est la variable la plus corrélée avec lpsa
- Le modèle simple explique ≈ 61.7 % de la variance
- Le modèle multiple améliore légèrement (≈ 66 %)
- Deux composantes principales expliquent ≈ 70 % de la variance

---

## 📅 Roadmap
- [x] Analyse descriptive
- [x] Transformation des données
- [x] ACP
- [x] Régression simple
- [x] Régression multiple
- [x] Sélection de variables
- [x] Rédaction du rapport
- [ ] Optimisation du code

---


## 👤 Auteur

- **T.Angéline** - [GitHub](https://github.com/ANGELlNE) *(tamilangeline@yahoo.com)*

Pour toute question ou suggestion, n'hésitez pas à me contacter directement ou à ouvrir une issue sur le dépôt GitHub.

## 📝 Notes de fin

Merci pour votre intérêt pour ce projet :)) !

Le travail effectué illustre une démarche complète de Data Science appliquée à des données médicales réelles.

![Last Commit](https://img.shields.io/github/last-commit/ANGELlNE/ProstateCancerDataScience)
![Repo Size](https://img.shields.io/github/repo-size/ANGELlNE/ProstateCancerDataScience)
