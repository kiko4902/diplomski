# Plan vizualizacija za diplomski rad — Poglavlje 5

## Grafikoni koje uključiti u rad

### 1. Usporedba svih metoda (globalni pregled)

| # | Grafikon | Što prikazuje | Gdje u radu |
|---|----------|---------------|-------------|
| 1.1 | **Boxplot** — F1 po SMOTE varijantama | Distribucija F1 za svih 20 metoda kroz sve datasete i klasifikatore. Pokazuje medijan, kvartile, outliere. | 5.2 ili 5.3 |
| 1.2 | **Violin plot** — F1 po SMOTE varijantama | Isto kao boxplot, ali s prikazom gustoće distribucije | 5.3 |
| 1.3 | **Heatmap** — SMOTE × Dataset (F1) | Matrica: retci = dataseti, stupci = SMOTE varijante, boja = F1. Odmah se vidi tko dominira na kojem skupu. | 5.3 |
| 1.4 | **CD dijagram** (Critical Difference) | Friedman + Nemenyi — vizualno pokazuje koje su metode u istoj statističkoj grupi (povezane crtom) | 5.4 (Statistička analiza) |

### 2. SMOTE vs Baseline — koliko pomaže?

| # | Grafikon | Što prikazuje |
|---|----------|---------------|
| 2.1 | **Bar chart** — Baseline vs Najbolji SMOTE po datasetu (F1) | Dva stupca po datasetu: NoOversampling vs najbolja SMOTE varijanta. Delta označena brojkom. Odmah se vidi gdje SMOTE pomaže, a gdje ne. |
| 2.2 | **Bar chart** — Baseline vs Najbolji SMOTE po datasetu (G-Mean) | Isto za G-Mean — ovdje će razlike biti puno veće (+0.141 vs +0.031) |
| 2.3 | **Delta F1 scatter** — IR vs poboljšanje | Svaka točka = jedan dataset. X = IR, Y = delta(SMOTE - Baseline). Vidi se korelira li IR s poboljšanjem |

### 3. Po grupama — gdje tko pobjeđuje

| # | Grafikon | Što prikazuje |
|---|----------|---------------|
| 3.1 | **Grouped bar chart** — Top 5 po IR grupi | 3 grupe (LOW/MED/HIGH IR), svaka s top 5 metoda. Vidi se kako se poredak mijenja s IR-om |
| 3.2 | **Grouped bar chart** — Top 5 po klasifikatoru | 8 klasifikatora, svaki s top 3-5 metoda |
| 3.3 | **Heatmap** — Metoda × Metrika | Retci = metode, stupci = F1/G-Mean/AUC-ROC. Vidi se koje metode su specijalizirane za koju metriku |

### 4. Usporedba metrika

| # | Grafikon | Što prikazuje |
|---|----------|---------------|
| 4.1 | **Scatter** — F1 vs G-Mean (po datasetu) | Svaka točka = dataset. Vidi se odnos ove dvije metrike. Na nekim skupovima F1 visok a G-Mean nizak. |
| 4.2 | **Ranking divergence** — Bočni bar | Pokazuje koliko se rang metode razlikuje između F1, G-Mean i AUC-ROC (npr. SMOTE-ENN #1 G-Mean, #10 F1) |

### 5. Specifične usporedbe (za diskusiju)

| # | Grafikon | Što prikazuje |
|---|----------|---------------|
| 5.1 | **Line chart** — SMOTE vs njegove top 3 varijante kroz datasete | X-os = dataseti (sortirani po IR), Y = F1. 4 linije: SMOTE, PolynomFit, SMOTE-Tomek, SafeLevel |
| 5.2 | **Scatter** — originalni + sintetički primjeri (PCA 2D) | 2-3 SMOTE varijante na istom skupu. Vizualno se vidi RAZLIKA u načinu generiranja. |
| 5.3 | **Bar chart** — Broj sintetičkih uzoraka po varijanti | Neke varijante generiraju manje (ENN briše), neke više. Zanimljivo za diskusiju. |

---

## Tablice za uključivanje

| # | Tablica | Sadržaj |
|---|---------|---------|
| T1 | **Datasetovi** — karakteristike | Naziv, n_samples, n_features, IR, source, separability |
| T2 | **Metode** — popis s referencama | Svi algoritmi, godina, citat |
| T3 | **Prosječni rang** (F1) | Svih 20 metoda, poredano. Boldano top 5. |
| T4 | **Prosječni rang** (G-Mean) | Isto za G-Mean |
| T5 | **Prosječni rang** (AUC-ROC) | Isto za AUC-ROC |
| T6 | **Friedman test** — rezultati | Statistika, p-vrijednost za sve 3 metrike |
| T7 | **Nemenyi post-hoc** — p-vrijednosti | Matrica 20×20 (može u prilog) |
| T8 | **Wilcoxon vs SMOTE** — F1 | Za svaku varijantu: statistika, p, značajnost, smjer |
| T9 | **Wilcoxon vs NoOversampling** — G-Mean | Isto, ali vs baseline |
| T10 | **Per-dataset best** | Za svaki dataset: baseline F1, best SMOTE F1, delta, koja varijanta |

---

## Struktura Poglavlja 5 — prijedlog

```
5. EKSPERIMENTALNA ANALIZA
  5.1. Opis skupova podataka
       - Tablica T1
  5.2. Postavke eksperimenta
  5.3. Rezultati — Globalna usporedba
       - Grafikon 1.1 (boxplot) + Grafikon 1.4 (CD dijagram)
       - Tablica T3 (ranking F1)
       - Tablica T4 (ranking G-Mean)
       - Tablica T5 (ranking AUC-ROC)
  5.4. Statistička analiza
       - Tablica T6 (Friedman)
       - Tablica T8, T9 (Wilcoxon)
       - Tablica T7 u prilogu (Nemenyi)
  5.5. Rezultati — Analiza po grupama
       - Grafikon 3.1 (po IR grupi)
       - Grafikon 3.2 (po klasifikatoru)
       - Grafikon 2.1, 2.2 (baseline vs SMOTE)
  5.6. Diskusija
       - Grafikon 1.3 (heatmap) — pregled svega
       - Tablica T10 (per-dataset best)
       - Grafikon 4.1 (F1 vs G-Mean)
       - Preporuke: kada koristiti koju metodu
       - Usporedba s literaturom (Rojnić 2020, Kovács 2019, Džoić 2024)
       - Ograničenja
```

---

## Popis postojećih gotovih grafova (već generirani)

Svi u `figures/`:
- `boxplot_f1.pdf`, `boxplot_g_mean.pdf`, `boxplot_auc_roc.pdf`
- `violin_f1.pdf`, `violin_g_mean.pdf`, `violin_auc_roc.pdf`
- `cd_diagram_f1.pdf`, `cd_diagram_g_mean.pdf`, `cd_diagram_auc_roc.pdf`
- `heatmap_f1.pdf`, `heatmap_g_mean.pdf`, `heatmap_auc_roc.pdf`
- `per_dataset_bars_f1.pdf`, `per_dataset_bars_g_mean.pdf`, `per_dataset_bars_auc_roc.pdf`

### Što još treba generirati

- [ ] Scatter: original + synthetic točke (PCA 2D) — 2-3 varijante na istom skupu
- [ ] Line chart: SMOTE vs top 3 varijante kroz datasete
- [ ] Grouped bar: top 5 po IR grupi
- [ ] Grouped bar: top 5 po klasifikatoru
- [ ] F1 vs G-Mean scatter
- [ ] Delta F1 vs IR scatter
- [ ] Ranking divergence chart (SMOTE-ENN #1 vs #10)
