# Diplomski rad — Status Report za mentora

## 1. Što je napravljeno

### Implementacija

20 metoda resamplinga implementirano from-scratch u Pythonu:

| Kategorija | Algoritmi | Broj |
|-----------|-----------|------|
| SMOTE varijante | SMOTE, Borderline-SMOTE (BS1, BS2), ADASYN, SafeLevel-SMOTE, KMeans-SMOTE, SVM-SMOTE, SMOTE-ENN, SMOTE-Tomek, G-SMOTE, Random-SMOTE, PolynomFit-SMOTE | 12 |
| Undersampling | NearMiss-1, NearMiss-2, NearMiss-3, Tomek Links, Edited Nearest Neighbors (ENN) | 5 |
| Baseline | NoOversampling, RandomOversampling, RandomUndersampling | 3 |

Sve SMOTE implementacije su from-scratch prema originalnim radovima. Biblioteka `imbalanced-learn` koristi se isključivo za validaciju i učitavanje datasetova.

**Validacija:** 90/95 testova prolazi (5 WGAN preskočeno — nema PyTorch). Usporedba s imbalanced-learn: 5/8 algoritama daje identičan broj sintetičkih uzoraka.

### Eksperimentalna postavka

| Parametar | Vrijednost | Obrazloženje |
|-----------|-----------|--------------|
| k | 3, 5, 7 | Chawla 2002 preporučuje k=5 |
| CV | 5-fold strat. × 10 (full run × 30) | Standard (Demšar 2006) |
| Klasifikatori | DT, RF, XGBoost, LR, SVM, kNN, GNB, MLP | 8 različitih familija |
| Metrike | F1, G-Mean, AUC-ROC (+ AUC-PR, BA, MCC, F2) | 7 metrika |
| Skupovi | 11 stvarnih + 7 sintetičkih | IR 1.7–28.1, d 3–100 |
| Statistika | Friedman + Nemenyi + Wilcoxon | Demšar 2006 |
| Ukupno | 20 × 8 × 18 × 7 = **54,432 reda** | Full run gotov |

---

## 2. Opis svakog algoritma

### SMOTE varijante (oversampling)

#### SMOTE — Synthetic Minority Over-sampling Technique (Chawla 2002)

**Ideja:** Umjesto dupliciranja postojećih manjinskih primjera (overfitting), generira nove linearne interpolacije. **Kako:** Za svaki manjinski primjer nađe k najbližih susjeda → nasumično bira jednog → generira na dužini: x_new = x_i + λ(x_nn − x_i), λ ~ U(0,1).

**Naši rezultati:** Rank #4 F1 (0.749), #5 G-Mean (0.822), #7 AUC-ROC (0.894). Robustan, nikad #1 ali uvijek u vrhu. Delta vs baseline: +0.031 F1, +0.141 G-Mean, +0.025 AUC-ROC.

---

#### Borderline-SMOTE (Han 2005)

**Ideja:** Nisu svi primjeri jednako važni — granični su ključni. **Kako:** Kategorizira manjinske u Safe/Danger/Noise prema broju većinskih susjeda. BS1 preuzorkuje samo Danger s manjinskim susjedima. BS2 dopušta i većinske uz λ∈(0,0.5).

**Naši rezultati:** Rank #12-13 F1, #8-10 G-Mean, #10-11 AUC-ROC. Konzistentno lošiji od SMOTE-a (p<0.001). Fokus na granične često pojačava šum.

---

#### ADASYN — Adaptive Synthetic Sampling (He 2008)

**Ideja:** Više sintetičkih primjera tamo gdje je većinska klasa gušća. **Kako:** Računa Γ_i = #većinskih susjeda/k, normalizira u distribuciju, raspoređuje ukupni broj sintetičkih proporcionalno.

**Naši rezultati:** Rank #11 F1, #7 G-Mean, #9 AUC-ROC. Lošiji od SMOTE-a (p<0.001). Problem: outlieri dobivaju najviše sintetičkih.

---

#### SafeLevel-SMOTE (Bunkhumpornpat 2009)

**Ideja:** Generiraj samo u "sigurnim" područjima — suprotno od Borderline/ADASYN. **Kako:** Sigurnosna razina = #manjinskih susjeda. Ako oba u nesigurnom → preskoči. Novi primjer bliže sigurnijem.

**Naši rezultati:** Rank #3 F1, #4 G-Mean, #6 AUC-ROC. **Iznenađujuće dobar!** Konzervativan pristup = stabilnost. Bolji od Borderline i ADASYN na svim metrikama.

---

#### KMeans-SMOTE (Douzas 2018)

**Ideja:** Riješiti unutar-klasni disbalans — manjinska klasa nije homogena. **Kako:** KMeans grupira manjinske → rijetki klasteri dobiju više sintetičkih (težina ∝ 1/veličina) → SMOTE unutar svakog klastera zasebno.

**Naši rezultati:** Rank #8 F1, #14 G-Mean, #13 AUC-ROC. Pomaže na high-IR: yeast_me2 +0.011, us_crime +0.017 vs SMOTE.

---

#### SVM-SMOTE (Nguyen 2011)

**Ideja:** Generiraj samo oko potpornih vektora (točaka na granici). **Kako:** SVM → izdvoji potporne vektore manjinske → SMOTE samo na njima. Preciznije od Borderline/ADASYN u identifikaciji granice, ali sporije i ovisi o kvaliteti SVM-a.

**Naši rezultati:** Rank #7 F1, #6 G-Mean, #8 AUC-ROC. Bolji od Borderline/ADASYN, ali ne nadmašuje SMOTE.

---

#### SMOTE-ENN (Batista 2004)

**Ideja:** Preuzorkuj pa očisti — prvo generiraj, onda ukloni šum. **Kako:** SMOTE → ENN: za SVAKI primjer gleda njegovih k susjeda — ako većina pripada drugoj klasi → briše ga. Briše i originalne i sintetičke, i manjinske i većinske.

**Naši rezultati:** Rank #10 F1, ali **#1 G-Mean i #1 AUC-ROC!** Agresivno čišćenje popravlja separabilnost klasa (G-Mean, AUC-ROC) ali pogoršava preciznost (F1 pad). Najbolji na šumnim sintetičkim podacima (synth_noisy +0.032).

---

#### SMOTE-Tomek (Batista 2004)

**Ideja:** Blaža verzija SMOTE-ENN — čisti samo Tomek Link parove. **Kako:** SMOTE → ukloni Tomek Linkove (dva primjera različitih klasa, međusobno najbliži). Puno manje brisanja od ENN-a.

**Naši rezultati:** Rank #2 F1, #3 G-Mean, #5 AUC-ROC. **Konzistentno u top 3!** Blago čišćenje + SMOTE = najpouzdanija kombinacija.

---

#### G-SMOTE — Geometric SMOTE (Douzas 2019)

**Ideja:** Ne generiraj samo na liniji — generiraj u cijelom geometrijskom sektoru. **Kako:** Generira unutar višedimenzionalnog sektora s faktorom deformacije α i skraćivanja. Veća raznolikost.

**Naši rezultati:** Rank #9 F1, #11 G-Mean, **#2 AUC-ROC!** Geometrijska raznolikost pomaže rank-based metrike (AUC-ROC) ali šteti precision-based (F1). Na optical_digits (d=64): +0.034 AUC-ROC vs SMOTE. Na RF-u gubi -0.020 F1.

---

#### Random-SMOTE (Dong 2011)

**Ideja:** Potpuno nasumičan smjer i udaljenost — maksimalna raznolikost. **Kako:** Za svaki manjinski primjer nasumično bira smjer (jedinični vektor) i udaljenost. Nema veze sa susjedima.

**Naši rezultati:** Rank #5 F1, #12 G-Mean, #4 AUC-ROC. **Najbolji na HIGH IR** (+0.011 F1 vs SMOTE). Posebno dobar na SVM-u (+0.014 F1). Jednostavan, a efektivan.

---

#### PolynomFit-SMOTE (Gazzah 2008)

**Ideja:** Linearna interpolacija (2 točke) je prejednostavna — polinom kroz više točaka. **Kako:** Polinomna interpolacija kroz više susjeda (degree 2-3) umjesto linearnog segmenta.

**Naši rezultati:** **Rank #1 F1 (0.751), #2 G-Mean, #3 AUC-ROC!** Neočekivano najbolji, iako slabo citiran (~40 citata). Konzistentno u top 3 na sve metrike.

---

### Undersampling algoritmi

#### NearMiss-1 (Mani & Zhang 2003)

**Ideja:** Zadrži većinske primjere koji su najbliži manjinskoj klasi — oni su najinformativniji. **Kako:** Za svaki većinski primjer računa prosječnu udaljenost do 3 najbliža manjinska. Zadržava N većinskih s najmanjom udaljenošću.

**Naši rezultati:** Rank #20 F1 (0.573), #18 G-Mean, #20 AUC-ROC. **Najgori algoritam ukupno.** Čuva samo granične većinske — ekstremno gubi informacije.

---

#### NearMiss-2 (Mani & Zhang 2003)

**Ideja:** Zadrži većinske primjere koji su najudaljeniji od manjinske klase. **Kako:** Za svaki većinski primjer računa prosječnu udaljenost do 3 NAJDALJA manjinska. Zadržava N većinskih s najvećom udaljenošću.

**Naši rezultati:** Rank #19 F1 (0.578), #16 G-Mean, #19 AUC-ROC. Malo bolji od NM-1, i dalje vrlo loš.

---

#### NearMiss-3 (Mani & Zhang 2003)

**Ideja:** Zadrži većinske primjere koji su najbliži SVIM manjinskim. **Kako:** Za svaki manjinski primjer zadržava M najbližih većinskih. "Svaki manjinski dobije svoje većinske susjede."

**Naši rezultati:** Rank #18 F1 (0.657), #15 G-Mean, #18 AUC-ROC. Najbolji od tri NearMiss varijante, ali i dalje značajno lošiji od SMOTE-a.

---

#### Tomek Links (samostalno, Tomek 1976)

**Ideja:** Ukloni dvosmislene točke s granice — parove različitih klasa koji su međusobno najbliži. **Kako:** Pronađi sve Tomek Link parove → ukloni oba primjera (ili samo većinski).

**Naši rezultati:** Rank #14 F1 (0.723), #17 G-Mean, #15 AUC-ROC. Samostalno — gubitak informacija. U kombinaciji sa SMOTE-om (SMOTE-Tomek) — odličan (#2 F1).

---

#### Edited Nearest Neighbors — ENN (samostalno, Wilson 1972)

**Ideja:** Ukloni sve primjere okružene "neprijateljima". **Kako:** Za svaki primjer gleda njegovih k susjeda. Ako većina pripada drugoj klasi → briše ga. Iterativno ili jednoprolazno.

**Naši rezultati:** Rank #17 F1 (0.689), #20 G-Mean, #17 AUC-ROC. Samostalno — preagresivno, briše previše. U kombinaciji sa SMOTE-om (SMOTE-ENN) — situacijski odličan (#1 G-Mean, #1 AUC-ROC).

---

### WGAN — Wasserstein GAN (u razvoju)

**Ideja:** Generativna suprotstavljena mreža za stvaranje visokokvalitetnih sintetičkih primjera. **Kako:** Generator stvara uzorke, kritičar procjenjuje. Wasserstein udaljenost daje stabilnije treniranje od klasičnog GAN-a.

**Izazovi vs SMOTE:** (1) Zahtijeva PyTorch i GPU, (2) premalo manjinskih primjera za treniranje GAN-a (<100), (3) mode collapse, (4) overfitting na malom broju uzoraka, (5) nestabilno treniranje, (6) 10-100× sporije.

**Status:** Implementacija u `smote_variants/gan.py`, početno stanje. Nije uključena u glavne eksperimente.

---

## 3. Detaljna analiza — F1, G-Mean, AUC-ROC

Nakon analize svih 7 metrika, fokusiramo se na 3 reprezentativne koje mjere različite aspekte performansi:
- **F1** — harmonijska sredina preciznosti i odziva (kažnjava lažne pozitive)
- **G-Mean** — geometrijska sredina odziva i specifičnosti (balans klasa)
- **AUC-ROC** — površina ispod ROC krivulje (separabilnost, neovisna o pragu)

*(Napomena: F2 je 0.942 koreliran s G-Mean-om, MCC je 0.973 koreliran s F1 — redundantni su. AUC-PR i Balanced Accuracy su u prilogu.)*

---

### 3.1. F1 — Preciznost + Odziv

**Globalni ranking:**
```
 1. PolynomFit-SMOTE          0.751
 2. SMOTE-Tomek               0.750
 3. SafeLevel-SMOTE           0.749
 4. SMOTE                     0.749  ← SMOTE
 5. Random-SMOTE              0.749
...
15. NoOversampling            0.719  ← Baseline
...
20. NearMiss-1                0.573
```

**Ključni nalazi F1:**
- Top 5 unutar 0.002 — svi praktički identični
- SMOTE +0.031 vs baseline — **postoji, ali skromno**
- NearMiss undersampling je katastrofalan (−0.146)

**Po IR grupi (F1):**
| IR grupa | SMOTE rank | Tko je bolji? |
|----------|-----------|---------------|
| LOW (<5) | #4 | NoOversampling najbolji — SMOTE ne pomaže |
| MED (5-15) | #4 | PolynomFit, SMOTE-Tomek, SafeLevel |
| HIGH (>15) | #4 | **Random-SMOTE (+0.011)**, PolynomFit (+0.004) |

**Po klasifikatoru (F1) — gdje SMOTE nije najbolji:**
| Klasifikator | Bolji od SMOTE? |
|-------------|-----------------|
| RF, LR, XGBoost | SMOTE je najbolji |
| DT | SMOTE-ENN (+0.005) |
| GNB | ENN (+0.033), TomekLinks (+0.023) |
| kNN | RandomOversampling (+0.010) |
| SVM | Random-SMOTE (+0.014) |
| MLP | Svi jednaki |

---

### 3.2. G-Mean — Balans klasa

**Globalni ranking:**
```
 1. SMOTE-ENN                 0.838
 2. PolynomFit-SMOTE          0.825
 3. SMOTE-Tomek               0.822
 4. SafeLevel-SMOTE           0.822
 5. SMOTE                     0.822  ← SMOTE
...
19. NoOversampling            0.681  ← Baseline
20. ENN                       0.618
```

**Ključni nalazi G-Mean:**
- SMOTE +0.141 vs baseline — **najveći delta od svih metrika!** SMOTE dramatično popravlja balans klasa
- SMOTE-ENN #1 — agresivno čišćenje popravlja separabilnost

**Po IR grupi (G-Mean):**
| IR grupa | SMOTE rank | Tko je bolji? |
|----------|-----------|---------------|
| LOW (<5) | **#1** | SMOTE je najbolji! |
| MED (5-15) | #4 | SMOTE-ENN (+0.015) |
| HIGH (>15) | #7 | SMOTE-ENN (+0.022), RandomUndersampling (+0.009) |

**Po klasifikatoru (G-Mean) — gdje SMOTE nije najbolji:**
| Klasifikator | Bolji od SMOTE? |
|-------------|-----------------|
| DT | SMOTE-ENN (+0.036) |
| MLP | Smote-ENN (+0.006), SVM-SMOTE (+0.005), Borderline-SMOTE2 (+0.006) |
| RF | RandomUndersampling (+0.054), SMOTE-ENN (+0.033) |
| SVM | RandomUndersampling (+0.018) |
| XGBoost | SMOTE-ENN (+0.038), RandomUndersampling (+0.037) |

**Zaključak:** SMOTE-ENN i RandomUndersampling dominiraju G-Mean na RF, SVM, XGBoost — metode koje agresivno čiste granicu odluke.

---

### 3.3. AUC-ROC — Separabilnost klasa

**Globalni ranking:**
```
 1. SMOTE-ENN                 0.897
 2. G-SMOTE                   0.896
 3. PolynomFit-SMOTE          0.895
 4. Random-SMOTE              0.895
 5. SMOTE-Tomek               0.894
 6. SafeLevel-SMOTE           0.894
 7. SMOTE                     0.894  ← SMOTE
...
16. NoOversampling            0.869  ← Baseline
...
20. NearMiss-1                0.752
```

**Ključni nalazi AUC-ROC:**
- SMOTE +0.025 vs baseline — umjereno poboljšanje
- SMOTE je tek #7 — najslabija pozicija na ovoj metrici
- Top 6 metoda su unutar 0.003 — sve praktički identične u separabilnosti

**Po IR grupi (AUC-ROC):**
| IR grupa | SMOTE rank | Tko je bolji? |
|----------|-----------|---------------|
| LOW (<5) | #2 | SMOTE-Tomek (#1, +0.000) |
| MED (5-15) | #7 | G-SMOTE (+0.003) |
| HIGH (>15) | #8 | SMOTE-ENN (+0.006), PolynomFit (+0.004) |

**Po klasifikatoru (AUC-ROC) — gdje SMOTE nije najbolji:**
| Klasifikator | Bolji od SMOTE? |
|-------------|-----------------|
| DT | SMOTE-ENN (+0.024) |
| GNB | Random-SMOTE (+0.019) |
| MLP | G-SMOTE (+0.004) |
| SVM | Random-SMOTE (+0.006), G-SMOTE (+0.004) |

---

## 4. Sažetak ključnih nalaza

### Što SMOTE postiže?

| Metrika | SMOTE | Baseline | Delta | Značaj |
|---------|-------|----------|-------|--------|
| F1 | 0.749 | 0.719 | **+0.031** | Skromno |
| G-Mean | 0.822 | 0.681 | **+0.141** | Dramatično |
| AUC-ROC | 0.894 | 0.869 | **+0.025** | Umjereno |

### Tri najbolje varijante (konzistentne kroz sve metrike)

| Varijanta | F1 | G-Mean | AUC-ROC | Prosjek |
|-----------|-----|--------|---------|---------|
| **PolynomFit-SMOTE** | #1 | #2 | #3 | **#2.0** |
| **SMOTE-Tomek** | #2 | #3 | #5 | **#3.3** |
| **SafeLevel-SMOTE** | #3 | #4 | #6 | **#4.3** |

### Koju metodu kada koristiti?

| Situacija | Preporuka |
|-----------|-----------|
| Općenito, svi datasetovi | **PolynomFit-SMOTE** ili **SMOTE-Tomek** |
| Mali IR (<5), čisti podaci | **NoOversampling** — SMOTE je suvišan |
| Visok IR (>15) | **Random-SMOTE** (F1) ili **SMOTE-ENN** (G-Mean/AUC-ROC) |
| Šumni podaci | **ENN** ili **SMOTE-ENN** (čisti šum) |
| RF, XGBoost klasifikator | **SMOTE** je dovoljan |
| SVM klasifikator | **Random-SMOTE** (+0.014 F1) |
| Želiš maksimalan G-Mean | **SMOTE-ENN** (+0.016 vs SMOTE) |

### Što NE koristiti?

- **NearMiss 1/2/3** — konzistentno najgori, gube 0.14 F1 vs SMOTE
- **Samostalni ENN** — preagresivan bez SMOTE-a (gubi 0.06 F1)
- **Borderline-SMOTE i ADASYN** — lošiji od SMOTE-a u 90% slučajeva

---

## 5. Usporedba sa završnim radom (Džoić, 2024)

| Nalaz iz završnog (2024) | Potvrda u diplomskom (2026) |
|--------------------------|---------------------------|
| k=5 optimalno | k=[3,5] dovoljno — k=7 ne donosi razliku |
| G-Mean raste s preuzorkovanjem, F1 može pasti | G-Mean +0.141, F1 samo +0.031 |
| SVM i GNB profitiraju najviše | SVM: Random-SMOTE +0.014; GNB: ENN +0.033 |
| Stablo odluke slabo profitira | DT: SMOTE tek +0.006 vs baseline |
| Parametri ovise o IR i klasifikatoru | Potvrđeno na 18 datasetova s 8 klasifikatora |
| "Bilo bi korisno istražiti druge SMOTE varijante" | **Upravo to je napravljeno** — 12 varijanti |

---

## 6. Što još treba

### Prioritet 1 — Pisanje LaTeX rada
- [x] Struktura i plan poglavlja (prva verzija)
- [ ] Poglavlje 2: Problem neuravnoteženosti i SMOTE (20-25 str)
- [ ] Poglavlje 3: Izvedenice i alternativni pristupi (10-15 str)
- [ ] Poglavlje 4: Programsko rješenje (8-10 str)
- [ ] Poglavlje 5: Eksperimentalna analiza — **najvažnije** (20-25 str)
- [ ] Poglavlje 6: Zaključak (2-3 str)

### Prioritet 2 — Završna analiza
- [x] Full run (8 klasifikatora) — gotovo
- [x] Statistička analiza — gotovo
- [x] Vizualizacije — gotovo (F:\results\figures\)
- [x] LaTeX tablice — gotovo (F:\results\tables\)

### Prioritet 3 — Opcionalno
- [ ] WGAN dovršetak (zahtijeva PyTorch)
- [ ] Parcijalno balansiranje (ne 1:1)

---

## 7. Struktura za 5-minutnu prezentaciju

1. **Što sam napravio** (30s): 20 metoda, 8 klasifikatora, 18 datasetova, 7 metrika, 54k redova
2. **Ključni nalazi** (2min):
   - SMOTE +0.141 G-Mean, +0.031 F1, +0.025 AUC-ROC — pomaže, ali metrika određuje koliko
   - Top 3 varijante: PolynomFit-SMOTE, SMOTE-Tomek, SafeLevel-SMOTE
   - NearMiss poduzorkovanje je beskorisno
3. **Demo** (1min): Web sučelje s 12 varijanti
4. **Status** (1min): Full run gotov, analiza gotova, treba napisati tekst
5. **WGAN** (30s): U razvoju, opcionalno
