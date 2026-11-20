# Research guide (Gold CREST)

Key concepts, formulas and research documents behind the DSSC Pigment Recommender.
References marked with a link were checked during the project; the rest are classic texts and
standards cited from memory, so check each DOI before it goes into a bibliography.

---

## 1. Key concepts

**How a DSSC works** (each step is a place efficiency can be lost):
1. A dye molecule absorbs a photon, which excites one of its electrons.
2. That electron is injected into the TiO₂ conduction band (the injection part of the code's "success rate").
3. The electron travels through the porous TiO₂ to the conductive FTO glass (collection).
4. The iodide/triiodide (I⁻/I₃⁻) electrolyte gives the dye back an electron (regeneration).
5. Triiodide is reduced back to iodide at the counter electrode.

**The project's central idea: complementary absorption (co-sensitisation).** Each pigment absorbs a
different part of the spectrum:
- Chlorophyll: blue (~430 nm) and red (~665 nm)
- Anthocyanin: green (~520–530 nm)
- Curcumin: violet-blue (~425 nm)

A mix covers more of the spectrum than any single pigment. Because the spectrum changes with cloud,
haze and sun height, the best mix should change too. That is the claim the tool tests.

**How each pigment sticks to TiO₂ (the "anchoring group")** largely decides injection efficiency:
- **Anthocyanin:** catechol –OH groups bind strongly to TiO₂, one reason it is among the best natural dyes.
- **Chlorophyll:** no good anchor, so studies often use chlorophyllin or chlorophyll derivatives.
- **Curcumin:** binds through its β-diketone group.
- **Expect interactions:** pigments may compete for space on the TiO₂ in a mix, which is what Phase 7 measures.

**Tropical sunlight varies more than you might think:**
- Cloud adds diffuse, bluer light.
- Haze (aerosol) removes blue light.
- Water vapour absorbs near-infrared.
- High air mass (low sun) makes light redder.

Malaysia's humidity and seasonal haze (e.g. from regional fires) make it an especially interesting test case.

**Light saturation (why mixes can beat single pigments).** A dyed film absorbs 1 − 10^−A of the light.
This curve flattens as absorbance rises, so the last bit of one pigment adds little, while the same
amount of a different pigment fills an unabsorbed part of the spectrum. See section 5.

**Uncertainty quantification.** Monte Carlo propagation of input uncertainties, plus checks of whether
the reported uncertainty can be trusted (coverage and reliability).

**Physics-informed machine learning.** A physics model does the main work and ML learns only the
correction (Kennedy–O'Hagan calibration, Gaussian processes).

**Experimental design.** Repeats; cell-to-cell variation as a floor on how precise any prediction can be;
holding out one mix at a time for validation; never testing on training data.

---

## 2. Formulas and equations used in the project

**Light**
```
Photon energy             E = hc / λ
Photon flux (per nm)      Φ(λ) = E_λ(λ) · λ / (h c)          E_λ in W/m²/nm
Air mass (Kasten–Young)   AM = 1 / [cos z + 0.50572 (96.07995 − z)^−1.6364]     z in degrees
Ångström aerosol law      τ(λ) = β λ^−α   →   τ₅₀₀ = τ₅₅₀ · (500/550)^−α,  α ≈ 1.14
Cloud scaling (Phase 1)   E(λ) = E_dir,clear(λ) · DNI_meas,h / ∫E_dir,clear  +  E_dif,clear(λ) · DHI_meas / ∫E_dif,clear
```

**Absorption (Beer–Lambert)**
```
A = ε · c · l                      absorbance = molar absorptivity × concentration × path length
T = 10^−A                          fraction transmitted
A_mix(λ) = Σ fᵢ · Aᵢ(λ)            absorbances add in a mixture (no interaction assumed)
LHE(λ) = 1 − 10^−A_mix(λ)          light-harvesting efficiency (fraction absorbed)
share_i(λ) = fᵢAᵢ(λ) / A_mix(λ)    how much of the absorbed light each pigment takes
```

**From light to current**
```
IPCE(λ) = LHE(λ) · φ_inj · η_coll          incident photon-to-current efficiency
J_sc = q ∫ Φ(λ) · IPCE(λ) dλ               short-circuit current density
(in the code, "success rate" = φ_inj · η_coll per pigment)
```

**The cell (single-diode model)**
```
I = I_L − I₀ [exp((V + I·R_s) / (n·V_T)) − 1] − (V + I·R_s) / R_sh,     V_T = kT/q
V_oc ≈ (n·k·T / q) · ln(I_L / I₀ + 1)
FF = P_max / (V_oc · I_sc)
η = P_max / (P_in · A) = V_oc · J_sc · FF / P_in
I₀(T) = I₀,ref · (T/T_ref)³ · exp[(E_g/k)(1/T_ref − 1/T)]       (De Soto temperature model)
DSSC view of V_oc:  qV_oc = E_F,n(TiO₂) − E_redox             (electron quasi-Fermi level minus redox level)
```

**Cell temperature (Ross)**
```
T_cell = T_air + k · G,     k = (NOCT − 20) / 800,   placeholder k = 0.0208 K·m²/W
```

**Recommendation and uncertainty**
```
Mixes on a 5% grid:      C(20+2, 2) = 231
Accuracy of mix m:        acc_m = (1/N) Σ_r 1[ η_m,r ≥ 0.85 · max_k η_k,r ]
Monte Carlo error:        SE = √(p(1−p)/N)  → at N = 500, at most ±2.2 points
Tie rule:                 tied if acc_m ≥ max(acc) − 5
Coverage:                 P(η_true ∈ [P5, P95]) should be ≈ 90%
Cell-to-cell variation:   CV = s / mean of the repeats
```

**Planned ML correction (Phase 7+)**
```
Kennedy–O'Hagan:   y(x) = η_sim(x, θ) + δ(x) + ε
This project:      η(x) = η_sim(x, θ) · exp(clip(δ(x), log 0.67, log 1.5))
Gaussian process:  δ ~ GP(0, k),  k(x,x') = σ² exp(−Σ_d (x_d − x'_d)² / 2ℓ_d²) + σ_n² 1[x = x']
  posterior mean:      μ* = k*ᵀ (K + σ_n² I)⁻¹ y
  posterior variance:  σ*² = k** − k*ᵀ (K + σ_n² I)⁻¹ k*
```

---

## 3. Research documents

**DSSC fundamentals**
- O'Regan & Grätzel (1991), "A low-cost, high-efficiency solar cell based on dye-sensitized colloidal TiO₂ films", *Nature* 353, 737. doi:10.1038/353737a0. The founding paper.
- Hagfeldt, Boschloo, Sun, Kloo & Pettersson (2010), "Dye-Sensitized Solar Cells", *Chem. Rev.* 110, 6595. doi:10.1021/cr900356p. The standard deep review.
- Smestad & Grätzel (1998), "Demonstrating electron transfer and nanotechnology: a natural dye-sensitized nanocrystalline energy converter", *J. Chem. Educ.* 75, 752. doi:10.1021/ed075p752. School-lab protocol with natural dyes; useful for Phase 7.

**Natural dyes**
- [Hug, Bader, Mair & Glatzel (2014), "Biophotovoltaics: Natural pigments in dye-sensitized solar cells", *Applied Energy* 115, 216–225](https://ideas.repec.org/a/eee/appene/v115y2014icp216-225.html). doi:10.1016/j.apenergy.2013.10.055.
- [Calogero et al. (2009), "Natural dye senstizers for photoelectrochemical cells", *Energy Environ. Sci.* 2, 1162](https://pubs.rsc.org/en/Content/ArticleLanding/2009/EE/B913248C). ("senstizers" is the published spelling.)
- [Shalini et al. (2015), "Review on natural dye sensitized solar cells: Operation, materials and methods", *Renew. Sustain. Energy Rev.* 51, 1306–1325](https://ore.exeter.ac.uk/repository/handle/10871/17884). doi:10.1016/j.rser.2015.07.052. Good source of literature-table rows.
- Narayan (2012), "Review: Dye sensitized solar cells based on natural photosensitizers", *Renew. Sustain. Energy Rev.* 16, 208. doi:10.1016/j.rser.2011.07.148.
- Kay & Grätzel (1993), "Artificial photosynthesis. 1. Photosensitization of TiO₂ solar cells with chlorophyll derivatives and related natural porphyrins", *J. Phys. Chem.* 97, 6272.

**Mixing pigments (co-sensitisation): most relevant to the research question**
- Reported mixed-dye results include an anthocyanin–chlorophyll cell at 2.39% (about 2.3–4.2× the single-dye cells) and a turmeric + red spinach blend at 1.08% vs 0.38% for turmeric alone. These come from search summaries, so read the papers before citing them. Starting points: [TD-DFT study of anthocyanin/chlorophyll complexes, RSC *Materials Advances* 2025](https://pubs.rsc.org/en/content/articlelanding/2025/ma/d5ma00273g); [Mukhokosi et al. 2024, dye blend ratios and stability](https://kyuspace.kyu.ac.ug/items/f6d6b163-1dbb-45d3-a642-88d9d607fda8).
- Curcumin cells are reported at about 0.36–0.8% efficiency, with poor stability under light ([example study](https://core.ac.uk/download/351119159.pdf)).
- The roadmap says "almost no published mix data", but there is some. Use these papers for mixed-cell rows in the literature table, and to compare against the Phase 7 interaction gaps.

**Sunlight modelling**
- Bird & Riordan (1986), "Simple solar spectral model for direct and diffuse irradiance…", *J. Climate Appl. Meteor.* 25, 87–97; also NREL report TR-215-2436 (1984). SPECTRL2.
- Kasten & Young (1989), "Revised optical air mass tables and approximation formula", *Applied Optics* 28, 4735.
- Gueymard (1994), *Solar Energy* 53, 57–71. The water-vapour estimate pvlib uses.
- Holmgren, Hansen & Mikofski (2018), "pvlib python", *JOSS* 3(29), 884. doi:10.21105/joss.00884. Cite for the software.
- ASTM G173-03 reference solar spectrum (AM1.5G); IEC 60904-3.

**The cell model**
- De Soto, Klein & Beckman (2006), "Improvement and validation of a model for photovoltaic array performance", *Solar Energy* 80, 78. doi:10.1016/j.solener.2005.06.010.
- Jain & Kapoor (2004), Lambert-W solution of the diode equation, *Sol. Energy Mater. Sol. Cells* 81, 269. How pvlib solves it.
- Ross (1976), "Interface design considerations for terrestrial solar cell modules", Proc. 12th IEEE PVSC, 801–806.
- Shockley & Queisser (1961), *J. Appl. Phys.* 32, 510. Theoretical efficiency limit, for context.

**Measuring anthocyanins**
- Lee, Durst & Wrolstad (2005), pH-differential method, *J. AOAC Int.* 88, 1269 ([USDA PDF](https://www.ars.usda.gov/ARSUserFiles/37108/pdf/leejaoac.pdf)). Makes "same strength" extracts quantitative (ε = 26,900 for cyanidin-3-glucoside).
- Giusti & Wrolstad (2001), "Characterization and Measurement of Anthocyanins by UV-Visible Spectroscopy", *Current Protocols in Food Analytical Chemistry*, F1.2.
- Taniguchi & Lindsey PhotochemCAD papers (2018, 2021), cited in `cell/spectra/SOURCES.md`.

**Uncertainty and validation**
- JCGM 101:2008, *GUM Supplement 1: Propagation of distributions using a Monte Carlo method*. The international standard for the Monte Carlo approach; free from the BIPM website.
- JCGM 100:2008, *Guide to the Expression of Uncertainty in Measurement* (GUM).
- Gneiting, Balabdaoui & Raftery (2007), "Probabilistic forecasts, calibration and sharpness", *J. R. Stat. Soc. B* 69, 243. Theory behind the reliability table.
- Efron & Tibshirani (1993), *An Introduction to the Bootstrap*. For CALIBRATION_STEPS.txt Step 10a.
- Saltelli et al. (2008), *Global Sensitivity Analysis: The Primer*. A rigorous version of the Phase 8 sensitivity test.

**Physics-informed machine learning**
- Kennedy & O'Hagan (2001), "Bayesian calibration of computer models", *J. R. Stat. Soc. B* 63, 425.
- Rasmussen & Williams (2006), *Gaussian Processes for Machine Learning*, MIT Press. Free at gaussianprocess.org/gpml; chapter 2 is enough.
- Karniadakis et al. (2021), "Physics-informed machine learning", *Nature Reviews Physics* 3, 422.
- Willard et al. (2022), "Integrating scientific knowledge with machine learning for engineering and environmental systems", *ACM Computing Surveys* (arXiv:2003.04919).

**Hardware**
- ams-OSRAM AS7341 datasheet: channel response curves and calibration (also confirms the band values in `environment/spectrum.py`).

---

## 4. Research angles that would make the report stand out

1. **Is adapting the mix worth it at all?** Using `data/sim/sim_dataset.csv`, compare "the best mix for each hour" with "the single best fixed mix all year". A gain under about 2% means a fixed mix is enough, which is a legitimate finding. The plateau result (often 10–30 mixes tied) hints the gain may be small.
2. **What does the spectrum change, and when?** Show how the best mix shifts with cloud, haze, sun height, season and location (e.g. Cameron Highlands vs coastal sites).
3. **Measuring pigment interactions.** Phase 7 mixed cells against the "no interaction" prediction is a clean experiment.
4. **Honest uncertainty.** Report the reliability table and the "errors 2× larger" result from `python -m model.check_uncertainty` like any other result.
5. **Limitations to state clearly:**
   - SPECTRL2 assumes clear skies.
   - The anthocyanin spectrum was traced from an image and measured in solution, not on TiO₂.
   - The AS7341 bands are approximations.
   - Natural dyes degrade in light; published cells lost roughly half their efficiency within 7–30 days.
