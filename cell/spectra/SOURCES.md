# Absorption spectra (placeholders until Phase 7 measurements)

All files: tab-separated, one header line, column 1 = wavelength (nm), column 2 = absorption.
The simulator scales each one so its highest point in 400–800 nm equals 1 ("same strength").

| File | Pigment | Solvent | Range | Units | Source |
|---|---|---|---|---|---|
| `chlorophyll_a.txt` | Chlorophyll a | methanol | 364–720 nm, ~2.5 nm steps | normalized (peak 666 nm = 1) | PhotochemCAD NCHL008, "Chl a, MeOH (Kobayashi, 2013)" |
| `curcumin.txt` | Curcumin | ethanol | 200–600 nm, 1 nm | ε, M⁻¹ cm⁻¹ (55 000 at 425 nm) | PhotochemCAD D08 |
| `anthocyanin_c3g.txt` | Cyanidin 3-O-glucoside | HPLC mobile phase (acidic) | 250–600 nm, 1 nm | µAU (detector units) | Digitized from Mendeley Data image, see below |

## Citations

- Taniguchi, M.; Lindsey, J. S. Absorption and fluorescence spectral database of chlorophylls and analogues. *Photochem. Photobiol.* 2021, 97, 136–165. doi:10.1111/php.13319 — data: https://www.photochemcad.com/databases/natural-chlorophylls/chl-a-meoh-kobayashi-2013
- Taniguchi, M.; Lindsey, J. S. Database of absorption and fluorescence spectra of >300 common compounds for use in PhotochemCAD. *Photochem. Photobiol.* 2018, 94, 290–327. doi:10.1111/php.12860 — data: https://www.photochemcad.com/databases/common-compounds/polyenes-polyynes/curcumin
- Custodio-Mendoza, J.; Kurek, M. UV–Vis and tandem mass spectrometry data supporting anthocyanin identification in purple corn extracts. Mendeley Data, V1, 2026. doi:10.17632/bvh8w92yv2.1 (CC BY 4.0). Related article: Custodio-Mendoza et al., *Anal. Bioanal. Chem.* 2025, doi:10.1007/s00216-025-05951-8.

## How the anthocyanin file was made

No anthocyanin is in PhotochemCAD, and no published spectrum was found as a data table.
`anthocyanin_c3g.txt` was traced from the dataset's `Cyanidin 3-O-glucoside.png` (kept here as
`anthocyanin_c3g_source.png`, 327×330 px): axes calibrated from tick marks (200–600 nm, 0–34 000 µAU),
the black curve's midline taken per pixel column, then interpolated to 1 nm.

- Resolution ≈ 1.5 nm per pixel; traced visible peak 526 nm vs 522 nm labelled on the image.
- Checked against labelled points: 278 nm ≈ 16 000, 349 nm ≈ 2 100, 453 nm ≈ 2 600 µAU.
- An in-line HPLC detector spectrum in acidic solvent, not a dye film on TiO₂ — on TiO₂ the band
  usually broadens and shifts red. Replace with your measured extract spectrum in Phase 7.
