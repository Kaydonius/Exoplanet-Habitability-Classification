# Exoplanet Habitability Analysis

Exploratory analysis of the NASA Exoplanet Archive (6,022 planets, pulled October 5, 2025) asking which known exoplanets look potentially habitable, and whether a machine learning model can rank them.

## Approach

1. **Data quality.** Insolation flux, the amount of starlight a planet receives and one of the most important habitability variables, is missing for 5,143 of 6,022 planets (85%).
2. **Label.** There is no "habitable" column, so I defined one: rocky-sized radius, a temperate host star, a single-star system, and insolation within a fixed window.
3. **Model.** XGBoost with stratified cross-validation and class-imbalance handling, evaluated with top-k precision, recall, and lift, then used to rank all planets.

## Limitations

The model's scores should not be read as real predictive performance, for three reasons:

- **Too few positives.** The label produces only about ten habitable planets, leaving 6 for training and 2 each for validation and testing. That's too few to validate any model.
- **Target leakage.** The columns that define the label (radius, stellar temperature, star count, insolation) are also model inputs, along with equilibrium temperature, which closely tracks insolation. The classifier mostly recovers the rule I wrote rather than learning anything new.
- **In-sample ranking.** The final ranking is scored on the same data the model was trained on, so the top-ranked planets are largely the ones the rule already labeled positive.

## Planned fix

Most planets missing insolation can have it calculated from physics:

- Stellar luminosity from stellar radius and temperature
- Orbital distance from orbital period and stellar mass (Kepler's third law)
- Insolation = luminosity / distance²

The rework will fill in missing insolation this way, replace the fixed insolation window with habitable-zone boundaries that shift with stellar temperature (Kopparapu et al.), and remove the label-defining columns from the model inputs. The headline result becomes how many additional candidates appear once missing data is derived from orbital mechanics.

## How to run

```
pip install -r requirements.txt
jupyter notebook exoplanet_analysis.ipynb
```

Data: exoplanet_df exported from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) on October 5, 2025. Save it as `exoplanet_data.csv` in the repo root.
