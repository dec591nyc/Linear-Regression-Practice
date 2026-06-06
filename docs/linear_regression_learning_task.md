# Linear Regression Learning Task

## Goal

Teach a non-technical learner how to understand this app and the meaning of linear regression parameters. The learner should be able to explain what the simulator controls do, what the AQI model is trying to predict, and why residuals matter.

## Session 1: What Linear Regression Is

Explain linear regression as a way to draw a line that makes reasonable guesses from numeric inputs.

- `x`: the input value.
- `y`: the actual value we want to explain or predict.
- Prediction: the value the model guesses.
- Residual: actual value minus predicted value.
- Large residual: a record where the model guessed poorly.

Use the simulator first, not the AQI case.

## Session 2: Simulator Parameters

Use the Lab page and change one control at a time.

- Sample size `n`: more rows usually make the line more stable; fewer rows make the result easier to distort.
- Slope `a`: controls how strongly `x` changes `y`.
- Intercept `b`: moves the line up or down.
- Noise variance `var`: adds randomness; higher noise makes the pattern harder to learn.
- Random seed: keeps random data reproducible; changing it creates a different generated sample.
- Model choice: compares OLS, Ridge, Lasso, and ElasticNet.

Expected teaching rule: change only one parameter, observe the line, metrics, and top residuals, then explain the effect in plain language.

## Session 3: Model Metrics

Explain metrics without formulas first.

- R-squared: how much of the target pattern the model can explain.
- RMSE: typical prediction miss, with larger misses punished more.
- MAE: average absolute miss, easier to read than RMSE.
- Top residual observations: the rows where the model is most surprised.

The goal is not to chase a perfect score. The goal is to learn when a simple line is enough and when the data is messier than the model.

## Session 4: AQI Case

Move to the AQI page only after the simulator is clear.

- Target: the air-quality number the model predicts, usually AQI or PM2.5.
- Features: pollutant and weather fields used as clues.
- Region: filters records by available county/city data.
- Model: changes the linear modeling method, not the source data.

Important boundary: Lab parameters such as slope, intercept, variance, and seed do not modify AQI data. They only affect synthetic data in the Lab page.

## Session 5: Final Explanation Practice

Ask the learner to explain:

- What data source is active.
- What target is being predicted.
- Which features were selected.
- What R-squared, RMSE, and MAE suggest.
- Why only the top residual rows are shown.
- Whether the current dataset is enough for the stated business question.

Completion standard: the learner can describe the model result in plain language without relying on formulas.
