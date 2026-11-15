# Data

This project uses the public Kaggle data **Store Item Demand Forecasting Challenge**:
https://www.kaggle.com/c/demand-forecasting-kernels-only

The file `train.csv` is **included in this repository** for convenience. Kaggle's terms for this data apply.

## The file
`train.csv` from the competition's "Data" tab. It has four columns (`date`, `store`, `item`, `sales`) and 913,000 rows (10 stores x 50 items x 1,826 days, 2013-01-01 to 2017-12-31). The other Kaggle files are not used and not included.

## Where it is
In the **repo root**, next to `requirements.txt` and `src/` (not in this `data/` folder):

```
store-demand-forecast/
  train.csv                       <- here
  src/
  results/
  store_demand_forecast.ipynb
  requirements.txt
  data/README.md                  <- this file
```

Do not edit the file. The run guard checks its SHA-256 fingerprint. Expected value:
`038F25690A65149C94F86DDD3DECEDA20C037A5CFD754CAFDFC539A72992F2ED`

How to run the notebook is in the main `README.md`. A run rewrites `results/` and adds entries to `results/final_test_log.json`. Please do not edit or delete that log.
