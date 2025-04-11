import pandas as pd
import numpy as np
import pymc as pm
import arviz as az

class BayesianVARForecaster:
    def __init__(self, data, var_names, location, gtdate, window_weeks=12, lags=2, forecast_steps=4, seed="PyMC LABS - BVAR"):
        self.data = data
        self.var_names = var_names
        self.loc = location
        self.gtdate = pd.to_datetime(gtdate)
        self.ref_date = self.gtdate + pd.Timedelta(weeks=1)
        self.window_weeks = window_weeks
        self.stdate = self.gtdate - pd.Timedelta(weeks=window_weeks)
        self.lags = lags
        self.forecast_steps = forecast_steps
        self.fctdates = [self.gtdate + pd.Timedelta(weeks=i) for i in range(1, forecast_steps + 1)]
        self.seed = sum(map(ord, seed)) if isinstance(seed, str) else seed
        self.rng = np.random.default_rng(self.seed)
        self.trace = None
        self.idata = None
        self.model = None
        self.forecast_draws = None
        self.out_df = None

        self._prepare_data()

    def _prepare_data(self):
        rdf = self.data[self.data.geo_value == self.loc]
        rdf = rdf.set_index("time_value")
        rdf = rdf[list(self.var_names)]
        rdf = rdf.dropna()
        self.alldata = rdf
        self.data_window = self.alldata.loc[self.stdate:self.gtdate]

    def build_model(self):
        coords = {
            "lags": np.arange(self.lags) + 1,
            "vars": self.var_names,
            "cross_vars": self.var_names,
            "time": range(len(self.data_window) - self.lags),
        }

        with pm.Model(coords=coords) as model:
            intercept = pm.Normal("intercept", mu=0, sigma=1, dims=("vars",))
            lag_coefs = pm.Normal("lag_coefs", mu=0, sigma=1, dims=("lags", "vars", "cross_vars"))
            noise = pm.HalfNormal("noise", dims=("vars",))

            ar_terms = []
            for v in range(len(self.var_names)):
                ar = pm.math.sum([
                    pm.math.sum(lag_coefs[i, v] * self.data_window.values[self.lags - (i + 1):-(i + 1)], axis=-1)
                    for i in range(self.lags)
                ], axis=0)
                ar_terms.append(ar)

            mean = intercept + pm.math.stack(ar_terms, axis=-1)
            obs = pm.Normal("obs", mu=mean, sigma=noise, observed=self.data_window[self.lags:], dims=("time", "vars"))

            self.model = model

    def fit(self):
        with self.model:
            self.trace = pm.sample(chains=4, random_seed=self.rng, nuts={"target_accept": 0.98})

    def _forecast_single(self, intercept, lag_coefs, noise):
        len_data = len(self.data_window)
        new_draws = np.zeros((len_data + self.forecast_steps, len(self.var_names)))
        new_draws[:len_data] = self.data_window[:]

        for i in range(self.forecast_steps):
            ar = []
            for v in range(len(self.var_names)):
                ar_v = np.sum(lag_coefs[:, v, :] * new_draws[len_data + i - self.lags:len_data + i], axis=(0, 1))
                ar.append(ar_v)
            mean = intercept + np.array(ar)
            new_draws[len_data + i] = self.rng.normal(mean, noise)

        new_draws[:len_data] = np.nan
        return new_draws

    def generate_forecast(self, var_index=0, draws=100):
        post = self.trace.posterior.stack(sample=("chain", "draw"))

        intercept_draws = post["intercept"].values  # shape: (vars, sample)
        lag_coefs_draws = post["lag_coefs"].values  # shape: (lags, vars, cross_vars, sample)
        noise_draws = post["noise"].values  # shape: (vars, sample)

        total_samples = intercept_draws.shape[-1]
        selected_draws = self.rng.integers(total_samples, size=draws)

        forecasted = []
        for i in selected_draws:
            intercept = intercept_draws[:, i]  # shape: (vars,)
            lag_coefs = lag_coefs_draws[:, :, :, i]  # shape: (lags, vars, cross_vars)
            noise = noise_draws[:, i]  # shape: (vars,)
            fcast = self._forecast_single(intercept, lag_coefs, noise)
            forecasted.append(fcast)

        forecasted = np.stack(forecasted, axis=-1)
        self.forecast_draws = forecasted[-self.forecast_steps:, var_index, :]

    def _gen_quantiles(self, samples):
        quantiles = np.append(np.append([0.01, 0.025], np.arange(0.05, 0.95 + 0.05, 0.05)), [0.975, 0.99])
        samples[samples < 0] = 0
        q_vals = np.quantile(samples, q=quantiles)
        return pd.DataFrame.from_dict(data=dict(zip(quantiles, q_vals)), orient='index').T

    def format_forecast(self, target_name="wk inc VAR"):
        fdf = pd.DataFrame(index=self.fctdates, columns=np.arange(self.forecast_draws.shape[-1]))
        fdf.index.name = "target_end_date"
        fdf.loc[:, :] = self.forecast_draws

        fctdf = pd.DataFrame()
        for d in fdf.index:
            temp = self._gen_quantiles(fdf.loc[[d], :])
            hr = (d - self.gtdate).days // 7
            temp["horizon"] = hr - 1
            temp["target"] = target_name
            fctdf = pd.concat([fctdf, temp])

        fctdf = fctdf.reset_index()
        fctdf = fctdf.rename(columns={"index": "target_end_date"})
        fctdf["location"] = self.loc
        fctdf["geo_res"] = "region"
        fctdf["method"] = "VAR-dn"
        fctdf["avl_date"] = self.gtdate
        fctdf["reference_date"] = self.ref_date
        fctdf["output_type"] = "quantile"

        id_vars = ["location", "target_end_date", "avl_date", "horizon", "reference_date", "method", "target", "output_type", "geo_res"]
        self.out_df = fctdf.melt(id_vars=id_vars, var_name="output_type_id")

    def run_inference_summary(self):
        with self.model:
            self.idata = pm.sample_prior_predictive()
            self.idata.extend(pm.sample(draws=2000, random_seed=130))
            pm.sample_posterior_predictive(self.idata, extend_inferencedata=True, random_seed=self.rng)

        summary = az.summary(self.idata)
        summary["location"] = self.loc
        summary["date"] = self.ref_date.strftime('%Y-%m-%d')
        return summary
