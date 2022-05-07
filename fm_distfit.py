from forecast_models import ForecastModelABC
from distfit import distfit
from tqdm import tqdm
import numpy as np


class DistfitModel(ForecastModelABC):
    def fit(self, forecast_md):
        model = {}
        for qt_enc, qtmd in tqdm(forecast_md.qtmds.items(),
                                 total=len(forecast_md.qtmds),
                                 desc="Fitting query templates."):
            qt = forecast_md.qt_enc.inverse_transform(qt_enc)
            print(f"Fitting query template {qt_enc}: {qt}")
            model[qt] = {}
            params = qtmd.get_historical_params()

            if len(params) == 0:
                # No parameters.
                continue

            for idx, col in enumerate(params.columns, 1):
                model[qt][idx] = {}
                if str(params[col].dtype) == "string":
                    model[qt][idx]["type"] = "sample"
                    model[qt][idx]["sample"] = params[col]
                    print(f"Query template {qt_enc} parameter {idx} is a string. "
                          "Storing values to be sampled.")
                else:
                    assert not str(params[col].dtype) == "object", "Bad dtype?"
                    dist = distfit()
                    dist.fit_transform(params[col], verbose=0)
                    print(f"Query template {qt_enc} parameter {idx} "
                          f"fitted to distribution: {dist.model['distr'].name} {dist.model['params']}")
                    model[qt][idx]["type"] = "distfit"
                    model[qt][idx]["distfit"] = dist
        self.model = model
        return self

    def generate_parameters(self, query_template, timestamp):
        # The timestamp is unused because we are just fitting a distribution.
        # Generate the parameters.
        params = {}
        for param_idx in self.model[query_template]:
            fit_obj = self.model[query_template][param_idx]
            fit_type = fit_obj["type"]

            param_val = None
            if fit_type == "distfit":
                dist = fit_obj["distfit"]
                param_val = str(dist.generate(n=1, verbose=0)[0])
            else:
                assert fit_type == "sample"
                param_val = np.random.choice(fit_obj["sample"])
            assert param_val is not None

            # Param dict values must be quoted for consistency.
            params[f"${param_idx}"] = f"'{param_val}'"
        return params
