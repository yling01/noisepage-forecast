from abc import abstractmethod, ABC

class ForecastModelABC(ABC):
    @abstractmethod
    def fit(self, forecast_md):
        """
        Fit the forecast model to the forecast metadata as of that point in time.

        Parameters
        ----------
        forecast_md : ForecastMD

        Returns
        -------

        """
        pass

    @abstractmethod
    def generate_parameters(self, query_template, timestamp):
        """
        Generate a set of parameters for the specified query template as of the specified time.

        Parameters
        ----------
        query_template : str
        timestamp : str

        Returns
        -------
        params : Dict[str, str]
            The generated parameters, a dict mapping from "$n" keys to quoted parameter values.
        """
        pass
