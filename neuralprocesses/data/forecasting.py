import lab as B
import numpy as np
from plum import convert
import pickle
import pandas as pd
from abc import ABC, abstractmethod
from torch import tensor

from ..dist import AbstractDistribution
from ..dist.uniform import UniformDiscrete
from .data import DataGenerator
from ..mask import Masked


class ForecasterDataGenerator(DataGenerator, ABC):
    data_length: int = 10000
    # data_offset: int = 0
    # data_offset: int = 250
    data_offset: int = 215
    """The data generator for forecasting tasks. """

    def __init__(
        self,
        dtype,
        seed=0,
        num_tasks=2**10,
        batch_size=16,
        horizon=48,
        history_max=60,
        sample_horizon=False,
        eval_mode=False,
        include_context_in_target=True,
        subsample_history=1,
        device="cpu",
    ):
        super().__init__(dtype, seed, num_tasks, batch_size=batch_size, device=device)

        self.history_max = history_max
        self.horizon = horizon
        self.sample_horizon = sample_horizon
        self.subsample_history = subsample_history

        # Load the data.
        df = self.data_loader()
        if eval_mode:
            df = df[self.data_offset+self.data_length:self.data_offset+2*self.data_length]  # validation set
            num_context = UniformDiscrete(history_max, history_max)
            num_target = UniformDiscrete(horizon, horizon)
        else:
            df = df[self.data_offset:self.data_offset+self.data_length]  # train set
            num_context = UniformDiscrete(0, history_max)
            num_target = UniformDiscrete(1, horizon)
        self.num_context = convert(num_context, AbstractDistribution)
        self.num_target = convert(num_target, AbstractDistribution)
        self.forecast_start = UniformDiscrete(0, self.data_length - self.horizon - 1)
        self.x, self.x_feat, self.y = self.process_data(df)
        self.include_context_in_target = include_context_in_target
        self.eval_mode = eval_mode
        self.timestamps = self.get_timestamps(df)

    @staticmethod
    @abstractmethod
    def data_loader(): ...

    @abstractmethod
    def process_data(self, df): ...

    @staticmethod
    @abstractmethod
    def get_timestamps(df): ...

    @staticmethod
    def fwd_idx(start, horizon):
        return np.arange(start + 1, start + horizon + 1, 1)

    @staticmethod
    def bwd_idx(start, history):
        return np.arange(start - history + 1, start + 1, 1)

    def generate_batch(self, for_plot=False):
        with B.on_device(self.device):
            self.state, starts = self.forecast_start.sample(self.state, self.dtype, self.batch_size)
            if for_plot:
                # horrible hack because I just want to plot the first dataset
                # starts = B.sort(starts - B.min(starts) + self.history_max)
                starts = B.minimum(
                    B.sort(starts - B.min(starts) + self.history_max),
                    (self.data_length - self.horizon - 1) * B.ones(int, self.batch_size),
                )
            self.state, num_context = self.num_context.sample(self.state, self.dtype, 1)
            if self.sample_horizon:
                self.state, num_target = self.num_target.sample(self.state, self.dtype, 1)
            else:
                num_target = self.horizon
            xc = B.concat(*[
                B.take(self.x, B.concat(self.bwd_idx(start, num_context), self.fwd_idx(start, num_target)), axis=-1) - start
                for start in starts
            ])
            xc_feat = B.concat(*[
                B.take(self.x_feat, B.concat(self.bwd_idx(start, num_context), self.fwd_idx(start, num_target)), axis=-1)
                for start in starts
            ])
            yc = B.concat(*[
                B.take(self.y, self.bwd_idx(start, num_context), axis=-1)
                for start in starts
            ])
            yt_ = B.concat(*[B.take(self.y, self.fwd_idx(start, num_target), axis=-1) for start in starts])
            xt_ = B.concat(*[B.take(self.x, self.fwd_idx(start, num_target), axis=-1) - start for start in starts])
            # if self.eval_mode:
            yt = yt_
            xt = xt_
            # else:
            #     MAGIC_NUM = 1
            #     xc_back = B.concat(*[
            #         B.take(self.x, self.bwd_idx(start, num_context)[-MAGIC_NUM:], axis=-1) - start
            #         for start in starts
            #     ])
            #     xt = B.concat(xc_back, xt_, axis=-1)
            #     # xt = B.concat(xc_back[:, :, ::self.subsample_history], xt_, axis=-1)
            #     yt = B.concat(yc[:, :, -MAGIC_NUM:], yt_, axis=-1)
            #     # yt = B.concat(yc[:, :, ::self.subsample_history], yt_, axis=-1)
            y_all = B.concat(yc, yt_ * B.nan, axis=-1)
            y_all_cat = B.concat(xc_feat, y_all, axis=1)
            available = ~B.isnan(y_all_cat)
            yc_cat = Masked(
                B.where(available, y_all_cat, B.zeros(y_all_cat)),
                B.cast(self.dtype, available),
            )
            batch = {
                "contexts": [(xc, yc_cat)],
                "xt": xt,
                "yt": yt,
                "timestamps": B.concat(*[
                    B.take(self.timestamps, B.concat(self.bwd_idx(start, num_context), self.fwd_idx(start, num_target)), axis=0)[None]
                    for start in starts
                ]),
            }
            return batch


class MULASolarGenerator(ForecasterDataGenerator):
    """The MULA solar generation data generator. """

    @staticmethod
    def data_loader():
        """MULA solar generation data.

        Source:


        Returns:
            :class:`pd.DataFrame`: Predator–prey data.
        """
        with open('/Users/williamwilkinson/data/mula_solar_generation.pickle', 'rb') as handle:
            df: pd.DataFrame = pickle.load(handle)
        return df

    def process_data(self, df):
        with B.on_device(self.device):
            # Convert the data frame to the framework tensor type.
            x = B.cast(self.dtype, np.array(df.reset_index().index))
            # features = df[["hour", "temp", "dew_point", "humidity", "wind_speed", "clouds_all"]]
            features = df[["hour"]]
            features /= features.max()
            x_feat = B.cast(
                self.dtype,
                np.array(features),
            )
            y = B.cast(self.dtype, np.array(df["generation"] / df["generation"].max() * 2))
            # Move them onto the GPU and make the shapes right.
            x = B.to_active_device(x)[None, None, :]
            x_feat = B.transpose(B.to_active_device(x_feat)[None, :, :])
            y = B.to_active_device(y)[None, None, :]
            return x, x_feat, y

    @staticmethod
    def get_timestamps(df):
        return df.index.values


class WhiteleeWindGenerator(ForecasterDataGenerator):
    """The Whitelee wind generation data generator. """

    @staticmethod
    def data_loader():
        """MULA solar generation data.

        Source:


        Returns:
            :class:`pd.DataFrame`: Predator–prey data.
        """
        with open('/Users/williamwilkinson/data/whitelee_wind_speed_actual.pickle', 'rb') as handle:
            df: pd.DataFrame = pickle.load(handle)
        return df

    def process_data(self, df):
        with B.on_device(self.device):
            # Convert the data frame to the framework tensor type.
            x = B.cast(self.dtype, np.array(df.reset_index().index))
            get_time_of_day = lambda time: time.hour + (time.minute / 60)
            df["time_of_day"] = df["update_time"].apply(get_time_of_day)
            features = df[["wind_speed", "time_of_day"]]
            features /= features.max()
            x_feat = B.cast(
                self.dtype,
                np.array(features),
            )
            y = B.cast(self.dtype, np.array(df["generation"] / df["generation"].max() * 2))
            # Move them onto the GPU and make the shapes right.
            x = B.to_active_device(x)[None, None, :]
            x_feat = B.transpose(B.to_active_device(x_feat)[None, :, :])
            y = B.to_active_device(y)[None, None, :]
            return x, x_feat, y

    @staticmethod
    def get_timestamps(df):
        return df["update_time"].values
