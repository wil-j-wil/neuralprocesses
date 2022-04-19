import lab as B
from stheno import EQ, Matern52

from .data import MixtureGenerator, GPGenerator, SawtoothGenerator

__all__ = ["construct_predefined_gens"]


def construct_predefined_gens(
    dtype,
    seed=0,
    batch_size=16,
    num_tasks=2**14,
    dim_x=1,
    dim_y=1,
    pred_logpdf=True,
    pred_logpdf_diag=True,
    device="cpu",
):
    """Construct a number of predefined data generators.

    Args:
        dtype (dtype): Data type to generate.
        seed (int, optional): Seed. Defaults to `0`.
        batch_size (int, optional): Batch size. Defaults to 16.
        num_tasks (int, optional): Number of tasks to generate per epoch. Must be an
            integer multiple of `batch_size`. Defaults to 2^14.
        dim_x (int, optional): Dimensionality of the input space. Defaults to `1`.
        dim_y (int, optional): Dimensionality of the output space. Defaults to `1`.
        pred_logpdf (bool, optional): Also compute the logpdf of the target set given
            the context set under the true GP. Defaults to `True`.
        pred_logpdf_diag (bool, optional): Also compute the logpdf of the target set
            given the context set under the true diagonalised GP. Defaults to `True`.
        device (str, optional): Device on which to generate data. Defaults to `cpu`.

    Returns:
        dict: A dictionary mapping names of data generators to the generators.
    """
    # Ensure that distances don't become bigger as we increase the input dimensionality.
    # We achieve this by blowing up all length scales by `sqrt(dim_x)`.
    factor = B.sqrt(dim_x)
    config = {
        "noise": 0.05,
        "seed": seed,
        "num_tasks": num_tasks,
        "batch_size": batch_size,
        "x_ranges": ((-2, 2),) * dim_x,
        "dim_y": dim_y,
        "device": device,
    }
    kernels = {
        "eq": EQ().stretch(factor * 0.25),
        "matern": Matern52().stretch(factor * 0.25),
        "weakly-periodic": EQ().stretch(factor * 0.5) * EQ().periodic(factor * 0.25),
    }
    gens = {
        name: GPGenerator(
            dtype,
            kernel=kernel,
            num_context_points=(0, 20),
            num_target_points=50,
            pred_logpdf=pred_logpdf,
            pred_logpdf_diag=pred_logpdf_diag,
            **config,
        )
        for name, kernel in kernels.items()
    }
    gens["sawtooth"] = SawtoothGenerator(
        dtype,
        freqs=(factor * 2, factor * 4),
        num_context_points=(0, 40),
        num_target_points=100,
        **config,
    )
    gens["mixture"] = MixtureGenerator(
        *gens.values(),
        seed=config["seed"],
    )
    return gens