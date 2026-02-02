# ruff: noqa: D103
"""Script for training the model."""

import logging

from utils.constants import RANDOM_SEED, Col
from utils.evaluator import Evaluator
from utils.pede_passos_loader import PedePassosLoader
from utils.pede_passos_pipeline import PedePassosPipeline
from utils.pede_passos_preprocessor import PedePassosPreprocessor
from utils.random_forest_regressor_model import RandomForestRegressorModel


def main() -> None:
    logging.basicConfig(level=logging.DEBUG)

    pipeline = PedePassosPipeline(
        loader=PedePassosLoader(),
        preprocessor=PedePassosPreprocessor(),
        model=RandomForestRegressorModel(),
        evaluator=Evaluator(),
        random_seed=RANDOM_SEED,
    )

    pipeline.run(
        features=[
            Col.age,
            Col.level,
            Col.ian,
            Col.iaa,
            Col.ieg,
            Col.first_year,
        ],
        target=Col.inde,
    )

    pipeline.save_model()


if __name__ == "__main__":
    main()
