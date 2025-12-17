"""File with all constant values."""

from pathlib import Path

DATA_FODLER_PATH = f"{Path(__file__).parent.parent.parent.resolve()}\\data"

RANDOM_SEED = 1


class Col:
    """`pede_passos.csv` dataset columns."""

    first_year = "INGRESSANTE"
    name = "NOME"
    year = "ANO"
    level = "FASE"
    age = "IDADE"
    inde = "INDE"
    ian = "IAN"
    iaa = "IAA"
    ieg = "IEG"
    ida = "IDA"
    ips = "IPS"
    ipp = "IPP"
    ipv = "IPV"
