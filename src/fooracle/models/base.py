from abc import ABC, abstractmethod
from dataclasses import dataclass

from pandas import DataFrame

from fooracle.logger import LOGGER


class Fooracle(ABC):
    """Football oracle
    based on historical results it will tell you what the future brings.
    Uses data from: https://www.kaggle.com/martj42/international-football-results-from-1872-to-2017/data
    """

    @abstractmethod
    def foretell(
        self, team1: str, team2: str, draw_allowed: bool = True, host: str | None = None
    ) -> tuple[int, int]: ...

    @abstractmethod
    def train(self, game_data: DataFrame) -> None: ...

    @abstractmethod
    def store(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...

    def talk(self, message: str) -> None:
        LOGGER.info(message)

    @abstractmethod
    def knows_team(self, team: str) -> bool: ...


@dataclass
class ConstantFooracle(Fooracle):
    """Constant based fooracle
    based on historical results it will tell you what the future brings.
    """

    home_score: int = 2
    away_score: int = 1

    def foretell(
        self, team1: str, team2: str, draw_allowed: bool = True, host: str | None = None
    ) -> tuple[int, int]:
        return self.home_score, self.away_score

    def train(self, game_data: DataFrame) -> None:
        pass

    def store(self, path: str) -> None:
        try:
            with open(path, "w") as file:
                file.write(f"{self.home_score},{self.away_score}")
        except Exception as e:
            LOGGER.error(f"Unable to write to {path}: {e}")

    def load(self, path: str) -> None:
        try:
            with open(path, "r") as file:
                self.home_score, self.away_score = map(int, file.read().split(","))
        except Exception as e:
            LOGGER.error(f"Unable to load from {path}: {e}")

    def knows_team(self, team: str) -> bool:
        return True
