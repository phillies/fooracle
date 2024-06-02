from dataclasses import dataclass, field
import numpy as np
from pandas import DataFrame
from torch import Tensor, nn

from fooracle.models.base import Fooracle


class MLP(nn.Module):
    """
    Multilayer Perceptron for simple regression.
    """

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(633, 1024),
            nn.RReLU(),
            nn.Linear(1024, 128),
            nn.RReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x: Tensor, game_noise=None) -> Tensor:
        """
        Forward pass
        """
        x = self.layers(x)

        return x


@dataclass
class MLPFooracle(Fooracle):
    """Neural network based fooracle
    based on historical results it will tell you what the future brings.
    """

    teams_encoder: dict[str, np.ndarray] = field(default_factory=dict)
    mlp: MLP = MLP()
    fairy_dust: float = 0.0

    def foretell(
        self, team1: str, team2: str, draw_allowed: bool = True, host: str | None = None
    ) -> tuple[int, int]:
        if host is None:
            host = ""
        if team1 == host:
            neutral = 0
        elif team2 == host:
            neutral = 0
            team2 = team1
            team1 = host
        else:
            neutral = 1
        game = np.hstack(
            [
                self.teams_encoder[team1].reshape(1, -1),
                self.teams_encoder[team2].reshape(1, -1),
                np.array([neutral]).reshape(1, 1),
            ]
        )
        res = self.mlp(Tensor(game)).detach().numpy().flatten()
        res = np.maximum(
            (1 - self.fairy_dust) * res + self.fairy_dust * np.random.randn(2) * 3.0, 0
        )
        return res[0], res[1]

    def train(self, game_data: DataFrame) -> None:
        pass

    def store(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass

    def knows_team(self, team: str) -> bool:
        return team in self.teams_encoder
