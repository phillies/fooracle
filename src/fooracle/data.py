from pandera import DataFrameModel


class HistoricDataSchema(DataFrameModel):
    date: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    tournament: str
    city: str
    country: str
    neutral: bool

    class Config:
        coerce = True
        strict = True
