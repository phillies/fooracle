# fooracle
football (soccer) oracle for prediction games based on historical match data.
See Jupyter file for a demo on how to use it.

Fooracle supports 2 modes, the 'dist' mode where a half normal distribution is fittet to the number of goals a team scored in the past and for each game, a random value is drawn from that distribution to predict the number of goals the team will score, and the 'mlp' mode, where a neural network was trained to predict the outcome of a game.

As the neural netowrk is deterministic and each time e.g. Germany faces Portugal, Germany will win 2:1, you can add some `fairy_dust` to have some random noise on the outcome and make things more surprising.

Data: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017