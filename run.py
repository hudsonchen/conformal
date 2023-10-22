import jax
import jax.numpy as jnp

from data.datasets import *
from models.metalearners import *

def main():
    n_observation = 1000
    n_intervention = 100
    d = 10

    synthetic_setups = dict({"A": 1, "B": 0})
    setup = 'A'
    alpha = 0.1
    df_o_T_0, df_o_T_1, df_i_T_0, df_i_T_1 = generate_data(n_observation=n_observation,
                                                            n_intervention=n_intervention,
                                                            d=d, 
                                                            gamma=synthetic_setups[setup], 
                                                            alpha=alpha) 
    df_o = pd.concat([df_o_T_0, df_o_T_1])
    df_i = pd.concat([df_i_T_0, df_i_T_1])

    oracle_width  = df_o["width"].loc[0]
    _ = weighted_conformal_prediction(df_o, 
                                      metalearner="DR", 
                                      quantile_regression=True, 
                                      alpha=0.1, 
                                      test_frac=0.1)
    conditional_coverage, average_interval_width, PEHE, conformity_scores = conformal_metalearner(df_o, 
                                                                                                  metalearner="DR", 
                                                                                                  quantile_regression=True, 
                                                                                                  alpha=0.1, 
                                                                                                  test_frac=0.1)
    print(conditional_coverage)
    print(average_interval_width)
    print(PEHE)
    pause = True
    return


if __name__ == '__main__':
    main()