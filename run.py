import jax
import jax.numpy as jnp

from data.datasets import *
from models.methods import *

def main():
    n_observation = 1000
    n_intervention = 100
    d = 10

    synthetic_setups = dict({"A": 1, "B": 0})
    setup = 'A'
    alpha = 0.1

    # df_train, df_test = generate_lilei_hua_data()
    # _ = weighted_conformal_prediction([df_train, df_test], 
    #                                   metalearner="DR", 
    #                                   quantile_regression=True, 
    #                                   alpha=0.1, 
    #                                   test_frac=0.1)
    # df_o = [df_train, df_test]
    rng_key = jax.random.PRNGKey(42)
    df_o, df_i = generate_data(rng_key=rng_key,
                               n_observation=n_observation,    
                                n_intervention=n_intervention,
                                d=d, 
                                gamma=synthetic_setups[setup], 
                                alpha=alpha) 
    _ = transductive_weighted_conformal(df_o,
                                        df_i,
                                        quantile_regression=True,
                                        alpha=0.1,
                                        test_frac=0.1,
                                        method="counterfactual")
    
    # _ = weighted_conformal_prediction(df_o, 
    #                                   quantile_regression=True, 
    #                                   alpha=0.1, 
    #                                   test_frac=0.1,
    #                                   method="counterfactual")
    # conditional_coverage, average_interval_width, PEHE, conformity_scores = conformal_metalearner(df_o, 
    #                                                                                               metalearner="DR", 
    #                                                                                               quantile_regression=True, 
    #                                                                                               alpha=0.1, 
    #                                                                                               test_frac=0.1)
    print(conditional_coverage)
    print(average_interval_width)
    print(PEHE)
    pause = True
    return


if __name__ == '__main__':
    main()