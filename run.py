import jax
import jax.numpy as jnp
from load_data import *

from baselines import *


def main():
    dataset = 'synthetic'
    if dataset == 'synthetic':
        race_ratio = 0.5
        sex_ratio = 0.5
        N = 1000
        regenerate = True
        if regenerate:
            rng_key = jax.random.PRNGKey(0)
            df = generate_synthetic_data(rng_key, N, sex_ratio, race_ratio)
            df.to_csv('./data/synthetic.csv', index=False)
        train_data, cal_data = load_synthetic_data()
    elif dataset == 'compas':
        train_data, cal_data = load_compas()

    rng_key = jax.random.PRNGKey(0)
    X_name = ['G', 'L']
    Y_name = ['F']
    A = ['R']

    alpha = 0.1 # 1-alpha is the desired coverage 
    print(f"alpha = {alpha}")

    model = LinearRegression()
    intervention_set = [0.0, 1.0]
    print('-' * 50)
    print(f'Method | {"Intervention":<5} | {"Width":<5} | {"Coverage":<5}')

    interval_width, coverage = marginal_conformal(rng_key, model, train_data, cal_data, X_name, Y_name, 
                                                  N, alpha, intervention_set, ground_truth_synthetic_intervene)
    for intervention, width, cover in zip(intervention_set, interval_width, coverage):
        print(f'Marginal Conformal | {intervention} | {width} | {cover}')

    interval_width, coverage = conditional_conformal(rng_key, model, train_data, cal_data, X_name, Y_name, 
                                                N, alpha, intervention_set, ground_truth_synthetic_intervene)
    for intervention, width, cover in zip(intervention_set, interval_width, coverage):
        print(f'Conditional Conformal | {intervention} | {width} | {cover}')

    pause = True
    return


if __name__ == '__main__':
    main()