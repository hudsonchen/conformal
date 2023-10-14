import pandas as pd
from sklearn.model_selection import train_test_split
import jax
import jax.numpy as jnp
import numpy as np

def load_law_data():
    data = pd.read_csv('./data/law_data.csv', index_col=0)
    data = pd.get_dummies(data,columns=['race'],prefix='',prefix_sep='')

    data['male'] = data['sex'].map(lambda z: 1. if z == 2 else 0.)
    data['female'] = data['sex'].map(lambda z: 1. if z == 1 else 0.)

    data['LSAT'] = data['LSAT'].apply(lambda x: float(jnp.round(x)))
    data['Black'] = data['Black'].apply(lambda x: float(x))

    data = data.drop(axis=1, columns=['region_first', 'first_pf', 'sander_index'])

    data.loc[(data["male"] == 1) & (data["Black"] == 1), "Group 1"] = 1
    data.loc[(data["male"] == 0) & (data["Black"] == 1), "Group 2"] = 1
    data.loc[(data["male"] == 1) & (data["Black"] == 0), "Group 3"] = 1
    data.loc[(data["male"] == 0) & (data["Black"] == 0), "Group 4"] = 1

    data = data.rename(columns={
                    'male': 'S',
                    'Black': 'R',
                    'LSAT': 'L',
                    'UGPA': 'G',
                    'ZFYA': 'F',
                    })
    return train_test_split(data, random_state = 1234, test_size = 0.2) 


def load_synthetic_data():
    data = pd.read_csv('./data/synthetic.csv')
    return train_test_split(data, random_state = 1234, test_size = 0.2)


def load_compas():
    df_compas = pd.read_csv('./data/compas.csv', sep=',', index_col=0)
    df_names = list(df_compas.columns)
    df_names.remove('sex')
    df_names.remove('age')
    df_names.remove('race')
    df_names.remove('two_year_recid')
    df_names.remove('priors_count')
    df_names.remove('juv_fel_count')
    df_names.remove('juv_misd_count')
    df_names.remove('juv_other_count')
    df_names.remove('c_charge_degree')
    df_compas = df_compas.drop(df_names, axis=1)

    df_compas['priors_count'] = np.log(df_compas['priors_count'] + 1)
    df_compas['juv_fel_count'] = (df_compas['juv_fel_count'] > df_compas['juv_fel_count'].values.mean()).astype(float)
    df_compas['juv_misd_count'] = (df_compas['juv_misd_count'] > df_compas['juv_misd_count'].values.mean()).astype(float)
    df_compas['juv_other_count'] = (df_compas['juv_other_count'] > df_compas['juv_other_count'].values.mean()).astype(float)

    df_compas['sex'] = (df_compas['sex'] == 'Male').astype(float)
    df_compas['race'] = (df_compas['race'] == 'Caucasian').astype(float)
    df_compas['c_charge_degree'] = (df_compas['c_charge_degree'] == 'F').astype(float)
    df_compas['two_year_recid'] = 1 - df_compas['two_year_recid']
    df_compas = df_compas.rename(columns={
                'priors_count': 'P',
                'juv_fel_count': 'J1',
                'juv_misd_count': 'J2',
                'juv_other_count': 'J3',
                'sex': 'S',
                'race': 'R',
                'age': 'A',
                'c_charge_degree': 'C',
                'two_year_recid': 'Y',
                })
    return train_test_split(df_compas, random_state = 1234, test_size = 0.2)


def generate_synthetic_data(rng_key, N, sex_ratio, race_ratio):
    key0, key1, key2, key3, key4, key5, key6 = jax.random.split(rng_key, 7)
    
    u = jax.random.uniform(key0, shape=(N,))
    p_nr = u    
    # p_nr = (u + race_ratio) / 2
    # p_nr = race_ratio
    p_ns = sex_ratio
    mu_nk = 0.0
    sigma_nk = 1.0
    
    R = jnp.where(jax.random.uniform(key1, shape=(N,)) < p_nr, 1., 0.)
    S = jnp.where(jax.random.uniform(key2, shape=(N,)) < p_ns, 1., 0.)
    
    K = jax.random.normal(key3, shape=(N,)) * sigma_nk + mu_nk
    # K = 0
    G = jax.random.normal(key4, shape=(N,)) * 0.1 + (K + 4.0 * R + 1.5 * S) * u + u 
    L = jax.random.normal(key5, shape=(N,)) * 0.1 + (K + 6.0 * R + 0.5 * S) * u + u 
    F = jax.random.normal(key6, shape=(N,)) * 0.1 + (K + 3.0 * R + 2.0 * S + jnp.exp(R + 1)) * u + u 

    df = pd.DataFrame({
        'R': R,
        'S': S,
        'K': K,
        'G': G,
        'L': L,
        'F': F
    })
    return df


def ground_truth_synthetic_intervene(rng_key, N, R, S):
    key0, key1, key2, key3, key4, key5, key6 = jax.random.split(rng_key, 7)

    u = jax.random.uniform(key0, shape=(N,))

    mu_nk = 0.0
    sigma_nk = 1.0
    K = jax.random.normal(key3, shape=(N,)) * sigma_nk + mu_nk
    # K = 0
    # PseudoDelta
    G = jax.random.normal(key4, shape=(N,)) * 0.1 + (K + 4.0 * R + 1.5 * S) * u + u 
    L = jax.random.normal(key5, shape=(N,)) * 0.1 + (K + 6.0 * R + 0.5 * S) * u + u 
    F = jax.random.normal(key6, shape=(N,)) * 0.1 + (K + 3.0 * R + 2.0 * S + jnp.exp(R + 1)) * u + u 
    return G, L, F


if __name__ == '__main__':
    rng_key = jax.random.PRNGKey(0)
    train_data, test_data = load_compas()
    pause = True
    # Law specific values
    # latent = 'K'
    # train_data, test_data = load_law_data()
    # unaware_cols = ['LSAT', 'UGPA']
    # full_cols = ['sex', 'Black', 'LSAT', 'UGPA']
    # protect_cols = ['Amerindian','Asian','Black','Hispanic','Mexican','Other','Puertorican','White','male','female']
    # pred = 'ZFYA'
    # X = ['UGPA', 'LSAT']
    # groups = ['Group 1', 'Group 2', 'Group 3', 'Group 4']
    # CausalModel = GroundTruthModelLaw
     