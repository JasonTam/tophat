import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import argparse
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials
from tophat.utils.config_parser import Config
from jobs.fit_job import FitJob
from time import time
from functools import partial


# We don't care about every param, just the ones being optimized
# TODO: could probably be defined via config
def params_to_s(params_d) -> pd.Series:
    s = pd.Series()

    s['l1_emb'] = params_d['l1_emb']
    s['l2_emb'] = params_d['l2_emb']
    s['loss_fn'] = params_d['loss_fn']
    s['sample_method'] = params_d['sample_method']

    for feat in [
        'age_bucket',
        'gender',
        'international_vs_domestic_location',
        'loyalty_current_level',

        'ops_brand_id',
        'taxonomy_key_lv1',
        'taxonomy_key_lv2',
        'taxonomy_key_lv3',
        'taxonomy_key_lv4',
        'median_sale_price_bin_id',
    ]:
        if feat in params_d['feature_weights_d'].keys():
            s[f'w-{feat}'] = params_d['feature_weights_d'][feat]
    return s


def composite_loss(scores_df):
    # Note: Semi-arbitrary hand-crafted composite loss for now
    # (going to scale up mapk)
    return -(50 * scores_df['mapk'] + scores_df['auc']) / 2.


def objective(params, path_log=None):
    job = FitJob(fit_config=params)
    job.model_init()
    tic = time()
    val_d = job.model_fit()
    dur_fit = time() - tic

    job.sess.close()
    tf.reset_default_graph()  # this sometimes screws up the next graph idk why

    scores_df = pd.DataFrame(val_d)
    # Note: can also return and make use of score variance (over users)

    scores_df['loss'] = composite_loss(scores_df)
    best_round = scores_df['loss'].idxmin()
    scores_best: pd.Series = scores_df.loc[best_round]

    ret_d = scores_best.to_dict()
    if np.isnan(scores_best['loss']) or scores_best['loss'] == 0:
        ret_d['status'] = STATUS_FAIL
    else:
        ret_d['status'] = STATUS_OK

    ret_d['best_round'] = best_round
    ret_d['best_round_dur'] = dur_fit * (best_round + 1) / len(scores_df)

    # Logging each result
    log_s: pd.Series = pd.concat([
        pd.Series(ret_d),
        pd.Series(params_to_s(params)),
    ])
    if path_log:
        with open(path_log, 'a+') as f:
            if not f.tell():
                # Write header if first line
                f.write(','.join(log_s.index) + '\n')
            log_s.to_frame().T.to_csv(f, header=False, index=False)

    return ret_d

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Example of hyper-parameter optimization')
    parser.add_argument('--path_log', help='Where to log params and scores',
                        default='/tmp/hpo_tophat.csv', nargs='?',)
    parser.add_argument('--path_trials',
                        help='Where to save pickled trials object',
                        default='/tmp/hpo_tophat.p', nargs='?',)
    parser.add_argument('--max_evals', help='Number of iterations',
                        default=1000, nargs='?', type=int,)

    args = parser.parse_args()

    config = Config(f'config/fit_config_hyperopt_local.py')

    obj_fn = partial(objective, path_log=args.path_log)

    trials = Trials()
    best = fmin(fn=obj_fn,
                space=config.__dict__['_params'],
                algo=tpe.suggest,
                max_evals=args.max_evals,
                trials=trials,
                )

    if args.path_trials:
        pickle.dump(trials, open(args.path_trials, 'wb'))
