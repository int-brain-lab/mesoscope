from meso_concat import plot_session_even_odd_similarity_vs_performance_subject


def main():
    df, fig, paths = plot_session_even_odd_similarity_vs_performance_subject(
        subject='SP058',
        filter_neurons=True,
        min_trials=10,
        rerun_canonical=False,
        rerun_trial_cuts=False,
        rerun_stacks=False,
        rerun_group=False,
        nclus_rm=20,
        shuf=True,
        shuf_seed=0,
        save_outputs=True,
        fail_fast=False,
        verbose=True,
    )

    print('\nRESULT_PATHS_START')
    for k, v in paths.items():
        if k != 'failures':
            print(f"{k}: {v}")
    print('RESULT_PATHS_END')

    print('\nSUMMARY_DF_START')
    cols = [
        'session_number', 'session_date', 'eid', 'performance', 'cv_sim_mean',
        'cv_sim_shuf_mean', 'cv_sim_delta', 'n_neurons', 'n_trials_scored'
    ]
    print(df[cols].to_string(index=False))
    print('SUMMARY_DF_END')


if __name__ == '__main__':
    main()
