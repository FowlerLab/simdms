#!/usr/bin/env python
#
#  Copyright 2016-2017 Alan F Rubin and Hannah Gelman
#
#  This file is part of simdms.
#
#  simdms is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  simdms is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with simdms.  If not, see <http://www.gnu.org/licenses/>.


from __future__ import print_function
import os
import json
import argparse
import numpy as np
import scipy.stats
import pandas as pd


def _create_counts_multiindex(conditions, rounds, replicates):
    """
    Create and return a pandas MultiIndex for a counts DataFrame. The levels
    are replicates (in the format 'repN') and rounds (in the format 'c_M').

    Parameters
    ----------
    conditions : list of str
        List of labels for the conditions in this simulation
    rounds : int
        Number of rounds of selection (including the input)
    replicates : int
        Number of replicate selections

    Returns
    -------
    pd.MultiIndex
    """
    replicate_labels = ['rep{}'.format(x + 1) for x in range(replicates)]
    round_labels = ['c_{}'.format(x) for x in range(rounds)]
    column_index = pd.MultiIndex.from_product([conditions, replicate_labels,
                                               round_labels])
    return column_index


def _create_variant_index(variants):
    """
    Create and return a pandas Index of variant row names. The format for
    variant names is 'varN', zero-padded such that the longest value has no
    leading zeroes. The first variant in the index is the wild type variant
    '_wt'.

    Parameters
    ----------
    variants : int
        Number of variants (including wild type)

    Returns
    -------
    pd.Index
    """
    width = np.ceil(np.log10(variants)).astype(np.int32)
    variant_names = ['var{}'.format('{}'.format(x + 1).zfill(width)) for x in
                     range(variants - 1)]
    variant_index = pd.Index(['_wt'] + variant_names)
    return variant_index


def _create_effects_multiindex(conditions, effect_name, replicates):
    """
    Create and return a pandas MultiIndex for an effects DataFrame. The format
    for replicates is 'repN'. The first element in the index is the effect
    name.

    Parameters
    ----------
    conditions : list of str
        List of labels for the conditions in this simulation
    effect_name : name
        Name of the effect, either 'pselect' or 'growth_rate'
    replicates : int
        Number of replicate selections

    Returns
    -------
    pd.Index
    """
    replicate_labels = ['rep{}'.format(x + 1) for x in range(replicates)]
    effects_index = pd.MultiIndex.from_product([conditions, [effect_name] +
                                                replicate_labels])
    return effects_index


def starting_count_distribution(pop_size, variants, wt_freq, sigma):
    """
    Create and return a dictionary containing a scipy.stats.lognorm
    distribution object and its associated parameters, used to generate
    starting counts for variants in a population with given size and wild type
    frequency.

    Parameters
    ----------
    pop_size : int
        Total number of cells in the population
    variants : int
        Number of unique variants (including wild type)
    wt_freq : float
        Frequency of the wild type in the starting population
    sigma : float
        Standard deviation of the underlying normal distribution

    Returns
    -------
    dict
    """
    mean_count = pop_size * (1 - wt_freq) / (variants - 1)
    distn = scipy.stats.lognorm(s=sigma, scale=mean_count * \
                                               (1 - sigma ** 2 / 2))

    dist_dict = {'pop_size': pop_size,
                 'variants': variants,
                 'wt_freq': wt_freq,
                 'sigma': sigma,
                 'mean_count': mean_count,
                 'distn': distn}

    return dist_dict


def effect_distribution(wt_effect, wt_pctile, sigma, effect_min, effect_max,
                        effect_cname):
    """
    Create and return a dictionary containing a scipy.stats.truncnorm
    distribution object and its associated parameters, used to generate true
    variant effects. The truncated normal is used to avoid having probabilities
    outside (0, 1).

    Parameters
    ----------
    wt_effect : float
        True effect for wild type
    wt_pctile : int
        Percentile of wild type in the effect distribution
    sigma : float
        Standard deviation of the effect distribution
    effect_min : float
        True effect lower bound
    effect_max : float
        True effect upper bound
    effect_cname : str
        Column name for this effect ('pselect' or 'growth_rate')

    Returns
    -------
    dict
    """
    # calculate the normal distribution mean using the wild type percentile
    mean = wt_effect - scipy.stats.norm.ppf(wt_pctile / 100., scale=sigma)

    # convert effect min and max into normal distribution units
    trunc_min = (effect_min - mean) / sigma
    trunc_max = (effect_max - mean) / sigma

    distn = scipy.stats.truncnorm(trunc_min, trunc_max, loc=mean, scale=sigma)

    distn_dict = {'wt_effect': wt_effect,
                  'wt_pctile': wt_pctile,
                  'sigma': sigma,
                  'mean': mean,
                  'effect_min': effect_min,
                  'effect_max': effect_max,
                  'distn': distn,
                  'effect_cname': effect_cname}

    return distn_dict


def generate_starting_counts(ddict, conditions, rounds, replicates,
                             replicate_mode):
    """
    Create and return a pandas DataFrame containing starting counts for a
    simulation. Variant counts are randomly assigned using a log-normal
    distribution, with the exception of wild type, which is included at the
    specified frequency.

    Parameters
    ----------
    ddict : dict
        Dictionary containing the counts distribution and associated parameters
    conditions : list of str
        List of labels for the conditions in this simulation
    rounds : int
        Number of rounds of selection (including the input)
    replicates : int
        Number of replicate selections
    replicate_mode : str
        One of 'bio' (biological) or 'tech' (technical). Biological replicates
        have an independently-generated set of starting counts for each
        replicate. Technical replicates have the same starting counts for each
        replicate. In both cases, starting counts are the same across
        conditions.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
        If `replicate_mode` has an unexpected value
    """
    idx = pd.IndexSlice

    counts_df = pd.DataFrame(0, index=_create_variant_index(ddict['variants']),
                             columns=_create_counts_multiindex(conditions,
                                                               rounds,
                                                               replicates),
                             dtype=np.int32)

    # fill in the wild type starting count
    counts_df.loc['_wt', idx[:, :, 'c_0']] = np.round(
        ddict['pop_size'] * ddict['wt_freq']).astype(np.int32)

    # randomly generate variant counts
    if replicate_mode == 'bio':
        counts_df.loc[counts_df.index[1:], idx[:, :, 'c_0']] = np.tile(
            np.around(ddict['distn'].rvs(size=[ddict['variants'] - 1,
                                               replicates])),
            len(conditions)).astype(np.int32)
    elif replicate_mode == 'tech':
        counts_df.loc[counts_df.index[1:], idx[:, :, 'c_0']] = np.tile(
            np.around(ddict['distn'].rvs(size=ddict['variants'] - 1)),
            (replicates * len(conditions), 1)).transpose().astype(np.int32)
    else:
        raise ValueError("Invalid replicate mode '{}'".format(replicate_mode))

    return counts_df


def generate_true_effects(ddict, variants, conditions, replicates):
    """
    Create and return a pandas DataFrame containing true effects for each
    variant in each replicate selection. Each variant has the same true effect
    in all replicates across all conditions.

    Parameters
    ----------
    ddict : dict
        Dictionary containing the effect distribution and associated parameters
    variants : int
        Number of unique variants (including wild type)
    conditions : list of str
        List of labels for the conditions in this simulation
    replicates : int
        Number of replicate selections

    Returns
    -------
    pd.DataFrame
    """
    idx = pd.IndexSlice

    df = pd.DataFrame(index=_create_variant_index(variants),
                      columns=_create_effects_multiindex(conditions,
                                                         ddict['effect_cname'],
                                                         replicates),
                      dtype=np.float64)
    df.loc[df.index[1:], idx[:, :]] = np.tile(
        ddict['distn'].rvs(
            size=variants - 1), ((replicates + 1) * len(conditions), 1)).\
        transpose()
    df.loc['_wt', :] = ddict['wt_effect']

    return df


def add_effect_noise(effects, distn, condition, noise_freq):
    """
    Resample true effects for a subset of variants in the specified condition.
    This subset of variants will be assigned a new true effect in a single
    replicate in the specified condition.

    Parameters
    ----------
    effects : pd.DataFrame
        Contains true effects for each variant
    distn : scipy.stats.rv_continuous
        Distribution the effects will be generated from
    condition : str
        Condition label for the condition to add effect noise to
    noise_freq : float
        Frequency of unique variants with a resampled true effect (noisy
        variants)

    Returns
    -------
    None
    """
    affected_variants = np.random.choice(
        effects.index[1:], size=np.round(len(effects) * noise_freq).
        astype(np.int32), replace=False)
    indices = np.around(np.linspace(0, len(affected_variants),
                                    num=len(effects.columns.levels[1]))).\
        astype(np.int32)
    for i, r in enumerate(effects.columns.levels[1][1:]):
        effects.loc[affected_variants[indices[i]: indices[i + 1]],
                    (condition, r)] = distn.rvs(size=indices[i + 1] -
                                                     indices[i])


def _resample_counts(counts, depth, full_index=True):
    """
    Create and return a pandas Series that contains counts randomly resampled
    from the provided `counts` Series, using the provided counts as weights
    for the resampling.

    Parameters
    ----------
    counts : pd.Series
        Variants and counts to be resampled.
    depth : int
        Average number of copies per variant in the result.
    full_index : bool
        If True, return count of 0 for variants that were not sampled in the
        result. Else omit these variants from the result.

    Returns
    -------
    pd.Series
    """
    sample = np.random.choice(counts.index,
                              size=len(counts.index) * depth,
                              replace=True,
                              p=counts / counts.sum().astype(np.float64))
    sample_counts = pd.Series(sample).value_counts()

    if full_index:
        sample_counts = sample_counts.append(pd.Series(
            data=0, index=counts.index.difference(sample_counts.index),
            dtype=np.int32))
    return sample_counts


def run_binding(counts, condition, effects, pop_size, messages=True):
    """
    Perform a binding experiment simulation using draws from a binomial
    distribution. Resulting counts are stored in the provided `counts` pandas
    DataFrame. The population is resampled after each round of selection to
    simulate regrowth of cells to the given population size.

    Parameters
    ----------
    counts : pd.DataFrame
        Contains starting counts for the binding experiment
    condition : str
        Condition label for the condition to simulate
    effects : pd.DataFrame
        Contains binding true effects for each variant, which is the
        probability of binding used in the binomial for each variant.
    pop_size : int
        Total number of cells in the population
    messages : bool
        If True, print a status message to stdout after each round of selection
        is completed.

    Returns
    -------
    None
    """
    for r in counts.columns.levels[1]:
        for i in range(1, len(counts.columns.levels[2])):
            for v in counts.index:
                counts.loc[v, (condition, r, 'c_{}'.format(i))] = \
                    np.random.binomial(counts.loc[v, (condition, r, 'c_{}'.
                                                      format(i - 1))],
                                       effects.loc[v, (condition, r)])
            counts[(condition, r, 'c_{}'.format(i))] = \
                _resample_counts(counts[(condition, r, 'c_{}'.format(i))],
                                 depth=pop_size / len(counts.index))
            if messages:
                print("Finished {}, {}, round {}".format(condition, r, i))

    return


def run_growth(counts, condition, effects, pop_size, wt_doublings=2,
               messages=True):
    """
    Perform a binding experiment simulation using draws from a negative binomial
    distribution. Resulting counts are stored in the provided `counts` pandas
    DataFrame. The population is resampled after each round of selection to
    simulate regrowth of cells to the given population size.

    Parameters
    ----------
    counts : pd.DataFrame
        Contains starting counts for the binding experiment.
    condition : str
        Condition label for the condition to simulate
    effects : pd.DataFrame
        Contains growth rate true effects for each variant, which is the
        growth rate used to calculate the probability used in the negative
        binomial for each variant.
    pop_size : int
        Total number of cells in the population
    wt_doublings : int
        Number of wild type doublings between each time point.
    messages : bool
        If True, print a status message to stdout after each round of selection
        is completed.

    Returns
    -------
    None
    """
    t = wt_doublings * np.log(2) / \
        effects.loc['_wt', (counts.columns.levels[0][0], 'growth_rate')]

    for r in counts.columns.levels[1]:
        for i in range(1, len(counts.columns.levels[2])):
            for v in counts.index:
                c = counts.loc[v, (condition, r, 'c_{}'.format(i - 1))]
                # handle variant drop out
                if c > 0:
                    counts.loc[v, (condition, r, 'c_{}'.format(i))] = \
                        np.random.negative_binomial(c,
                            np.exp(-1 * t * effects.loc[v, (condition, r)]))
                else:
                    counts.loc[v, (condition, r, 'c_{}'.format(i))] = 0
            counts[(condition, r, 'c_{}'.format(i))] = \
                _resample_counts(counts[(condition, r, 'c_{}'.format(i))],
                                 depth=pop_size / len(counts.index))
            if messages:
                print("Finished {}, {}, round {}".format(condition, r, i))


def add_amplification_artifacts(counts, condition, pct_high,
                                mult_high, pct_low, mult_low):
    """
    Simulates amplification/depletion artifacts for a single condition.
    Variants may be randomly affected by an artifact in a single time point.
    No variant will be affected more than once.

    Parameters
    ----------
    counts : pd.DataFrame
        Contains counts for each time point in the experiment.
    condition : str
        Condition label for the condition to add amplification artifacts to
    pct_high : int
        Percentage of variants affected by an amplification event.
    mult_high : float
        Multiplier for amplified variants.
    pct_low : int
        Percentage of variants affected by a depletion event.
    mult_low : float
        Divisor for depleted variants.

    Returns
    -------
    None

    Raises
    ------
    ValueError if the percentages add up to more than 100%.
    """
    if pct_high + pct_low > 100:
        raise ValueError('Artifact percentages add up to more than 100%')
    elif pct_high + pct_low > 0:
        # randomly choose the right total number of affected variants
        affected_variants = np.random.choice(
            counts.index[1:], size=np.round(
                len(counts.index[1:]) *
                ((pct_high + pct_low) / 100.)).astype(np.int32), replace=False)
        # split up the affected variants between replicates
        indices = np.around(np.linspace(0, len(affected_variants), num=len(
            counts.columns.levels[1]) + 1)).astype(np.int32)
        # randomly choose an affected time point for each variant
        affected_timepoints = np.random.choice(
            counts.columns.levels[2], size=len(affected_variants),
            replace=True)
        for i, r in enumerate(counts.columns.levels[1]):
            # assign variants to amplification or depletion
            high_len = np.round(
                float(indices[i + 1] - indices[i]) * pct_high /
                (pct_high + pct_low)).astype(np.int32)
            high_variants = affected_variants[indices[i]: indices[i] + high_len]
            low_variants = affected_variants[
                indices[i] + high_len: indices[i + 1]]
            # time points need to be tuples for the multiindex
            high_timepoints = [(condition, r, x) for x in
                               affected_timepoints[
                               indices[i]: indices[i] + high_len]]
            low_timepoints = [(condition, r, x) for x in
                              affected_timepoints[
                              indices[i] + high_len: indices[i + 1]]]
            # perform amplifications
            for v, t in zip(high_variants, high_timepoints):
                counts.loc[v, t] = np.around(
                    counts.loc[v, t] * mult_high).astype(np.int32)
            # perform depletions
            for v, t in zip(low_variants, low_timepoints):
                counts.loc[v, t] = np.around(
                    counts.loc[v, t] / float(mult_low)).astype(np.int32)
        print("Finished adding jackpot variants.")
    else:
        print("Not generating jackpot variants.")


def generate_sequencing_counts(counts, depth):
    """
    Create and return a pandas DataFrame with simulated counts from sequencing
    a variant population.

    Parameters
    ----------
    counts : pd.DataFrame
        Contains counts for each time point in the experiment.
    depth : int
        Average number of reads per variant in the result.

    Returns
    -------
    pd.DataFrame
    """
    seq_counts = pd.DataFrame(0, index=counts.index, columns=counts.columns,
                              dtype=np.int32)

    for c in seq_counts.columns.levels[0]:
        for r in seq_counts.columns.levels[1]:
            for t in seq_counts.columns.levels[2]:
                seq_counts[(c, r, t)] = _resample_counts(counts[(c, r, t)],
                                                         depth)
    return seq_counts


def calculate_expected_growth_scores(effects, wt_doublings=2):
    """
    Create and return a pandas DataFrame with expected Enrich2 regression-based
    scores based on the variant true effects for a growth experiment.

    Parameters
    ----------
    effects : pd.DataFrame
        Contains growth rate true effects for each variant, which is the
        growth rate used to calculate the probability used in the negative
        binomial for each variant.
    wt_doublings : int
        Number of wild type doublings between each time point.

    Returns
    -------
    pd.DataFrame
    """
    scores = pd.DataFrame(index=effects.index, columns=['expected_score'],
                          dtype=np.float64)

    c = effects.columns.levels[0][0]
    n = len(effects.columns.levels[1]) - 1

    scores['expected_score'] = n * wt_doublings * \
        (effects[(c, 'growth_rate')] - effects.loc['_wt', (c, 'growth_rate')])
    return scores


def calculate_expected_binding_scores(effects):
    """
    Create and return a pandas DataFrame with expected Enrich2 regression-based
    scores based on the variant true effects for a binding experiment.

    Parameters
    ----------
    effects : pd.DataFrame
        Contains binding true effects for each variant, which is the
        probability of binding used in the binomial for each variant.

    Returns
    -------
    pd.DataFrame
    """
    scores = pd.DataFrame(index=effects.index, columns=['expected_score'],
                          dtype=np.float64)
    c = effects.columns.levels[0][0]
    n = len(effects.columns.levels[1]) - 1

    scores['expected_score'] = n * (np.log(effects[(c, 'pselect')]) -
                               np.log(effects.loc['_wt', (c, 'pselect')]))
    return scores


def output_simulation(sequence_counts, name, outdir):
    """
    Generate output files for the simulation.

    Parameters
    ----------
    sequence_counts : pd.DataFrame
        Contains the sequencing counts.
    name : str
        Experiment name.

    Returns
    -------
    None
    """
    result_dir = os.path.join(outdir, name, "Results")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    data_dir = os.path.join(outdir, name, "Data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    experiment = dict()
    experiment['name'] = name
    experiment['output directory'] = os.path.join(outdir, name, "Results")
    experiment['conditions'] = list()
    for c in sequence_counts.columns.levels[0]:
        cnd = dict()
        cnd['name'] = c
        cnd['selections'] = list()
        for r in sequence_counts.columns.levels[1]:
            sel = dict()
            sel['name'] = "{}_{}".format(c, r)
            sel['libraries'] = list()
            for t in sequence_counts.columns.levels[2]:
                seqlib = dict()
                seqlib['counts file'] = os.path.join(
                    data_dir, "{}_{}_{}.tsv".format(c, r, t))
                seqlib['identifiers'] = dict()
                seqlib['name'] = "{}_{}_{}".format(c, r, t)
                seqlib['report filtered reads'] = False
                seqlib['timepoint'] = int(t.split('_')[-1])
                sel['libraries'].append(seqlib)
            cnd['selections'].append(sel)
        experiment['conditions'].append(cnd)
    with open(os.path.join(outdir, name, 'config.json'), 'w') as \
            outfile:
        json.dump(experiment, outfile, sort_keys=True, indent=2)

    for c in sequence_counts.columns.levels[0]:
        for r in sequence_counts.columns.levels[1]:
            for t in sequence_counts.columns.levels[2]:
                with open(os.path.join(data_dir,
                                       "{}_{}_{}.tsv".format(c, r, t)),
                          'w') as outfile:
                    output_df = pd.DataFrame(sequence_counts.loc[:, (c, r, t)])
                    output_df.columns = ['count']
                    output_df = output_df.loc[output_df['count'] != 0]
                    output_df.to_csv(outfile, sep='\t')


if __name__ == "__main__":
    desc = "Deep mutational scanning dataset simulator."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("config", help="configuration file in .json format")
    args = parser.parse_args()

    with open(args.config) as handle:
        cfg = json.load(handle)

    idx = pd.IndexSlice

    # experiment parameters
    num_rounds = cfg['experiment']['rounds']
    num_replicates = cfg['experiment']['replicates']
    depths = cfg['experiment']['depths']

    # population parameters
    pop_size = cfg['population']['size']
    num_variants = cfg['population']['variants']
    starting_wt_freq = cfg['population']['starting wt freq']
    starting_counts_sigma = cfg['population']['starting counts sigma']

    count_dist_dict = starting_count_distribution(
        pop_size=pop_size, variants=num_variants, wt_freq=starting_wt_freq,
        sigma=starting_counts_sigma)

    # binding experiment parameters
    binding_wt_effect = cfg['binding']['wt effect']
    binding_wt_pctile = cfg['binding']['wt pctile']
    binding_sigma = cfg['binding']['sigma']
    binding_min_effect = cfg['binding']['min effect']
    binding_max_effect = cfg['binding']['max effect']

    binding_dist_dict = effect_distribution(
        wt_effect=binding_wt_effect, wt_pctile=binding_wt_pctile,
        sigma=binding_sigma, effect_min=binding_min_effect,
        effect_max=binding_max_effect, effect_cname='pselect')

    # growth experiment parameters
    growth_wt_effect = cfg['growth']['wt effect']
    growth_wt_pctile = cfg['growth']['wt pctile']
    growth_sigma = cfg['growth']['sigma']
    growth_min_effect = cfg['growth']['min effect']
    growth_max_effect = cfg['growth']['max effect']
    growth_wt_doublings = cfg['growth']['wt doublings']

    growth_dist_dict = effect_distribution(
        wt_effect=growth_wt_effect, wt_pctile=growth_wt_pctile,
        sigma=growth_sigma, effect_min=growth_min_effect,
        effect_max=growth_max_effect, effect_cname='growth_rate')

    # parameters for noise from PCR artifacts
    artifacts_pct_high = cfg['noise']['jackpot']['pct high']
    artifacts_pct_low = cfg['noise']['jackpot']['pct low']
    artifacts_mult_high = cfg['noise']['jackpot']['mult high']
    artifacts_mult_low = cfg['noise']['jackpot']['mult low']

    # parameters for noise from new true effects
    effect_noise_freq = cfg['noise']['reperror']['freq']

    conditions = sorted(['clean', 'reperror', 'jackpot'])
    for assay, effects_ddict, run_function, calc_expected in zip(
            ['binding', 'growth'], [binding_dist_dict, growth_dist_dict],
            [run_binding, run_growth],
            [calculate_expected_binding_scores,
             calculate_expected_growth_scores]):

        counts = generate_starting_counts(
            ddict=count_dist_dict, conditions=conditions, rounds=num_rounds,
            replicates=num_replicates, replicate_mode='bio')

        effects = generate_true_effects(
            ddict=effects_ddict, variants=num_variants,
            conditions=conditions, replicates=num_replicates)

        add_effect_noise(effects=effects, distn=effects_ddict['distn'],
                         condition='reperror', noise_freq=effect_noise_freq)

        run_function(counts=counts, condition='clean',
                     effects=effects, pop_size=pop_size)
        run_function(counts=counts, condition='reperror',
                     effects=effects, pop_size=pop_size)

        # copy the clean results and add amplification artifacts for the
        # jackpot noise case
        counts.loc[:, idx['jackpot', :, :]] = \
            counts.loc[:, idx['clean', :, :]].values
        add_amplification_artifacts(
            counts=counts, condition='jackpot',
            pct_high=artifacts_pct_high, mult_high=artifacts_mult_high,
            pct_low=artifacts_pct_low, mult_low=artifacts_pct_low)

        expected = calc_expected(effects)

        for d in depths:
            sequencing = generate_sequencing_counts(counts=counts, depth=d)

            name = "{}_simulation_depth_{}".format(assay, d)
            output_simulation(sequencing, name, cfg['outdir'])

            # output to HDF5
            store = pd.HDFStore(os.path.join(cfg['outdir'], name,
                                             "{}.h5".format(assay)))
            store.put(key='popcounts', value=counts)
            store.put(key='seqcounts', value=sequencing)
            store.put(key='effects', value=effects)
            store.put(key='expected', value=expected)
            store.close()



