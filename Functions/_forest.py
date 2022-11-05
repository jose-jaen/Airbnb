# Modified part of sklearn's ensemble forest.py (lines 169 - 196)
# Added exponentially distributed random samples in bootstrap
# Source (authors): Matt Taddy, Chun-Sheng Chen, Jun Yu, Mitch Wyle
# Source (repository): https://github.com/TaddyLab/bayesian-forest

def _parallel_build_trees(
    tree,
    bootstrap,
    X,
    y,
    sample_weight,
    tree_idx,
    n_trees,
    verbose=0,
    class_weight=None,
    n_samples_bootstrap=None,
):
    """
    Private function used to fit a single tree in parallel."""
    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))

    if bootstrap:
        n_samples = X.shape[0]
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
        else:
            curr_sample_weight = sample_weight.copy()
            
        # Adding Bayesian Bootsrap
        if bootstrap == 2:
            sample_counts = np.random.exponential(1, n_samples)
        else:
            indices = _generate_sample_indices(
                tree.random_state, n_samples, n_samples_bootstrap
            )
            sample_counts = np.bincount(indices, minlength=n_samples)
        curr_sample_weight *= sample_counts

        if class_weight == "subsample":
            with catch_warnings():
                simplefilter("ignore", DeprecationWarning)
                curr_sample_weight *= compute_sample_weight("auto", y, indices=indices)
        elif class_weight == "balanced_subsample":
            curr_sample_weight *= compute_sample_weight("balanced", y, indices=indices)

        tree.fit(X, y, sample_weight=curr_sample_weight, check_input=False)
    else:
        tree.fit(X, y, sample_weight=sample_weight, check_input=False)

    return tree
