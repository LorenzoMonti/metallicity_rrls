import os
import numpy as np
import matplotlib.pyplot as plt

def plot_training_predictions(y_test, y_pred, results):

    plt.scatter(y_test, y_pred, color='blue', label= 'data')
    plt.plot(y_pred, y_pred, color='red', linewidth=2) 
    plt.title('Regressor')
    plt.xlabel('observed')
    plt.ylabel('predicted')
    plt.savefig(results)
    plt.close()

def plot_weights(alpha, grid, z, z_points, y_train, weights, results):
    
    plt.plot(grid, z)
    #plt.scatter(y_train, z_points, marker='+', label="alpha=" + str(alpha))
    plt.scatter(y_train, weights / weights.max(), marker='+')
    plt.legend()
    plt.savefig(results)
    plt.close()

def progress_plots(histories, filename, start_epoch: int = 0, title: str = None,
                   moving_avg: bool = False, beta: float = 0.9, plot_folds: bool = False):
    """
    Plot various metrics as a function of training epoch.

    :param histories: list
        List of history objects (each corresponding to a validation fold,
        captured from the output of the tf.keras.model.fit() method)
    :param filename: str
        Prefix for the name of the output figure file.
    :param start_epoch: int
        The first training epoch to be plotted.
    :param title: str
        Title of the figure.
    :param moving_avg: boolean
        If True, compute exponentially weighted moving averages (EWMA)
    :param beta: float
        Parameter for computing EWMA.
    :param plot_folds: boolean
        If True, individual cross-validation folds will be plotted in addition to their mean.
    """

    # Extract the list of metrics from histories:
    metrics = []
    for metric, _ in histories[0].history.items():
        if "val_" not in metric:
            metrics.append(metric)

    for metric in metrics:
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        fig.subplots_adjust(bottom=0.13, top=0.87, hspace=0.3, left=0.05, right=0.85, wspace=0)

        if title is not None:
            plt.title(title, fontsize=8)
        if metric == 'loss':
            plt.ylabel("log loss")
        else:
            plt.ylabel(metric)
        plt.xlabel('training epoch')
        train_seq = np.array([h.history[metric] for h in histories], dtype=object)
        
        if len(train_seq.shape) == 2:  # if there are the same number of epochs for each fold
            n_epochs = train_seq.shape[1]
            mean_train_seq = np.mean(train_seq, axis=0)
            epochs = np.linspace(1, n_epochs, n_epochs, endpoint=True).astype(int)

            # Plot training sequence:
            if metric == 'loss': pass
                #train_seq = np.log10(train_seq)
                #mean_train_seq = np.log10(mean_train_seq)
            line_train, = plt.plot(epochs[start_epoch:], mean_train_seq[start_epoch:], '-', c='darkred', linewidth=1,
                                   zorder=9, label='TR')
            if plot_folds:
                plt.plot(epochs[start_epoch:], train_seq.T[start_epoch:], 'r-', linewidth=1, zorder=8, alpha=0.3)

            if moving_avg:
                ewma_train = ewma(mean_train_seq, beta=beta)  # compute exponentially weighted moving averages
                plt.plot(epochs[start_epoch:], ewma_train[start_epoch:], 'k-', linewidth=1, zorder=10)

            if 'val_' + metric in histories[0].history:

                val_seq = np.array([h.history['val_' + metric] for h in histories])
                mean_val_seq = np.mean(val_seq, axis=0)
                if metric == 'loss':
                    val_seq = np.log10(val_seq)
                    mean_val_seq = np.log10(mean_val_seq)
                line_val, = plt.plot(epochs[start_epoch:], mean_val_seq[start_epoch:], 'g-', linewidth=1, zorder=11,
                                     label='CV')
                if plot_folds:
                    plt.plot(epochs[start_epoch:], val_seq.T[start_epoch:], 'g-', linewidth=1, zorder=12, alpha=0.3)
                if moving_avg:
                    ewma_val = ewma(mean_val_seq, beta)  # compute exponentially weighted moving averages
                    plt.plot(epochs[start_epoch:], ewma_val[start_epoch:], 'k-', linewidth=1, zorder=12)
                plt.legend(handles=[line_train, line_val], loc='upper left')
            else:
                plt.legend(handles=[line_train], loc='upper left')

        else:  # if each fold can have a different number of epochs

            # Plot training sequence:
            for tseq in train_seq:
                if metric == 'loss':
                    tseq = np.log10(np.array(tseq))
                else:
                    tseq = np.array(tseq)
                epochs = np.linspace(1, len(tseq), len(tseq), endpoint=True).astype(int)
                line_train, = plt.plot(epochs[start_epoch:], tseq[start_epoch:], 'r-', linewidth=1,
                                       zorder=8, alpha=0.5, label='TR')

            if 'val_' + metric in histories[0].history:
                val_seq = np.array([h.history['val_' + metric] for h in histories], dtype=object)
                for vseq in val_seq:
                    if metric == 'loss':
                        vseq = np.log10(np.array(vseq))
                    else:
                        vseq = np.array(vseq)
                    epochs = np.linspace(1, len(vseq), len(vseq), endpoint=True).astype(int)
                    line_val, = plt.plot(epochs[start_epoch:], vseq[start_epoch:], 'g-', linewidth=1,
                                         zorder=8, alpha=0.5, label='CV')
                    plt.legend(handles=[line_train, line_val], loc='upper left')
            else:
                plt.legend(handles=[line_train], loc='upper left')

        ax.tick_params(axis='both', direction='in', labelleft=False, labelright=True)
        ax.yaxis.tick_right()
        # if 'root_mean_squared_error' in metric:
        #     ax.set_ylim((0.1, 0.3))
        #     # plt.yticks(np.arange(0.0, 0.6, 0.1))
        # if 'loss' in metric:
        #     ax.set_ylim((-0.2, 0.25))
        # #     ax.set_yticks(np.arange(0.9, 1.005, 0.005))
        plt.grid(True)
        plt.savefig(filename + metric + '.pdf', format='pdf')
        plt.close(fig)

def ewma(y, beta=0.9, bias_corr=True):
    ma = np.zeros(y.shape)
    beta1 = 1.0 - beta

    v = 0
    for i in np.arange(len(y)) + 1:
        v = beta * v + beta1 * y[i - 1]
        if bias_corr:
            ma[i - 1] = v / (1. - beta ** i)
        else:
            ma[i - 1] = v
    return ma

def plot_predictions(y_train_true, y_train_pred, y_val_true=None, y_val_pred=None,
                     colors=None, suffix: str = '', figformat: str = 'png',
                     bins: str = 'sqrt', rootdir='.'):
    """
    Plot training and CV predictions vs true values and their histograms.
    """

    ll = np.linspace(-3, 0.5, 100)
    if y_val_true is not None and y_val_pred is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5, 10))
    else:
        fig, ax1 = plt.subplots(1, 1, sharex=True, figsize=(5, 5))
        ax2 = None
        ax1.set_xlabel('[Fe/H]')
    fig.subplots_adjust(bottom=0.13, top=0.87, hspace=0, left=0.15, right=0.95, wspace=0)
    if colors is not None:
        ax1.scatter(y_train_true, y_train_pred, s=2, marker='.', alpha=0.2, c=colors)
    else:
        ax1.scatter(y_train_true, y_train_pred, s=2, marker='.', c='k', alpha=0.2)
    ax1.plot(ll, ll, 'r-')
    # ax1.set_xlabel('[Fe/H]')
    ax1.set_ylabel('[Fe/H] (pred., T)')
    ax1.set_xlim((-3.1, 0.1))
    ax1.set_ylim((-3.1, 0.1))
    ax1.tick_params(direction='in')
    if ax2 is not None:
        if colors is not None:
            ax2.scatter(y_val_true, y_val_pred, s=2, marker='.', alpha=0.2, c=colors)
        else:
            ax2.scatter(y_val_true, y_val_pred, s=2, marker='.', c='k', alpha=0.2)
        ax2.plot(ll, ll, 'r-')
        ax2.set_xlabel('[Fe/H]')
        ax2.set_ylabel('[Fe/H] (pred., V)')
        ax2.set_xlim((-3.1, 0.1))
        ax2.set_ylim((-3.1, 0.1))
        ax2.tick_params(direction='in')
    plt.savefig(os.path.join(rootdir, 'pred_vs_true_' + suffix + '.' + figformat), format=figformat)
    plt.close(fig)

    # ---------------------------------------------------------------
    # Plot histograms of training and CV predictions and real values.

    if y_val_true is not None and y_val_pred is not None:
        fig, (ax12, ax34) = plt.subplots(2, 2, figsize=(10, 7))
        ax1, ax2 = ax12
        ax3, ax4 = ax34
    else:
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(5, 10))
        ax2, ax4 = None, None

    fig.subplots_adjust(bottom=0.13, top=0.87, hspace=0.2, wspace=0.2, left=0.15, right=0.95)

    if ax2 is not None:
        values, bins, _ = ax2.hist(y_val_true, facecolor='red', alpha=0.2, bins=bins, density=True)
        ax2.hist(y_val_pred, facecolor='black', alpha=0.2, bins=bins, density=True)
        ax2.set_xlabel('[Fe/H]')
        ax2.set_ylabel('norm. log count (V)')
        ax2.set_xlim((-2.6, 0.1))
        ax2.set_yscale('log')
        ax2.tick_params(direction='in')

    ax1.hist(y_train_true, facecolor='red', alpha=0.2, bins=bins, density=True, label="true")
    ax1.hist(y_train_pred, facecolor='black', alpha=0.2, bins=bins, density=True, label="predicted")
    ax1.set_xlabel('[Fe/H]')
    # ax1.set_xlabel('[Fe/H]')
    ax1.set_ylabel('norm. log count (T)')
    ax1.set_xlim((-2.6, 0.1))
    ax1.set_yscale('log')
    if ax2 is not None:
        ax1.legend(loc="upper right", bbox_to_anchor=(1.4, 1.2), fancybox=True, shadow=True, ncol=2)
    else:
        ax1.legend(loc="upper right", bbox_to_anchor=(0.8, 1.2), fancybox=True, shadow=True, ncol=2)
    ax1.tick_params(direction='in')

    ax3.hist(y_train_true, facecolor='red', alpha=0.2, bins=bins, density=True)
    ax3.hist(y_train_pred, facecolor='black', alpha=0.2, bins=bins, density=True)
    ax3.set_xlabel('[Fe/H]')
    ax3.set_ylabel('norm. count (T)')
    ax3.set_xlim((-2.6, 0.1))
    ax3.tick_params(direction='in')

    if ax4 is not None:
        ax4.hist(y_val_true, facecolor='red', alpha=0.2, bins=bins, density=True)
        ax4.hist(y_val_pred, facecolor='black', alpha=0.2, bins=bins, density=True)
        ax4.set_xlabel('[Fe/H]')
        ax4.set_xlabel('[Fe/H]')
        ax4.set_ylabel('norm. count (V)')
        ax4.set_xlim((-2.6, 0.1))
        ax4.tick_params(direction='in')

    plt.savefig(os.path.join(rootdir, 'pred_true_hist_' + suffix + '.' + figformat), format=figformat)
    plt.close(fig)

def plot_all_lc(phases, sequences, nmags=100, shift=0, fname=None, indx_highlight=None, figformat="png", nn_type='cnn'):
    """
    Plot all input time series as phase diagrams.
    """
    n_roll = int(np.round(shift * nmags))

    try:
        n_samples = sequences.shape[0]
    except:
        n_samples = len(sequences)

    if indx_highlight is not None:
        assert (indx_highlight >= 0)
        assert (indx_highlight == int(indx_highlight))
        assert (indx_highlight < n_samples)

    fig = plt.figure(figsize=(5, 4))
    fig.subplots_adjust(bottom=0.13, top=0.94, hspace=0.3, left=0.15, right=0.98, wspace=0)
    for ii in range(n_samples):
        if nn_type == "cnn":
            plt.plot(phases, np.roll(sequences[ii, :], n_roll), ls='-', color='grey', lw=0.3, alpha=0.3)
        elif nn_type == "rnn":
            ph = phases[ii]
            seq = sequences[ii]
            mask = (ph <= 1)
            seq = seq[mask]
            ph = ph[mask]
            # plt.plot(phases[ii], sequences[ii], ',', color='grey', alpha=0.5)
            plt.plot(ph, seq, ',', color='grey', alpha=0.5)
    if indx_highlight is not None:
        if nn_type == "cnn":
            plt.plot(phases, np.roll(sequences[indx_highlight, :], n_roll), 'ko')
        elif nn_type == "rnn":
            # plt.plot(phases[indx_highlight], sequences[indx_highlight], 'ko')
            ph = phases[indx_highlight]
            seq = sequences[indx_highlight]
            mask = (ph <= 1)
            seq = seq[mask]
            ph = ph[mask]
            plt.plot(ph, seq, 'ko')
    plt.xlabel('phase')
    plt.ylabel('mag')
    # plt.ylim(-1.1, 0.8)
    plt.gca().invert_yaxis()
    plt.savefig(fname + "." + figformat, format=figformat)
    plt.close(fig)
