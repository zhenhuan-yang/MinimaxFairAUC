import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from myutils.roc_calculus import roc
from itertools import product, permutations
import os

class Plotter:

    def __init__(self, fig_path="", show=False, param=None):

        # self.fig_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), path)
        # self.save = save
        self.fig_path = fig_path
        self.show = show
        if param:
            self.param = param
        else:
            self.param = {
                # Use LaTeX to write all text
                # "text.usetex": True,
                "font.family": "sans-serif",
                # Use 10pt font in plots, to match 10pt font in document
                # "axes.labelsize": 12,
                # "font.size": 12,
                # Make the legend/label fonts a little smaller
                "legend.fontsize": 10,
                # "xtick.labelsize": 12,
                # "ytick.labelsize": 12,
                # marker
                # "markers.fillstyle": 'none',
                "figure.figsize": (4, 3),
                "figure.dpi": 100,
                "figure.autolayout": True,
                "savefig.dpi": 100,
                "savefig.bbox": "tight",
                "savefig.pad_inches": 0.
            }
        self.legend_prop = {'weight':'bold'}
        self.colorListIntra = ['blue', 'red']
        self.colorListInter = ['darkgreen', 'darkgoldenrod']
        self.colorListAll = ['blue', 'darkgreen', 'darkgoldenrod', 'red']
        self.shapeList = ['o', 'x']
        self.num2letter = {0:'a', 1:'b', 2:'a', 3:'b'}
        self.num2num = {0:'-1', 1:'1'}

    def plot_postdist(self, data_writter, summary_writter):
        
        plt.style.use('seaborn')
        plt.rcParams.update(self.param)
        plt.figure()

        for idx, i in enumerate(range(data_writter.n_groups)):
            plt.hist(np.average(summary_writter.y_pred_proba[:, data_writter.grouplabel_ind[2*i]], axis=0, weights=np.arange(1, 1 + summary_writter.n_steps)), 
                label='Y=-1, Z={}'.format(self.num2letter[2*i]), color=self.colorListALl[idx % data_writter.n_groups ** 2], bins=100, alpha=0.5)
            plt.hist(np.average(summary_writter.y_pred_proba[:, data_writter.grouplabel_ind[2*i+1]], axis=0, weights=np.arange(1, 1 + summary_writter.n_steps)), 
                label='Y=1, Z={}'.format(self.num2letter[2*i+1]), color=self.colorListAll[idx % data_writter.n_groups ** 2], bins=100, alpha=0.5)
            

        plt.legend(prop=self.legend_prop, loc='best')
        if self.fig_path:
            plt.savefig(os.path.join(self.fig_path, "postdist.png"))
        if self.show:
            plt.show()
        plt.close()
        return

    def plot_weight(self, data_writter, summary_writter):
        
        plt.style.use('seaborn')
        plt.rcParams.update(self.param)
        plt.figure()

        if hasattr(summary_writter, "groupgroupweights"):
            for idx, (i, j) in enumerate(product(range(data_writter.n_groups), range(data_writter.n_groups))):
                plt.plot(np.arange(1, 1 + summary_writter.n_steps), summary_writter.groupgroupweights[:, idx], linestyle='solid',
                         label='GW(Y=1, Y\'=-1, Z={}, Z\'={})'.format(self.num2letter[i], self.num2letter[j]), color=self.colorListAll[idx % data_writter.n_groups ** 2])
        elif hasattr(summary_writter, "xgroupweights"):
            for idx, (i, j) in enumerate(permutations(range(data_writter.n_groups), r=2)):
                plt.plot(np.arange(1, 1 + summary_writter.n_steps), summary_writter.xgroupweights[:, idx], linestyle='solid', label='GW(Y=1, Y\'=-1, Z={}, Z\'={})'.format(self.num2letter[i], self.num2letter[j]),
                         color=self.colorListInter[idx % (data_writter.n_groups ** 2 - data_writter.n_groups)])
        else:
            for i in range(data_writter.n_groups):
                plt.plot(np.arange(1, 1 + summary_writter.n_steps), summary_writter.groupweights[:, i], linestyle='solid', label='GW(Y=1, Y\'=-1, Z={}, Z\'={})'.format(self.num2letter[i], self.num2letter[i]),
                         color=self.colorListIntra[i % data_writter.n_groups])

        plt.legend(prop=self.legend_prop, loc='best')
        if self.fig_path:
            plt.savefig(os.path.join(self.fig_path, "weight.png"))
        if self.show:
            plt.show()
        plt.close()
        return

    def plot_loss(self, data_writter, train_writter, test_writter):
        
        plt.style.use('seaborn')
        plt.rcParams.update(self.param)
        plt.figure()
        if hasattr(train_writter, "groupgroupweights"):
            for idx, (i, j) in enumerate(product(range(data_writter.n_groups), range(data_writter.n_groups))):
                plt.plot(np.arange(1, 1 + train_writter.n_steps), train_writter.loss[:, idx],
                         linestyle='dashed', label='Train N-{} P-{}'.format(i, j), color=self.colorListAll[idx % data_writter.n_groups ** 2])
                plt.plot(np.arange(1, 1 + test_writter.n_steps), test_writter.loss[:, idx],
                         linestyle='solid', label='Test N-{} P-{}'.format(i, j), color=self.colorListAll[idx % data_writter.n_groups ** 2])

        elif hasattr(train_writter, "xgroupweights"):
            for idx, (i, j) in enumerate(permutations(range(data_writter.n_groups), r=2)):
                plt.plot(np.arange(1, 1 + train_writter.n_steps), train_writter.loss[:, idx],
                         linestyle='dashed', label='Train N-{} P-{}'.format(i, j), color=self.colorListInter[idx % data_writter.n_groups ** 2 - data_writter.n_groups])
                plt.plot(np.arange(1, 1 + test_writter.n_steps), test_writter.loss[:, idx],
                         linestyle='solid', label='Test N-{} P-{}'.format(i, j), color=self.colorListInter[idx % data_writter.n_groups ** 2 - data_writter.n_groups])

        else:
            for idx in range(data_writter.n_groups):
                plt.plot(np.arange(1, 1 + train_writter.n_steps), train_writter.loss[:, idx],
                         linestyle='dashed', label='TrainN-{} P-{}'.format(idx, idx), color=self.colorListIntra[idx % data_writter.n_groups ** 2])
                plt.plot(np.arange(1, 1 + test_writter.n_steps), test_writter.loss[:, idx],
                         linestyle='solid', label='Test N-{} P-{}'.format(idx, idx), color=self.colorListIntra[idx % data_writter.n_groups ** 2])


        plt.legend(prop=self.legend_prop, loc='best')
        if self.fig_path:
            plt.savefig(os.path.join(self.fig_path, "loss.png"))
        if self.show:
            plt.show()
        plt.close()
        return

    def plot_traj(self, train_writter, test_writter):
        
        plt.style.use('seaborn')
        plt.rcParams.update(self.param)
        plt.figure()

        if hasattr(train_writter, "groupgroupweights"):
            plt.plot(np.arange(1, 1 + train_writter.n_steps), np.multiply(train_writter.loss, train_writter.groupgroupweights).sum(axis=1),
                     linestyle='dashed', label='Train', color=self.colorListAll[0])
            plt.plot(np.arange(1, 1 + test_writter.n_steps), np.multiply(test_writter.loss, train_writter.groupgroupweights).sum(axis=1),
                     linestyle='solid', label='Test', color=self.colorListAll[0])

        elif hasattr(train_writter, "xgroupweights"):
            plt.plot(np.arange(1, 1 + train_writter.n_steps), np.multiply(train_writter.loss, train_writter.xgroupweights).sum(axis=1),
                     linestyle='dashed', label='Train', color=self.colorListAll[0])
            plt.plot(np.arange(1, 1 + train_writter.n_steps), np.multiply(test_writter.loss, train_writter.xgroupweights).sum(axis=1),
                     linestyle='solid', label='Test', color=self.colorListAll[0])
        else:
            plt.plot(np.arange(1, 1 + train_writter.n_steps), np.multiply(train_writter.loss, train_writter.groupweights).sum(axis=1),
                     linestyle='dashed', label='Train', color=self.colorListAll[0])
            plt.plot(np.arange(1, 1 + train_writter.n_steps), np.multiply(test_writter.loss, train_writter.groupweights).sum(axis=1),
                     linestyle='solid', label='Test', color=self.colorListAll[0])

        plt.legend(prop=self.legend_prop, loc='best')
        if self.fig_path:
            plt.savefig(os.path.join(self.fig_path, "trajectory.png"))
        if self.show:
            plt.show()
        plt.close()
        return

    def plot_mis(self, data_writter, fairness, baseline):
        
        plt.style.use('seaborn')
        plt.rcParams.update(self.param)
        plt.figure()

        for idx, i in enumerate(range(data_writter.n_groups)):
            plt.plot(np.arange(1, 1 + fairness.n_steps), fairness.agg_mis_grouperrs[i], label='fairness G-{}'.format(i),
                     linestyle='solid',
                     color=self.colorListIntra[idx % data_writter.n_groups])
            plt.plot(np.arange(1, 1 + fairness.n_steps), baseline.agg_mis_grouperrs[i] * np.ones(fairness.n_steps),
                     label='baseline G-{}'.format(i),
                     linestyle='dashed', color=self.colorListIntra[idx % data_writter.n_groups])

        plt.legend(prop=self.legend_prop, loc='best')
        if self.fig_path:
            plt.savefig(os.path.join(self.fig_path, "mis.png"))
        if self.show:
            plt.show()
        plt.close()
        return

    def plot_bce(self, data_writter, fairness, baseline):
        plt.figure()
        plt.style.use('seaborn')
        plt.rcParams.update(self.param)
        
        for idx, i in enumerate(range(data_writter.n_groups)):
            plt.plot(np.arange(1, 1 + fairness.n_steps), fairness.agg_bce_grouperrs[i], label='fairness G-{}'.format(i),
                     linestyle='solid',
                     color=self.colorListIntra[idx % data_writter.n_groups])
            plt.plot(np.arange(1, 1 + fairness.n_steps), baseline.agg_bce_grouperrs[i] * np.ones(fairness.n_steps),
                     label='baseline G-{}'.format(i),
                     linestyle='dashed', color=self.colorListIntra[idx % data_writter.n_groups])

        plt.legend(prop=self.legend_prop, loc='best')
        if self.fig_path:
            plt.savefig(os.path.join(self.fig_path, "bce.png"))
        if self.show:
            plt.show()
        plt.close()
        return

    def plot_auc(self, data_writter, fairness, baseline, legend=False, nab='all', metric='auc', start=0, end=None):
        
        plt.style.use('seaborn')
        plt.rcParams.update(self.param)
        plt.figure()
        assert start >= 0
        if end:
            assert start < end
            assert end <= fairness.n_steps
            weights = np.arange(1, 1 + (end - start))
        else:
            assert start < fairness.n_steps
            weights = np.arange(1, 1 + (fairness.n_steps - start))

        if metric == 'pauc':
            grouperrs = fairness.agg_pauc_grouperrs
            baseline_grouperrs = baseline.agg_pauc_grouperrs
        elif metric == 'lauc':
            grouperrs = fairness.agg_lauc_grouperrs
            baseline_grouperrs = baseline.agg_lauc_grouperrs
        else:
            grouperrs = fairness.agg_auc_grouperrs
            baseline_grouperrs = baseline.agg_auc_grouperrs

        for idx, (i, j) in enumerate(product(range(data_writter.n_groups), range(data_writter.n_groups))):
            if nab == 'Intra':
                if i == j:
                    plt.plot(weights, grouperrs[j + i * data_writter.n_groups, start:end], label='AUC(f|Y=1 Y\'=-1 Z={}, Z\'={})'.format(self.num2letter[j], self.num2letter[i]), linestyle='solid',
                            color=self.colorListAll[idx % data_writter.n_groups ** 2])
                    plt.plot(weights, baseline_grouperrs[j + i * data_writter.n_groups] * np.ones(len(weights)),
                            linestyle='dashed', color=self.colorListAll[idx % data_writter.n_groups ** 2])
            elif nab == 'Inter':
                if i != j:
                    plt.plot(weights, grouperrs[j + i * data_writter.n_groups, start:end], label='AUC(f|Y=1 Y\'=-1 Z={}, Z\'={})'.format(self.num2letter[j], self.num2letter[i]), linestyle='solid',
                        color=self.colorListAll[idx % data_writter.n_groups ** 2])
                    plt.plot(weights, baseline_grouperrs[j + i * data_writter.n_groups] * np.ones(len(weights)),
                        linestyle='dashed', color=self.colorListAll[idx % data_writter.n_groups ** 2])
            else:
                plt.plot(weights, grouperrs[j + i * data_writter.n_groups, start:end], label='AUC(f|Y=1 Y\'=-1 Z={}, Z\'={})'.format(self.num2letter[j], self.num2letter[i]), linestyle='solid',
                     color=self.colorListAll[idx % data_writter.n_groups ** 2])
                plt.plot(weights, baseline_grouperrs[j + i * data_writter.n_groups] * np.ones(len(weights)),
                     linestyle='dashed', color=self.colorListAll[idx % data_writter.n_groups ** 2])
        
        if legend:
            plt.legend(prop=self.legend_prop, loc='best')

        if self.fig_path:
            plt.savefig(os.path.join(self.fig_path, "auc.png"))
        if self.show:
            plt.show()
        plt.close()
        return

    def plot_roc(self, data_writter, fairness, baseline, start=0, end=None, legend=False, nab='all'):
        
        plt.style.use('seaborn')
        plt.rcParams.update(self.param)
        plt.figure()
        assert start >= 0
        if end:
            assert start < end
            assert end <= fairness.n_steps
            weights = np.arange(1, 1 + (end - start)) ** 2
        else:
            assert start < fairness.n_steps
            weights = np.arange(1, 1 + (fairness.n_steps - start)) ** 2
        for idx, (i, j) in enumerate(product(range(data_writter.n_groups), range(data_writter.n_groups))):
            fpr, tpr = roc(np.average(fairness.y_pred_proba[start:end, data_writter.grouplabel_ind[2*i]], axis=0,
                                      weights=weights),
                           np.average(fairness.y_pred_proba[start:end, data_writter.grouplabel_ind[2*j+1]], axis=0,
                                      weights=weights))
            baseline_fpr, baseline_tpr = roc(baseline.y_pred_proba[-1][data_writter.grouplabel_ind[2*i]],
                           baseline.y_pred_proba[-1][data_writter.grouplabel_ind[2*j+1]])
            if nab == 'Intra':
                if i == j:
                    plt.plot(fpr, tpr, label='ROC(f|Y=1 Y\'=-1 Z={}, Z\'={})'.format(self.num2letter[j], self.num2letter[i]), linestyle='solid',
                            color=self.colorListAll[idx % data_writter.n_groups ** 2])
                    plt.plot(baseline_fpr, baseline_tpr, linestyle='dashed', color=self.colorListAll[idx % data_writter.n_groups ** 2])
            elif nab == 'Inter':
                if i != j:
                    plt.plot(fpr, tpr, label='ROC(f|Y=1 Y\'=-1 Z={}, Z\'={})'.format(self.num2letter[j], self.num2letter[i]), linestyle='solid',
                            color=self.colorListAll[idx % data_writter.n_groups ** 2])
                    plt.plot(baseline_fpr, baseline_tpr, linestyle='dashed', color=self.colorListAll[idx % data_writter.n_groups ** 2])
            else:
                plt.plot(fpr, tpr, label='ROC(f|Y=1 Y\'=-1 Z={}, Z\'={})'.format(self.num2letter[j], self.num2letter[i]), linestyle='solid',
                        color=self.colorListAll[idx % data_writter.n_groups ** 2])
                plt.plot(baseline_fpr, baseline_tpr, linestyle='dashed', color=self.colorListAll[idx % data_writter.n_groups ** 2])
        if legend:
            plt.legend(prop=self.legend_prop, loc='best')

        if self.fig_path:
            plt.savefig(os.path.join(self.fig_path, "roc.png"))
        if self.show:
            plt.show()
        plt.close()
        return 

    def plot_two(self, data_writter, fairness, baseline, start=0, end=None, legend=False):
        
        
        plt.style.use('seaborn')
        plt.rcParams.update(self.param)
        double_figsize = tuple([2 * i for i in self.param['figure.figsize']])
        plt.rcParams.update({'figure.figsize': double_figsize})
        plt.figure()
        assert start >= 0
        if end:
            assert start < end
            assert end <= fairness.n_steps
            weights = np.arange(1, 1 + (end - start)) ** 2
        else:
            assert start < fairness.n_steps
            weights = np.arange(1, 1 + (fairness.n_steps - start)) ** 2
        plt.subplot(2,2,1)
        for idx, i in enumerate(range(data_writter.n_groups)):
            fpr, tpr = roc(np.average(fairness.y_pred_proba[start:end, data_writter.grouplabel_ind[2*i]], axis=0,
                                      weights=weights),
                           np.average(fairness.y_pred_proba[start:end, data_writter.grouplabel_ind[2*i+1]], axis=0,
                                      weights=weights))
            baseline_fpr, baseline_tpr = roc(baseline.y_pred_proba[-1][data_writter.grouplabel_ind[2*i]],
                           baseline.y_pred_proba[-1][data_writter.grouplabel_ind[2*i+1]])
            plt.plot(fpr, tpr, label='ROC(f|Y=1 Y\'=-1 Z={}, Z\'={})'.format(self.num2letter[i], self.num2letter[i]), linestyle='solid',
                    color=self.colorListIntra[idx % data_writter.n_groups ** 2])
            plt.plot(baseline_fpr, baseline_tpr, linestyle='dashed', color=self.colorListIntra[idx % data_writter.n_groups ** 2])
        if legend:
            plt.legend(prop=self.legend_prop, loc='best')

        plt.subplot(2,2,2)
        for idx, (i, j) in enumerate(permutations(range(data_writter.n_groups), r=2)):
            fpr, tpr = roc(np.average(fairness.y_pred_proba[start:end, data_writter.grouplabel_ind[2*i]], axis=0,
                                      weights=weights),
                           np.average(fairness.y_pred_proba[start:end, data_writter.grouplabel_ind[2*j+1]], axis=0,
                                      weights=weights))
            baseline_fpr, baseline_tpr = roc(baseline.y_pred_proba[-1][data_writter.grouplabel_ind[2*i]],
                           baseline.y_pred_proba[-1][data_writter.grouplabel_ind[2*j+1]])
            plt.plot(fpr, tpr, label='ROC(f|Y=1 Y\'=-1 Z={}, Z\'={})'.format(self.num2letter[j], self.num2letter[i]), linestyle='solid',
                    color=self.colorListIntra[idx % data_writter.n_groups ** 2])
            plt.plot(baseline_fpr, baseline_tpr, linestyle='dashed', color=self.colorListIntra[idx % data_writter.n_groups ** 2])
        if legend:
            plt.legend(prop=self.legend_prop, loc='best')

        if self.fig_path:
            plt.savefig(os.path.join(self.fig_path, "two.png"))
        if self.show:
            plt.show()
        plt.close()
        return

    def plot_data(self, X, Y, Z, legend=True):
        
        plt.style.use('seaborn')
        plt.rcParams.update(self.param)
        plt.figure()
        n_groups = len(np.unique(Z))
        for idx, (i, j) in enumerate(product(range(2), range(n_groups))):
            plt.scatter(X[(Y == i) & (Z == j), 0], X[(Y == i) & (Z == j), 1], label='N(X|Y={},Z={})'.format(self.num2num[i], self.num2letter[j]), 
            facecolor=self.colorListIntra[i], marker=self.shapeList[j], edgecolor= 'k' if j == 0 else 'none', alpha=0.5 if j == 0 else 0.2)
        if legend:
            plt.legend(loc= 'lower right', frameon=True, prop=self.legend_prop)

        if self.fig_path:
            plt.savefig(os.path.join(self.fig_path, "gaussian2d.png"))
        if self.show:
            plt.show()
        plt.close()
        return 
    

    def plot_four(self, X, Y, Z, data_writter, fairness, baseline, start=0, end=None, legend=False):

        assert start >= 0
        if end:
            assert start < end
            assert end <= fairness.n_steps
            weights = np.arange(1, 1 + (end - start)) ** 2
        else:
            assert start < fairness.n_steps
            weights = np.arange(1, 1 + (fairness.n_steps - start)) ** 2


        new_size = tuple([4 * i for i in self.param['figure.figsize']])
        plt.style.use('seaborn')
        plt.rcParams.update(self.param)
        plt.rcParams.update(({'figure.figsize': new_size}))

        plt.figure()
        plt.subplot(4,4,1)
        n_groups = len(np.unique(Z))
        for idx, (i, j) in enumerate(product(range(2), range(n_groups))):
            plt.scatter(X[(Y == i) & (Z == j), 0], X[(Y == i) & (Z == j), 1], label='N(X|Y={},Z={})'.format(self.num2num[i], self.num2letter[j]), 
            facecolor=self.colorListIntra[i], marker=self.shapeList[j], edgecolor= 'k' if j == 0 else 'none', alpha=0.5 if j == 0 else 0.2)
        if legend:
            plt.legend(loc= 'lower right', frameon=True, prop=self.legend_prop)

        plt.subplot(4,4,2)
        for i in range(2):
            score = np.average(fairness.y_pred_proba[:, data_writter.grouplabel_ind[i]], axis=0, weights=np.arange(1, 1 + fairness.n_steps) ** 2)
            baseline_score = np.average(baseline.y_pred_proba[:, data_writter.grouplabel_ind[i]], axis=0, weights=np.arange(1, 1 + baseline.n_steps) ** 2)
            kde = stats.gaussian_kde(score)  
            baseline_kde = stats.gaussian_kde(baseline_score)
            x = np.linspace(0, 1, score.shape[0])
            xx = np.linspace(0, 1, baseline_score.shape[0])
            plt.plot(x, kde(x), label='KDE(f|Y={}, Z=a)'.format(self.num2num[i]), color=self.colorListIntra[i], linestyle='solid')
            plt.plot(xx, baseline_kde(xx), color=self.colorListIntra[i], linestyle='dashed')
            plt.fill_between(x, kde(x), alpha=0.3, color=self.colorListIntra[i])
            plt.fill_between(xx, baseline_kde(xx), alpha=0.1, color=self.colorListIntra[i])
        if legend:
            plt.legend(prop=self.legend_prop, loc='best', frameon=True)    

        plt.subplot(4,4,3)
        for i in range(2):
            score = np.average(fairness.y_pred_proba[:, data_writter.grouplabel_ind[2+i]], axis=0, weights=np.arange(1, 1 + fairness.n_steps) ** 2)
            baseline_score = np.average(baseline.y_pred_proba[:, data_writter.grouplabel_ind[2+i]], axis=0, weights=np.arange(1, 1 + baseline.n_steps) ** 2)
            kde = stats.gaussian_kde(score)  
            baseline_kde = stats.gaussian_kde(baseline_score)
            x = np.linspace(0, 1, score.shape[0])
            xx = np.linspace(0, 1, baseline_score.shape[0])
            plt.plot(x, kde(x), label='KDE(f|Y={}, Z=b)'.format(self.num2num[i]), color=self.colorListIntra[i], linestyle='solid')
            plt.plot(xx, baseline_kde(xx), color=self.colorListIntra[i], linestyle='dashed')
            plt.fill_between(x, kde(x), alpha=0.3, color=self.colorListIntra[i])
            plt.fill_between(xx, baseline_kde(xx), alpha=0.1, color=self.colorListIntra[i])
        if legend:
            plt.legend(prop=self.legend_prop, loc='best', frameon=True)    
        
        plt.subplot(4,4,4)
        for idx, (i, j) in enumerate(product(range(data_writter.n_groups), range(data_writter.n_groups))):
            fpr, tpr = roc(np.average(fairness.y_pred_proba[start:end, data_writter.grouplabel_ind[2*i]], axis=0,
                                      weights=weights),
                           np.average(fairness.y_pred_proba[start:end, data_writter.grouplabel_ind[2*j+1]], axis=0,
                                      weights=weights))
            baseline_fpr, baseline_tpr = roc(baseline.y_pred_proba[-1][data_writter.grouplabel_ind[2*i]],
                           baseline.y_pred_proba[-1][data_writter.grouplabel_ind[2*j+1]])

            plt.plot(fpr, tpr, label='ROC(f|Y=1 Y\'=-1 Z={}, Z\'={})'.format(self.num2letter[j], self.num2letter[i]), linestyle='solid',
                        color=self.colorListAll[idx % data_writter.n_groups ** 2])
            plt.plot(baseline_fpr, baseline_tpr, linestyle='dashed', color=self.colorListAll[idx % data_writter.n_groups ** 2])
        if legend:
            plt.legend(prop=self.legend_prop, loc='best', frameon=True)
        if self.fig_path:
            plt.savefig(os.path.join(self.fig_path, "gaussian2d.png"))
        if self.show:
            plt.show()
        plt.close()
        return

if __name__ == "__main__":
    print(os.path.dirname(__file__))