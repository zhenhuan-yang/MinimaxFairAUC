import numpy as np

class SummaryWritter:
    '''
    For writting stat and result by experiments
    '''
    def __init__(self, X_train, y_train, z_train):

        # save the input
        try:
            self.n_samples = X_train.shape[0]
            self.n_features = X_train.shape[1]
        except:
            self.n_samples = len(y_train)
            self.n_features = 0

        self.n_labels = len(np.unique(y_train))
        assert self.n_labels == 2, "Only support binary output!"
        self.n_groups = len(np.unique(z_train))
        # get the number of positive and negative samples
        self.n_neg = len(y_train[(y_train < 0.5)])
        self.n_pos = len(y_train[(y_train > 0.5)])
        # get the number of each grouplabel
        self.n_grouplabel = []
        # order: group neg pos
        for i in range(self.n_groups):
            self.n_grouplabel.append(len(z_train[(y_train < 0.5) & (z_train == i)]))
            self.n_grouplabel.append(len(z_train[(y_train > 0.5) & (z_train == i)]))
        # reshaped index
        n_grouplabel_cs = np.cumsum(self.n_grouplabel).tolist()
        n_grouplabel_cs = [0] + n_grouplabel_cs
        self.grouplabel_ind = []
        self.group_ind = []
        self.label_ind = []
        neg_ind = []
        pos_ind = []
        # order: group neg pos
        for i in range(self.n_groups):
            group_neg_ind = list(range(n_grouplabel_cs[2*i] , n_grouplabel_cs[2*i+1]))
            group_pos_ind = list(range(n_grouplabel_cs[2*i+1] , n_grouplabel_cs[2*i+2]))
            neg_ind += group_neg_ind
            pos_ind += group_pos_ind
            self.grouplabel_ind.append(group_neg_ind)
            self.grouplabel_ind.append(group_pos_ind)
            self.group_ind.append(group_neg_ind + group_pos_ind)
        self.label_ind.append(neg_ind)
        self.label_ind.append(pos_ind)

    def grouplabelsplitter(self, X_train, y_train, z_train):

        X_train_list = []

        for i in range(self.n_groups):
            X_train_group_neg = X_train[(y_train < 0.5) & (z_train == i)]
            X_train_group_pos = X_train[(y_train > 0.5) & (z_train == i)]
            X_train_list.append(X_train_group_neg)
            X_train_list.append(X_train_group_pos)
        
        return X_train_list
