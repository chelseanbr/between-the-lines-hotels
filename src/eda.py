import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Pandas df exploration ##############################

def get_nulls(df):
    # Get count, pct, and type of missing data (per column)
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    missing_data['Type'] = [df[col].dtype for col in missing_data.index]
    return missing_data

def print_unique_ct(df):
    # Print how many unique values each column has
    print('Count of Unique Values per Column:\n')
    for col in df.columns:
        print('{}: {}'.format(col, len(df[col].unique())))

def get_cols_of_type(df, type):
    # Print names of columns of given type
    cols = list(df.select_dtypes(type).columns)
    print('{} Columns ({}): \n{}'.format(type, len(cols), cols))
    return cols

#######################################################

# Plotting #############################################

def plot_pie(series, fig, ax):
    series.value_counts().plot.pie(ax=ax, autopct='%1.2f%%')    
    fig.tight_layout()
    return fig, ax

def plot_counts(df, col, fig, ax):
    sns.countplot(data=df, x=col, ax=ax)
    fig.tight_layout()
    return fig, ax



#######################################################

def plot_hist(df, var, fig, ax):
    # histogram of var
    sns.distplot(df[var], ax=ax)
    # skewness and kurtosis
    print('Skewness: {:.4f}'.format(df[var].skew()))
    print('Kurtosis: {:.4f}'.format(df[var].kurt()))
    return fig, ax

def plot_scatter(df, var, target, ylim=(0,800000)):
    # scatterplot of var/target
    data = pd.concat([df[target], df[var]], axis=1)
    data.plot.scatter(x=var, y=target, ylim=ylim, color='b')

def plot_boxplot(df, var, target, figsize=(8, 6), ylim=(0,800000)):
    # boxplot of var/target
    data = pd.concat([df[target], df[var]], axis=1)
    fig, ax = plt.subplots(figsize=figsize)
    fig = sns.boxplot(x=var, y=target, data=data)
    fig.axis(ymin=ylim[0], ymax=ylim[1])

def plot_corr(df, figsize=(12, 9), vmax=.8):
    # corr matrix of df
    corrmat = df.corr()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corrmat, vmax=vmax, square=True, xticklabels=True, yticklabels=True)

def plot_target_corr(df, target, num_vars=5, figsize=(12, 9)):
    # target correlation matrix
    corrmat = df.corr()
    cols = corrmat.nlargest(num_vars, target)[target].index
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.25)
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 12}, 
                     yticklabels=cols.values, xticklabels=cols.values)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    fig.tight_layout()

def plot_scattermatrix(df, cols, height=2.5):
    # scatter matrix
    sns.set()
    sns.pairplot(df[cols], height = height)

def plot_counts_bygroup(df, features, groupby, fig, axs):
    # fig, axs = plt.subplots(6, 4, figsize=(14,18))

    for feature, ax in zip(features, axs.flatten()[:len(features)]):
        ax = sns.countplot(data=df, x=feature, hue=groupby, ax=ax, order=df[feature].value_counts().index)
        ax.legend_.remove()

    fig.tight_layout()
    return fig, axs

def plot_topN_features(feature_importances, feature_list, N, fig, ax):
    # Plot the feature importance
    idxes = np.argsort(-feature_importances)
    feature_list[idxes]
    rev_sort_feature_importances = feature_importances[idxes]
    rev_sort_feature_cols = feature_list[idxes]

    feat_scores = pd.DataFrame({'Fraction of Samples Affected' : rev_sort_feature_importances[:N]},
                               index=rev_sort_feature_cols[:N])
    feat_scores = feat_scores.sort_values(by='Fraction of Samples Affected')
    feat_scores.plot(kind='barh', ax=ax)
    
    ax.set_title('Feature Importances (Top {})'.format(N), size=25)
    ax.set_ylabel('Features', size=25)
    return fig, ax, rev_sort_feature_cols

#######################################################
