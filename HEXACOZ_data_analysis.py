import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns

K = 7


def predict_clusters_from_fitted_model(fitted_model, new_data_file_path, columns_to_exclude=None):
    """
    Predicts cluster labels for new data based on a fitted KMeans model.

    Parameters:
    - fitted_model: A fitted KMeans model
    - new_data_file_path: File path to the new data CSV
    - columns_to_exclude: List of column names to exclude from the new data

    Returns:
    - predicted_clusters: Array of predicted cluster labels
    """
    # Read the new dataset CSV file
    new_data = pd.read_csv(new_data_file_path)

    # Optionally, remove specified columns
    if columns_to_exclude:
        new_data = new_data.drop(columns=columns_to_exclude)

    # Convert data to float
    new_data = new_data.astype(float)

    # Fill NaN values with the mean of each column
    new_data.fillna(new_data.mean(), inplace=True)

    # Predict clusters for all rows in the new dataset
    predicted_clusters = fitted_model.predict(new_data)

    return predicted_clusters


def predict_clusters_classifier(clf, new_data_file_path, columns_to_exclude=None):
    """
    Predicts cluster labels for new data based on a fitted KMeans model.

    Parameters:
    - fitted_model: A fitted KMeans model
    - new_data_file_path: File path to the new data CSV
    - columns_to_exclude: List of column names to exclude from the new data

    Returns:
    - predicted_clusters: Array of predicted cluster labels
    """
    # Read the new dataset CSV file
    new_data = pd.read_csv(new_data_file_path)

    # Optionally, remove specified columns
    if columns_to_exclude:
        new_data = new_data.drop(columns=columns_to_exclude)

    # Convert data to float
    new_data = new_data.astype(float)

    # Fill NaN values with the mean of each column
    new_data.fillna(new_data.mean(), inplace=True)

    return clf.predict_proba(new_data), clf.predict(new_data)


def predict_cluster_probabilities(fitted_model, new_data_file_path, columns_to_exclude=None):
    """
    Predicts cluster probabilities for new data based on a fitted KMeans model using distance metrics.

    Parameters:
    - fitted_model: A fitted KMeans model
    - new_data_file_path: File path to the new data CSV
    - columns_to_exclude: List of column names to exclude from the new data

    Returns:
    - predicted_probabilities: Array of predicted cluster probabilities
    """
    try:
        # Read the new dataset CSV file
        new_data = pd.read_csv(new_data_file_path)

        # Optionally, remove specified columns
        if columns_to_exclude:
            new_data = new_data.drop(columns=columns_to_exclude)

        # Convert data to float
        new_data = new_data.astype(float)

        # Fill NaN values with the mean of each column
        new_data.fillna(new_data.mean(), inplace=True)

        # Get the centroids from the fitted KMeans model
        centroids = fitted_model.cluster_centers_

        # Calculate the distance from each point to all centroids
        distances = cdist(new_data, centroids, 'euclidean')

        # Transform these distances into probabilities (inversely proportional to distance)
        probabilities = np.exp(-distances)
        sum_prob = np.sum(probabilities, axis=1)

        # Normalize the probabilities so that they sum to 1 for each data point
        predicted_probabilities = probabilities / sum_prob[:, None]

        return predicted_probabilities
    except Exception as e:
        return f"An error occurred: {e}"


def group_features_by_letter(df, letter):
    return df[[col for col in df.columns if col.startswith(letter)]].mean(axis=1)


def calc_correlation_matrix(k_fit, features):
    # Calculate the correlation matrix between cluster centroids
    cluster_centroids = pd.DataFrame(k_fit.cluster_centers_, columns=features.columns[:-1])
    cluster_correlation = cluster_centroids.corr()

    # Display the correlation matrix using a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cluster_correlation, annot=True, cmap='coolwarm', cbar=True, square=True, fmt='.2f',
                annot_kws={'size': 10})
    plt.title('Correlation Matrix Between Cluster Centroids')
    plt.show()


def run_kmeans(file_path, columns_to_exclude=None):
    # Read the CSV file
    data = pd.read_csv(file_path, delimiter='\t') # Change the delimiter if needed

    # Optionally, remove specified columns
    if columns_to_exclude:
        data = data.drop(columns=columns_to_exclude)

    data = data.astype(float)

    # Fill NaN values with the mean of each column
    data.fillna(data.mean(), inplace=True)

    # Extract features for clustering
    features = data.copy()
    kmeans = KMeans(n_clusters=K)
    k_fit = kmeans.fit(features)

    predictions = k_fit.labels_
    features['Clusters'] = predictions
    features_b = features.copy()

    clf = RandomForestClassifier()
    clf.fit(data, predictions)

    calc_correlation_matrix(k_fit, features)

    # Perform PCA for visualization
    # 3 components:
    pca = PCA(n_components=3)
    pca_fit = pca.fit_transform(features)

    # Create a DataFrame with the principal components
    df_pca = pd.DataFrame(data=pca_fit,
                          columns=['Principal Component 1', 'Principal Component 2', 'Principal Component 3'])
    df_pca['Cluster'] = predictions

    # Plotting:
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i, color in enumerate(sns.color_palette('Set2', K)):
        ax.scatter(df_pca.loc[df_pca['Cluster'] == i, 'Principal Component 1'],
                   df_pca.loc[df_pca['Cluster'] == i, 'Principal Component 2'],
                   df_pca.loc[df_pca['Cluster'] == i, 'Principal Component 3'],
                   c=[sns.color_palette('Set2')[i]],
                   edgecolor='white', s=100, alpha=0.8, label=f'Cluster {i}')

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.title('3-Component PCA with 7 Clusters')
    plt.legend()
    plt.show()

    # 2 components:
    pca = PCA(n_components=2)
    pca_fit = pca.fit_transform(features_b)
    df_pca = pd.DataFrame(data=pca_fit, columns=['PCA1', 'PCA2'])
    df_pca['Clusters'] = predictions
    df_pca.head()

    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='Clusters', palette='Set2', alpha=0.8)
    plt.title('2-Component PCA with 7 Clusters')
    plt.show()

    return k_fit, clf



file_path = r'C:\Users\noga\PycharmProjects\pythonProject\Neuro\hexaco_plus_z.csv'
columns_to_exclude = ['elapse', 'country', 'V1', 'V2']

n = 10
means = np.zeros(6)
for _ in range(20):
    fitted_model, clf = run_kmeans(file_path, columns_to_exclude)
    new_samples = r'C:\Users\noga\PycharmProjects\pythonProject\Neuro\new_samples.csv'
    means += predict_clusters_classifier(clf, new_samples)[1] / n
    print(means)


