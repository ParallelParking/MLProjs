import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm


def processdata(file):
    dataframe = pd.read_csv(file)

    wind_direction_mapping = {
        "N": 0,
        "NNE": 22.5,
        "NE": 45,
        "ENE": 67.5,
        "E": 90,
        "ESE": 112.5,
        "SE": 135,
        "SSE": 157.5,
        "S": 180,
        "SSW": 202.5,
        "SW": 225,
        "WSW": 247.5,
        "W": 270,
        "WNW": 292.5,
        "NW": 315,
        "NNW": 337.5
    }
    for column in ['WindGustDir', 'WindDir9am', 'WindDir3pm']:
        dataframe[f'{column}_deg'] = dataframe[column].map(wind_direction_mapping)
        dataframe[f'{column}_sin'] = np.sin(np.deg2rad(dataframe[f'{column}_deg']))
        dataframe[f'{column}_cos'] = np.cos(np.deg2rad(dataframe[f'{column}_deg']))
        dataframe.drop(f'{column}', axis=1, inplace=True)
        dataframe.drop(f'{column}_deg', axis=1, inplace=True)

    dataframe['RainToday'] = dataframe['RainToday'].map({'Yes': 1, 'No': 0})
    dataframe['RainTomorrow'] = dataframe['RainTomorrow'].map({'Yes': 1, 'No': 0})

    dataframe['Date'] = pd.to_datetime(dataframe['Date'], format='%Y-%m-%d')
    dataframe['Day'] = dataframe['Date'].dt.dayofyear
    dataframe['Year'] = dataframe['Date'].dt.year
    dataframe.drop('Date', axis=1, inplace=True)
    baseline_year = dataframe['Year'].min()
    dataframe['Year'] = dataframe['Year'] - baseline_year

    dataframe['Location'] = dataframe['Location'].astype('category').cat.codes - 2
    # location column is clean, +2 is to handle missing vals.

    threshold_percent = 0.15
    for feature in dataframe.columns:
        missing_percent = dataframe[feature].isnull().sum() / len(dataframe)
        if missing_percent > threshold_percent:
            dataframe.drop(feature, axis=1, inplace=True)
        else:
            dataframe[feature].fillna(dataframe[feature].mean(), inplace=True)

    scaler = StandardScaler()
    dataframe = pd.DataFrame(scaler.fit_transform(dataframe))

    return dataframe.to_numpy()


def clustering(data, cluster_count, tolerance=0.001):
    # np.random.seed(Random_seed)  for testing reproduceability
    centroids = data[np.random.choice(len(data), cluster_count, replace=False)]

    while True:
        distances_squared = np.linalg.norm(data[:, np.newaxis, :] - centroids, axis=2) ** 2
        cluster_updating = np.argmin(distances_squared, axis=1)

        centroid_updating = np.array([data[cluster_updating == i].mean(axis=0) for i in range(cluster_count)])

        if np.linalg.norm(centroid_updating - centroids) < tolerance:
            cluster_set = cluster_updating
            break

        centroids = centroid_updating

    return cluster_set, centroids


def calculate_inertia(data, centroids, cluster_set):
    return np.sum((data - centroids[cluster_set]) ** 2)


def find_optimal_cluster(data, cluster_count, iterations=10):
    optimal_inertia = float('inf')
    optimal_result = None

    for _ in range(iterations):
        cluster_set, centroids = clustering(data, cluster_count)

        inertia = calculate_inertia(data, centroids, cluster_set)
        if optimal_inertia > inertia:
            inertia, optimal_inertia = optimal_inertia, inertia
            optimal_result = (cluster_set, centroids)

    return optimal_result


def elbow_graphing(inertiae):
    plt.plot(np.arange(len(inertiae)) + 1, inertiae, marker='o')
    plt.axvline(x=4, color='red', linestyle='--')  # on testing, k = 3 looks nice
    plt.xlabel('K')
    plt.ylabel('inertia')
    plt.show()


def elbowing(data, k_count=10):
    inertiae = []
    for cluster_count in range(1, k_count + 1):
        cluster_set, centroids = find_optimal_cluster(data, cluster_count)
        inertiae = np.append(inertiae, calculate_inertia(data, centroids, cluster_set))

    k = 4
    return k, inertiae


def PCAdecomposition(data, cluster_set):
    pca = PCA()
    transformed_data = pca.fit_transform(data)
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=cluster_set, cmap='viridis')
    plt.title('K-means Clustering with PCA Visualization')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


# Calinski Harabasz score tester function
def CHS(data, k_count=10):
    chs = []
    for cluster_count in range(2, k_count + 1):
        cluster_set, centroids = find_optimal_cluster(data, cluster_count)
        chs = np.append(chs, calinski_harabasz_score(data, cluster_set))
        print(f"{cluster_count} done")

    plt.plot(range(2, k_count + 1), chs, marker='o')
    plt.title('Calinski-Harabasz Index for Different Values of k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Calinski-Harabasz Index Score')
    plt.show()


def silhouetting(data, k_count=6):
    for cluster_count in range(2, k_count + 1):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(data) + (k_count + 1) * 10])

        cluster_set, centroids = find_optimal_cluster(data, cluster_count)
        print(f"calc {cluster_count} done")

        silhouette_average = silhouette_score(data, cluster_set)
        print(f"For n_clusters ={cluster_count}\tThe average silhouette_score is :{silhouette_average}")

        sample_silhouette_values = silhouette_samples(data, cluster_set)

        y_lower = 10
        for i in range(cluster_count):
            ith_silhouette_values = sample_silhouette_values[cluster_set == i]
            ith_silhouette_values.sort()
            size_cluster_i = ith_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = cm.nipy_spectral(float(i) / cluster_count)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
        ax1.axvline(x=silhouette_average, color="red", linestyle="--")
        ax1.set_yticks([])
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % cluster_count,
            fontsize=14,
            fontweight="bold",
        )
    plt.show()


if __name__ == "__main__":
    df = processdata('weatherAUS.csv')

    # cc, i = elbowing(df)
    # silhouetting(df)
    # CHS(df)

    cs, c = find_optimal_cluster(df, 3)

    print(cs)
    print(c)

    PCAdecomposition(df, cs)
