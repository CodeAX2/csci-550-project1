import pandas as pd
import numpy as np
import random
import math


# Calculates clusters on a given data set using k-means
# dataFrame denotes the data to perform clustering on
# k denotes the number of clusters to find
# epsilon denotes the change threshold to determine the algorithm's termination
def kMeans(dataFrame: pd.DataFrame, k: int, epsilon: float):
    assignments = []
    observations = dataFrame.values

    # Create random assignments of all points
    for _ in observations:
        assignments.append(random.randint(0, k - 1))

    # Compute initial centroids
    centroids = np.zeros((k, observations.shape[1]))
    for i in range(0, k):
        centroids[i] = computeCentroid(observations, assignments, i)

    while True:
        # Save previous centroids
        prevCentroids = centroids
        assignments = []

        # Get best cluster for each observation
        for observation in observations:
            bestCluster = closestCentroid(observation, centroids)
            assignments.append(bestCluster)

        # Compute new centroids
        centroids = np.zeros((k, observations.shape[1]))
        for i in range(0, k):
            centroids[i] = computeCentroid(observations, assignments, i)

        # Check change in centroids
        changed = False
        for i in range(0, k):
            if dist(prevCentroids[i], centroids[i]) > epsilon:
                changed = True
                break

        # If centroids did not change, exit loop
        if not changed:
            break

    return (centroids, assignments)


# Computes the centroid of a desired cluster, means-based
# observations represents the set of all points in the dataset
# curAssignments represents the current cluster assignments for all points
# cluster denotes the specific cluster to compute the centroid for
def computeCentroid(observations: np.ndarray, curAssignments: list, cluster: int):
    # Center begins at origin
    center = np.zeros(observations.shape[1])
    numInCluster = 0
    # Loop over each observation
    for i, obs in enumerate(observations):
        # If observation matches desired cluster, add it to center
        if curAssignments[i] == cluster:
            center += obs
            numInCluster += 1
    # Divide by number in cluster
    if numInCluster != 0:
        center /= numInCluster
    else:
        # If cluster has no members, set center to +infty
        for i in range(observations.shape[1]):
            center[i] = np.inf
    return center


# Finds the closest centroid to a given point
# observation denotes the point to find a centroid for
# centroids is the list of centroid points
def closestCentroid(observation, centroids):
    closestIndex = 0
    minDist = np.inf
    # Check every centroid to find the closest
    for i, centroid in enumerate(centroids):
        curDist = dist(observation, centroid)
        # Require strict inequality, so that lower indexed centroids are prefered
        if curDist < minDist:
            minDist = curDist
            closestIndex = i

    return closestIndex


def dist(x1, x2):
    # Euclidian distance squared
    total = 0
    for i in range(len(x1)):
        total += (x1[i] - x2[i]) * (x1[i] - x2[i])
    return math.sqrt(total)
