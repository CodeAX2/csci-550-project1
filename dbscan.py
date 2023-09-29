import pandas as pd
import numpy as np

class DBScan:
    # Creates a new instance of DBScan
    # minPts denotes the minimum number of neighbor points to denote a core point
    # epsilon denotes the maximum distance between two points for them to still be considered in a neighborhood
    # noiseValue denotes the cluster value to assign to noise points
    def __init__(self, minPts: int, epsilon: float, noiseValue: int = -1) -> None:
        self.__minPts = minPts
        self.__epsilon = epsilon
        self.__noiseValue = noiseValue

    # Runs the DBScan algorithm on a given dataFrame
    # dataFrame denotes the data to run DBScan on
    # Returns a tuple containing the cluster assignments, the list of core points, list of border points, and list of noise points
    def runAlgorithm(self, dataFrame: pd.DataFrame):
        # Reset our list of core points
        self.__corePts = []

        # Set our list of all data points
        self.__observations = dataFrame.values

        # Calculate our neighborhoods
        self.__neighborhoods = DBScan.__computeNeighborhoods(dataFrame, self.__epsilon)

        # Assignment denotes unassigned cluster (useful for coloring noise points)
        self.__assignments = np.full(self.__observations.shape[0], self.__noiseValue)

        # Find core points
        for i, observation in enumerate(self.__observations):
            # Check we meet the minimum number of points
            if len(self.__neighborhoods[i]) >= self.__minPts:
                self.__corePts.append(i)

        # Assign clusters
        self.__curCluster = 1
        for i in self.__corePts:
            # Check the point is not already assigned
            if self.__assignments[i] != self.__noiseValue:
                continue
            # Assign the cluster
            self.__assignments[i] = self.__curCluster
            # Assign clusters of nearby points
            self.__densityConnected(i)
            # Go onto the next cluster
            self.__curCluster += 1

        # Calculate our noise and border points
        noisePts = []
        borderPts = []
        for i, observation in enumerate(self.__observations):
            # If point is unassigned, it is noise
            if self.__assignments[i] == self.__noiseValue:
                noisePts.append(i)
            # Otherwise, if point is not core, it is border
            elif i not in self.__corePts:
                borderPts.append(i)

        return (self.__assignments, self.__corePts, borderPts, noisePts)

    # Assigns clusters to points connected to a given points x
    def __densityConnected(self, x):

        # Performs a breadth-first search of nearby eligible points
        visited = set()
        toVisit = []

        toVisit.append(x)

        while len(toVisit) != 0:
            curPt = toVisit.pop()
            visited.add(curPt)
            for i in self.__neighborhoods[curPt]:
                if i not in visited:
                    self.__assignments[i] = self.__curCluster
                    if i in self.__corePts:
                        toVisit.append(i)

    # Static methods

    # Returns the squared euclidian distance between points x1 and x2
    def __distSquared(x1, x2):
        # Euclidian distance squared
        total = 0
        for i in range(len(x1)):
            total += (x1[i] - x2[i]) * (x1[i] - x2[i])
        return total

    # Computes the set of epsilon-neighborhoods for every point in a given dataframe
    def __computeNeighborhoods(dataFrame: pd.DataFrame, epsilon: float):
        observations = dataFrame.values

        neighborhoods = []
        epsilonSquared = epsilon * epsilon
        # Loop over each data point
        for i, currentObservation in enumerate(observations):
            # Generate neighborhood for the point
            curNeighborhood = set()
            # Check all other points
            for j, potentialNeighbor in enumerate(observations):
                # Do not self-reference point
                if i != j:
                    # Check if within epsilon
                    # (technically checking distSquared < epsilonSquared, since it is faster)
                    if (
                        DBScan.__distSquared(currentObservation, potentialNeighbor)
                        < epsilonSquared
                    ):
                        curNeighborhood.add(j)
            neighborhoods.append(curNeighborhood)

        return neighborhoods
