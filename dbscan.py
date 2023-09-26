import pandas as pd
import numpy as np


class DBScan:
    def __init__(self, minPts: int, epsilon: float) -> None:
        self.__minPts = minPts
        self.__epsilon = epsilon

    def runAlgorithm(self, dataFrame: pd.DataFrame):
        self.__corePts = []

        self.__observations = dataFrame.values
        self.__neighborhoods = DBScan.__computeNeighborhoods(dataFrame, self.__epsilon)

        # Assignment of -1 denotes unassigned cluster
        self.__assignments = np.full(self.__observations.shape[0], -1)

        # Find core points
        for i, observation in enumerate(self.__observations):
            if len(self.__neighborhoods[i]) >= self.__minPts:
                self.__corePts.append(i)

        # Assign clusters
        self.__curCluster = 0
        for i in self.__corePts:
            if self.__assignments[i] != -1:
                continue
            self.__assignments[i] = self.__curCluster
            self.__densityConnected(i)
            self.__curCluster += 1

        noisePts = []
        borderPts = []
        for i, observation in enumerate(self.__observations):
            # If point is unassigned, it is noise
            if self.__assignments[i] == -1:
                noisePts.append(i)
            # Otherwise, if point is not core, it is border
            elif i not in self.__corePts:
                borderPts.append(i)

        return (self.__assignments, self.__corePts, borderPts, noisePts)

    def __densityConnected(self, i, connected: list = []):
        connected.append(i)
        for j in self.__neighborhoods[i]:
            self.__assignments[j] = self.__curCluster
            if j in self.__corePts and j not in connected:
                self.__densityConnected(j, connected)

    # Static methods

    def __distSquared(x1, x2):
        # Euclidian distance squared
        return np.sum(np.square(x1 - x2))

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
