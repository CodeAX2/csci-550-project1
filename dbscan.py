import pandas as pd
import numpy as np


class DBScan:
    def __init__(self, minPts: int, epsilon: float, noiseValue: int = -1) -> None:
        self.__minPts = minPts
        self.__epsilon = epsilon
        self.__noiseValue = noiseValue

    def runAlgorithm(self, dataFrame: pd.DataFrame):
        self.__corePts = []

        self.__observations = dataFrame.values
        self.__neighborhoods = DBScan.__computeNeighborhoods(dataFrame, self.__epsilon)

        # Assignment denotes unassigned cluster (useful for coloring noise points)
        self.__assignments = np.full(self.__observations.shape[0], self.__noiseValue)

        # Find core points
        for i, observation in enumerate(self.__observations):
            if len(self.__neighborhoods[i]) >= self.__minPts:
                self.__corePts.append(i)

        # Assign clusters
        self.__curCluster = 1
        for i in self.__corePts:
            if self.__assignments[i] != self.__noiseValue:
                continue
            self.__assignments[i] = self.__curCluster
            self.__densityConnected(i)
            self.__curCluster += 1

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

    def __densityConnected(self, x):
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

    def __distSquared(x1, x2):
        # Euclidian distance squared
        total = 0
        for x1i in x1:
            for x2i in x2:
                total += (x1i-x2i)^2
        return total

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
