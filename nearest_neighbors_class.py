"""
Problem: Given a set of N coordinates as (X,Y) pairs,
we want to compute how many coordinates are within R meters of an (X,Y) centroid
where the distance metric is Euclidean.
Goal: Design and write a class that is able to solve instances of this problem.
The interface should be simple, documented,
and allow a typical developer to use your API to efficiently query coordinates and centroids
to find coordinate counts in proximity to centroids.
Using this class and the sample data provided to provide solutions to the following questions.
1. How many coordinates are within 5 meters of at least one of the centroids?
2. How many coordinates are within 10 meters of at least one of the centroids?
3. What is the minimum radius R such that 80 percent of the coordinates are within R
meters of at least one of K centroids?
4. Bonus: What is the maximum radius R such that the number of coordinates within a
distance strictly less than R of any centroid is at most 1000?
Files:
1. coordinates.csv: contains 1 million X,Y pairs with a header, units in meters
2. centroids.csv: contains 1000 cluster centroids as X,Y pairs with a header, units in meters
Deliverable:
1. State your assumptions. Provide direction to run your code and
to recreate the solutions to the questions.
This includes installing all the dependencies, specifying path,
or running the executable. Assume the developer executing
and validating your code using Linux distribution.
2. Please provide simple unit tests for your software.
3. Provide solutions along with runtimes and peak memory usage for each question.
4. Document the computation and memory complexity of each API call in your class as a
function of the K centroids and N coordinates.
"""
import pandas as pd
import numpy as np
import torch, sys
from scipy.spatial import KDTree

class nearest_neighbor_search:

    def build_tree(self,data):
        data_tensor = torch.tensor((pd.read_csv(data, encoding="UTF-8")).values)
        return KDTree(data_tensor)

    def coord_within_radius(self, centroid_tree, coordinate_data, radius, k=1):
        nearest_d, nearest_i = centroid_tree.query(coordinate_data, k=k, distance_upper_bound=radius)
        nearest_i = filter(lambda x: x != len(centroid_tree.data), nearest_i)  # filter out values where index is out of range
        return list(nearest_i)

    def min_radius_with_percentage(self, centroid_tree, coordinate_data, percentile):
        nearest_d, nearest_i = centroid_tree.query(coordinate_data, k=1)
        return np.percentile(nearest_d, percentile, interpolation='linear')

    def max_radius_within_centroid(self, coordinate_tree, centroid_data, k):
        nearest_d, nearest_i = coordinate_tree.query(centroid_data, k=[k])
        return nearest_d.max()

