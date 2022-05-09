from nearest_neighbors_class import *
import argparse
import sys
from datetime import datetime
import tracemalloc


path = "/Users/macbookair/PycharmProjects/EatLab/"
sys.path.append(path)

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description="This is Range Query program")
        parser.add_argument('--centroids', help='centroids.csv',default='centroids.csv')
        parser.add_argument('--coords', help='coordinates.csv',default='coordinates.csv')
        parser.add_argument('--r', help='minimum radius of centroid', default=10)
        parser.add_argument('--p', help='percentage of coordinates', default=80)
        parser.add_argument('--c', help='number of coordinates limit per centroids', default=1000)


        ns = parser.parse_args(sys.argv[1:])
        nns = nearest_neighbor_search()

        r = int(ns.r)   #radius
        p = int(ns.p)   #percentage
        cen = int(ns.c)  #centroid_limit

        tracemalloc.start()  # Compute peak memory allocation of tensor
        start = datetime.now()

        coordinates_kdtree = nns.build_tree(ns.coords)
        centroids_kdtree = nns.build_tree(ns.centroids)

        current, peak = tracemalloc.get_traced_memory()  # a tuple: (current_mem: int, peak_mem: int)
        print("Execution time = %f ms \n" % (
                (datetime.now() - start).total_seconds() * 1000))  # in millisecond
        print(f"{current:0.2f}, {peak:0.2f}")
        tracemalloc.stop()

        print("Coordinates withing within %s meters: %s" %(r,len(nns.coord_within_radius(centroids_kdtree, coordinates_kdtree.data, r, 1))))
        print("Min radius to cover %s %% of coordinates: %s m" %(p,nns.min_radius_with_percentage(centroids_kdtree, coordinates_kdtree.data,p)))
        print("Max radius s.t. no more than %s coordinates per charger: %s m" %(cen,nns.max_radius_within_centroid(
          coordinates_kdtree, centroids_kdtree.data, cen)))

    except argparse.ArgumentError as e:
        print(e)
        sys.exit(-1)