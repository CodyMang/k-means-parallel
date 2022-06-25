# Import Library

import matplotlib.pyplot as plt
import numpy as np
import csv
from math import dist
from sklearn.datasets import make_blobs,make_moons,make_circles
# Define Data
information = []
def main():
    data_points = []
    with open('blobs.txt', 'r') as f:
        for line in f.read().split('\n'):
            x = line.split(',')
            if x[0] and x[1]:    
                data_points.append((float(x[0]),float(x[1])))

    actual_colors = []
    with open('blobs_actual_colors.txt', 'r') as f:
        for line in f.read().split('\n'):
            if line:
                actual_colors.append(int(line))

    plt.subplot(222)
    plt.title('Actual Colors')
    plt.scatter([a[0] for a in data_points], [a[1] for a in data_points], marker="o", c=actual_colors, s=25, edgecolor="k")

    
    cluster_seeds = []
    with open('cluster_points.txt', 'r') as f:
        for line in f.read().split('\n'):
            x = line.split(',')
            if x:
                cluster_seeds.append((float(x[0]),float(x[1])))

    
    closest_to_blob_clusters = []
    for x,y in data_points:
         closest_to_blob_clusters.append(closest_to_idx(cluster_seeds,x,y))

    plt.subplot(223)
    plt.title('Our K-means Implementation')
    plt.scatter([a[0] for a in data_points], [a[1] for a in data_points], marker="o", c= closest_to_blob_clusters, s=25, edgecolor="k")
    for x,y in cluster_seeds:
        plt.scatter(x, y, marker='8',color = 'orange')

    plt.show()



def closest_to_idx(cluster_seeds,x,y):
        m = dist(cluster_seeds[0],(x,y))
        res = 0
        for idx,elem in enumerate(cluster_seeds):
            if dist(elem, (x,y)) < m:
              m = dist(elem, (x,y))
              res = idx
        return res

def make_blobs_dataset(output_file_name="blobs.txt",output_actual="blobs_actual_colors.txt"):
    n = 1000000 # number of datapoints
    # X1, Y1 = make_moons(n,noise=0.1)
    X1, Y1 = make_blobs(n, centers=5,n_features=5,cluster_std=30,center_box=(0,3000))
    with open(output_file_name ,'w') as f:
        for x in X1[:-1]:
            f.write(f"{x[0]},{x[1]}\n")
        f.write(f"{X1[-1][0]},{X1[-1][1]}")
    
    with open(output_actual, 'w') as f:
        for y in Y1[:-1]:
            f.write(f'{y}\n')
        f.write(f"{Y1[-1]}") # this way there is no empty line at the end


    # Display 
    plt.scatter(X1[:, 0], X1[:, 1], marker="o", c=Y1, s=25, edgecolor="k")
    plt.show()

def make_moons_dataset(output_file_name="moons.txt",output_actual="moons_actual_colors.txt"):
    n = 10000 # number of datapoints
    X1, Y1 = make_moons(n,noise=0.1)
    X1, Y1 = make_blobs(n, centers=5,n_features=5,cluster_std=30,center_box=(0,1000))
    with open(output_file_name ,'w') as f:
        for x in X1[:-1]:
            f.write(f"{x[0]},{x[1]}\n")
        f.write(f"{X1[-1][0]},{X1[-1][1]}")
    
    with open(output_actual, 'w') as f:
        for y in Y1[:-1]:
            f.write(f'{y}\n')
        f.write(f"{Y1[-1]}") # this way there is no empty line at the end

    plt.scatter(X1[:, 0], X1[:, 1], marker="o", c=Y1, s=25, edgecolor="k")
    plt.show()


if __name__ == '__main__':
    # make_blobs_dataset()

    main()

