# KDTree.h by Julian Kent #

A C++11 KD-Tree that is:

* single file
* header only
* high performance K Nearest Neighbor searches
* allows dynamic insertions
* has a simple API
* depends only on the STL
* templatable on your custom data type to store in the leaves. No need to keep a separate data structure!
* templatable on double, float etc
* templatable on L1, L2 or custom distance metric
* templated on number of dimensions for efficient inlining

# Motivation #

I previously wrote a very high performance KD-Tree in Java, but these days I work mostly with C++.
I hadn't yet found a KD-Tree with a simple API that is also fast and doesn't bring a bunch of dependencies
or a weird build system, so I decided to write my own. It is based roughly on what I did in the Java tree, but of
course with a bunch of optimizations that weren't possible in Java.

It allows dynamic insertion of points, followed by queries, then insertion of more points (at the expense of some performance).
However this allows many use cases for problems where queries are interspersed with new data, and rebuilding the tree
every time isn't a viable option. The buckets allow for good splits even if the tree is built entirely dynamically.

# Performance #

Included in the main.cpp are accuracy and performance tests. For a rough idea for performance tests:

## Static building ##

* Adding 10 million random 4D points: 0.7 seconds
* Recursively splitting: 3.1 seconds
* Performing 500k 5-nearest-neighbor queries: 2.7 seconds

## Dynamic building (split as you go, tree gets full speed queries at any point) ##

* Adding 10 million random 4D points: 9.6 seconds
* Recursively splitting: 0 seconds (already split)
* Performing 500k 5-nearest-neighbor queries: 3.9 seconds

As seen, there is a slight penalty for choosing dynamic insertion in the search time and build time. However, it is
less than rebuilding the tree several times, so for use-cases needing searches during data aquisition may be necessary.

# License #

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0.

For additional licensing rights, feature requests or questions, please contact Julian Kent <jkflying@gmail.com>

# Example usage #
```
#!cpp
using tree_t = jk::tree::KDTree<std::string, 2>;
using point_t = std::array<double,2>;
tree_t tree;
tree.addPoint(point_t{{1,2}}, "George");
tree.addPoint(point_t{{1,3}}, "Harold");
tree.addPoint(point_t{{7,7}}, "Melvin");

point_t monsterLocation{{6,6}};
const std::size_t monsterHeads = 2;
auto victims = tree.searchK(monsterLocation, monsterHeads);
for (const auto& victim : victims)
{
    std::cout << victim.payload << " eaten by monster, with distance " << victim.distance << "!" << std::endl;
}
```
# Tuning tips #

If you need to add a lot of points before or between doing any queries, set the optional `autosplit` parameter to false,
then call splitOutstanding(). This will reduce temporaries and result in a better balanced tree.

Set the bucket size to be roughly twice the K in a KNN query. If you have more dimensions, it is better to have a
larger bucket size. 32 is a good starting point.

If you experience linear search performance, check that you don't have a bunch of duplicate point locations. This
will result in the tree being unable to split the bucket the points are in, degrading search performance.

The tree adapts to the parallel-to-axis dimensionality of the problem. Thus, if there is one dimension with a much
larger scale than the others, most of the splitting will happen on this dimension. This is achieved by trying to
keep the bounding boxes of the buckets equal lengths in all axes. Note, this does not imply that the subspace a
subtree is parent of is square, for example if the data distribution in it is unequal.

# Release Notes #

0.2

* Performance enhancements:

** Use indexing for better cache locality of bounds (don't depend on the allocator)

** Recycle memory for less allocator pressure during building the tree incrementally

** Pre-allocate the search stack to prevent extra allocations during the search

** Provide a 1-NN interface which only needs a single internal allocation (of search stack depth, for unwinding)


0.1

* First release of simple tree, written to "Good modern C++" standards.
