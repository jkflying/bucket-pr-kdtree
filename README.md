# KDTree.h by Julian Kent #

A C++11 KD-Tree with the following features:

* single file
* header only
* high performance K Nearest Neighbor and ball searches
* dynamic insertions
* simple API
* depends only on the STL
* templatable on your custom data type to store in the leaves. No need to keep a separate data structure!
* templatable on double, float etc
* templatable on L1, SquaredL2 or custom distance functor
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

Included in the main.cpp are accuracy and performance tests. For a rough idea of performance:

### Static building (all points added upfront before any queries)###

* Adding 10 million random 4D points: 0.7 seconds
* Recursively splitting: 2.7 seconds
* Performing 500k 5-nearest-neighbor queries: 2.5 seconds

### Dynamic building (split as you add, tree gets full speed queries at any point during construction) ###

* Adding 10 million random 4D points: 8.6 seconds
* Recursively splitting: 0 seconds (already split)
* Performing 500k 5-nearest-neighbor queries: 3.5 seconds

As seen, there is a slight penalty for choosing dynamic insertion in the search time and build time. However, it is
less than rebuilding the tree several times, so for use-cases needing searches during data aquisition may be necessary.

# License #

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0.

A high level explanation of MPLv2: You may use this in any software provided you give attribution. You *must* make
available any changes you make to the source code of this file to anybody you distribute your software to.

Upstreaming features and bugfixes are highly appreciated via https://bitbucket.org/jkflying/KDTree.h

For additional licensing rights, feature requests or questions, please contact Julian Kent <jkflying@gmail.com>

# Example usage #
```
#!cpp
// setup
using tree_t = jk::tree::KDTree<std::string, 2>;
using point_t = std::array<double, 2>;
tree_t tree;
tree.addPoint(point_t{{1, 2}}, "George");
tree.addPoint(point_t{{1, 3}}, "Harold");
tree.addPoint(point_t{{7, 7}}, "Melvin");

// KNN search
point_t lazyMonsterLocation{{6, 6}}; // this monster will always try to eat the closest people
const std::size_t monsterHeads = 2; // this monster can eat two people at once
auto lazyMonsterVictims = tree.searchKnn(lazyMonsterLocation, monsterHeads);
for (const auto& victim : lazyMonsterVictims)
{
    std::cout << victim.payload << " closest to lazy monster, with distance " << sqrt(victim.distance) << "!"
                << std::endl;
}

// ball search
point_t stationaryMonsterLocation{{8, 8}}; // this monster doesn't move, so can only eat people that are close
const double neckLength = 6.0; // it can only reach within this range
auto potentialVictims = tree.searchBall(stationaryMonsterLocation, neckLength * neckLength); // metric is SquaredL2
std::cout << "Stationary monster can reach any of " << potentialVictims.size() << " people!" << std::endl;

// hybrid KNN/ball search
auto actualVictims
    = tree.searchCapacityLimitedBall(stationaryMonsterLocation, neckLength * neckLength, monsterHeads);
std::cout << "The stationary monster will try to eat ";
for (const auto& victim : actualVictims)
{
    std::cout << victim.payload << " and ";
}
std::cout << "nobody else." << std::endl;

```
Output:

    Melvin closest to lazy monster, with distance 1.41421!
    Harold closest to lazy monster, with distance 5.83095!
    Stationary monster can reach any of 1 people!
    The stationary monster will try to eat Melvin and nobody else.

# Tuning tips #

If you need to add a lot of points before doing any queries, set the optional `autosplit` parameter to false,
then call splitOutstanding(). This will reduce temporaries and result in a better balanced tree.

Set the bucket size to be at least twice the K in a typical KNN query. If you have more dimensions, it is better to
have a larger bucket size. 32 is a good starting point. If possible use powers of 2 for the bucket size.

If you experience linear search performance, check that you don't have a bunch of duplicate point locations. This
will result in the tree being unable to split the bucket the points are in, degrading search performance.

The tree adapts to the parallel-to-axis dimensionality of the problem. Thus, if there is one dimension with a much
larger scale than the others, most of the splitting will happen on this dimension. This is achieved by trying to
keep the bounding boxes of the data in the buckets equal lengths in all axes.

Random data performs worse than 'real world' data with structure. This is because real world data has tighter
bounding boxes, meaning more branches of the tree can be eliminated sooner. On pure random data, more than 7 dimensions
won't be much faster than linear. However, most data isn't actually random. The tree will adapt to any locally reduced
dimensionality, which is found in most real world data.

Hybrid ball/KNN searches are faster than either type on its own, because subtrees can be more aggresively eliminated.

# Release Notes #

0.3

* New feature: ball search, and ball/knn hybrid search which limits on both radius and number of values

0.2

* Performance enhancements:
    * Use indexing for better cache locality of bounds (don't depend on the allocator)
    * Recycle memory for less allocator pressure during building the tree incrementally
    * Pre-allocate the search stack to prevent extra allocations during the search
* Provide a 1-NN interface `search()` which only needs a single internal allocation per call

0.1

* First release of simple tree, written to "Good modern C++" standards.
