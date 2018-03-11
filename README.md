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

# License #

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0.

For additional licensing rights, feature requests or questions, please contact Julian Kent <jkflying@gmail.com>

# Example usage #
```
#!cpp
jk.tree.KDTree<std::string, 2> tree;
tree.addPoint(std::array<double,2>{{1,2}}, "George");
tree.addPoint(std::array<double,2>{{1,3}}, "Harold");
tree.addPoint(std::array<double,2>{{7,7}}, "Melvin");

std::array<double,2> monsterLocation{{6,6}};
const std::size_t monsterHeads = 2;
auto nearestTwo = tree.searchKnn(monster, monsterHeads);
for (const auto& victim : nearestTwo)
{
    std::cout << victim.payload << " was eaten by the monster!" << std::endl;
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
