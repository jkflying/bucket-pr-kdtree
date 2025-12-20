#pragma once

/**
 * KDTree.h by Julian Kent
 * A C++11 KD-Tree with the following features:
 *     single file
 *     header only
 *     high performance K Nearest Neighbor and ball searches
 *     dynamic insertions
 *     simple API
 *     depends only on the STL
 *     templatable on your custom data type to store in the leaves. No need to keep a separate data structure!
 *     templatable on double, float etc
 *     templatable on L1, SquaredL2 or custom distance functor
 *     templated on number of dimensions for efficient inlining
 *
 * -------------------------------------------------------------------
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not
 * distributed with this  file, you can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * A high level explanation of MPLv2: You may use this in any software provided you give attribution. You *must* make
 * available any changes you make to the source code of this file to anybody you distribute your software to.
 *
 * Upstreaming features and bugfixes are highly appreciated via:
 *
 * https://github.com/jkflying/bucket-pr-kdtree/tree/master/C%2B%2B
 *
 * For additional licensing rights, feature requests or questions, please contact Julian Kent <jkflying@gmail.com>
 *
 * -------------------------------------------------------------------
 *
 * Example usage:
 *
 * // setup
 * using tree_t = jk::tree::KDTree<std::string, 2>;
 * using point_t = std::array<double, 2>;
 * tree_t tree;
 * tree.addPoint(point_t{{1, 2}}, "George");
 * tree.addPoint(point_t{{1, 3}}, "Harold");
 * tree.addPoint(point_t{{7, 7}}, "Melvin");
 *
 * // KNN search
 * point_t lazyMonsterLocation{{6, 6}}; // this monster will always try to eat the closest people
 * const std::size_t monsterHeads = 2; // this monster can eat two people at once
 * auto lazyMonsterVictims = tree.searchKnn(lazyMonsterLocation, monsterHeads);
 * for (const auto& victim : lazyMonsterVictims)
 * {
 *     std::cout << victim.payload << " closest to lazy monster, with distance " << sqrt(victim.distance) << "!"
 *               << std::endl;
 * }
 *
 * // ball search
 * point_t stationaryMonsterLocation{{8, 8}}; // this monster doesn't move, so can only eat people that are close
 * const double neckLength = 6.0; // it can only reach within this range
 * auto potentialVictims = tree.searchBall(stationaryMonsterLocation, neckLength * neckLength); // metric is SquaredL2
 * std::cout << "Stationary monster can reach any of " << potentialVictims.size() << " people!" << std::endl;
 *
 * // hybrid KNN/ball search
 * auto actualVictims
 *     = tree.searchCapacityLimitedBall(stationaryMonsterLocation, neckLength * neckLength, monsterHeads);
 * std::cout << "The stationary monster will try to eat ";
 * for (const auto& victim : actualVictims)
 * {
 *     std::cout << victim.payload << " and ";
 * }
 * std::cout << "nobody else." << std::endl;
 *
 * Output:
 *
 * Melvin closest to lazy monster, with distance 1.41421!
 * Harold closest to lazy monster, with distance 5.83095!
 * Stationary monster can reach any of 1 people!
 * The stationary monster will try to eat Melvin and nobody else.
 *
 * -------------------------------------------------------------------
 *
 * Tuning tips:
 *
 * If you need to add a lot of points before doing any queries, set the optional `autosplit` parameter to false,
 * then call splitOutstanding(). This will reduce temporaries and result in a better balanced tree.
 *
 * Set the bucket size to be at least twice the K in a typical KNN query. If you have more dimensions, it is better to
 * have a larger bucket size. 32 is a good starting point. If possible use powers of 2 for the bucket size.
 *
 * If you experience linear search performance, check that you don't have a bunch of duplicate point locations. This
 * will result in the tree being unable to split the bucket the points are in, degrading search performance.
 *
 * The tree adapts to the parallel-to-axis dimensionality of the problem. Thus, if there is one dimension with a much
 * larger scale than the others, most of the splitting will happen on this dimension. This is achieved by trying to
 * keep the bounding boxes of the data in the buckets equal lengths in all axes.
 *
 * Random data performs worse than 'real world' data with structure. This is because real world data has tighter
 * bounding boxes, meaning more branches of the tree can be eliminated sooner.
 *
 * On pure random data, more than 7 dimensions won't be much faster than linear. However, most data isn't actually
 * random. The tree will adapt to any locally reduced dimensionality, which is found in most real world data.
 *
 * Hybrid ball/KNN searches are faster than either type on its own, because subtrees can be more aggresively eliminated.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <queue>
#include <set>
#include <vector>

namespace jk
{
namespace tree
{
    template <std::size_t Dim> struct Loop
    {
        template <typename F> static void run(F f)
        {
            Loop<Dim - 1>::run(f);
            f(Dim - 1);
        }
    };

    template <> struct Loop<0>
    {
        template <typename F> static void run(F f) { }
    };

    struct L1
    {
        template <std::size_t Dimensions, typename Scalar>
        static Scalar distance(const std::array<Scalar, Dimensions>& location1,
                               const std::array<Scalar, Dimensions>& location2)
        {
            auto abs = [](Scalar v) { return v >= 0 ? v : -v; };
            Scalar dist = 0;
            Loop<Dimensions>::run([&](std::size_t i) { dist += abs(location1[i] - location2[i]); });
            return dist;
        }
    };

    struct SquaredL2
    {
        template <std::size_t Dimensions, typename Scalar>
        static Scalar distance(const std::array<Scalar, Dimensions>& location1,
                               const std::array<Scalar, Dimensions>& location2)
        {
            auto sqr = [](Scalar v) { return v * v; };
            Scalar dist = 0;
            Loop<Dimensions>::run([&](std::size_t i) { dist += sqr(location1[i] - location2[i]); });
            return dist;
        }
    };

    template <class Payload,
              std::size_t Dimensions,
              std::size_t BucketSize = 32,
              class Distance = SquaredL2,
              typename Scalar = double>
    class KDTree
    {
    public:
        using distance_t = Distance;
        using scalar_t = Scalar;
        using payload_t = Payload;
        using point_t = std::array<Scalar, Dimensions>;
        static const std::size_t dimensions = Dimensions;
        static const std::size_t bucketSize = BucketSize;
        using tree_t = KDTree<Payload, Dimensions, BucketSize, Distance, Scalar>;

        struct LocationPayload
        {
            point_t location;
            Payload payload;
        };

    private:
        struct Node;
        struct Range
        {
            Scalar min, max;
        };
        using bounds_t = std::array<Range, Dimensions>;
        std::vector<Node> m_nodes;
        std::vector<bounds_t> m_nodeBounds;
        std::vector<std::size_t> m_waitingForSplit;
        std::vector<LocationPayload> m_data;

    public:
        KDTree()
        {
            m_nodes.emplace_back(0, 0); // initialize the root node
            m_nodeBounds.emplace_back();
            m_nodeBounds.back().fill({std::numeric_limits<Scalar>::max(), std::numeric_limits<Scalar>::lowest()});
        }

        size_t size() const { return m_data.size(); }

        void addPoint(const point_t& location, const Payload& payload, bool autosplit = true)
        {
            std::size_t addNode = 0;

            while (m_nodes[addNode].m_splitDimension != Dimensions)
            {
                expandBounds(addNode, location);
                if (location[m_nodes[addNode].m_splitDimension] < m_nodes[addNode].m_splitValue)
                {
                    addNode = m_nodes[addNode].m_children.first;
                }
                else
                {
                    addNode = m_nodes[addNode].m_children.second;
                }
            }

            std::size_t insertPos = m_nodes[addNode].m_dataEnd;
            m_data.insert(m_data.begin() + insertPos, LocationPayload {location, payload});
            m_nodes[addNode].m_dataEnd++;
            expandBounds(addNode, location);

            // Update all subsequent nodes' offsets if they were affected by the insert
            for (std::size_t i = 0; i < m_nodes.size(); ++i)
            {
                if (i != addNode && m_nodes[i].m_splitDimension == Dimensions && m_nodes[i].m_dataBegin >= insertPos)
                {
                    m_nodes[i].m_dataBegin++;
                    m_nodes[i].m_dataEnd++;
                }
            }

            if (m_nodes[addNode].shouldSplit() && m_nodes[addNode].m_entries % BucketSize == 0)
            {
                if (autosplit)
                {
                    split(addNode);
                }
                else if (!m_nodes[addNode].m_isWaitingForSplit)
                {
                    m_nodes[addNode].m_isWaitingForSplit = true;
                    m_waitingForSplit.push_back(addNode);
                }
            }
        }

        using iterator = typename std::vector<LocationPayload>::const_iterator;
        iterator begin() const { return m_data.begin(); }
        iterator end() const { return m_data.end(); }

        void rebalance()
        {
            tree_t newTree;
            newTree.m_data.reserve(m_data.size());
            for (const auto& lp : m_data)
            {
                newTree.addPoint(lp.location, lp.payload, false);
            }
            newTree.splitOutstanding();
            *this = std::move(newTree);
        }

        void splitOutstanding()
        {
            std::vector<std::size_t> searchStack;
            std::swap(searchStack, m_waitingForSplit);
            while (searchStack.size() > 0)
            {
                std::size_t addNode = searchStack.back();
                searchStack.pop_back();
                m_nodes[addNode].m_isWaitingForSplit = false;
                if (m_nodes[addNode].m_splitDimension == Dimensions && m_nodes[addNode].shouldSplit() && split(addNode))
                {
                    searchStack.push_back(m_nodes[addNode].m_children.first);
                    searchStack.push_back(m_nodes[addNode].m_children.second);
                }
            }
        }

        struct DistancePayload
        {
            Scalar distance;
            Payload payload;
            bool operator<(const DistancePayload& dp) const { return distance < dp.distance; }
        };

        std::vector<DistancePayload> searchKnn(const point_t& location, std::size_t maxPoints) const
        {
            return searcher().search(location, std::numeric_limits<Scalar>::max(), maxPoints);
        }

        std::vector<DistancePayload> searchBall(const point_t& location, Scalar maxRadius) const
        {
            return searcher().search(location, maxRadius, std::numeric_limits<std::size_t>::max());
        }

        std::vector<DistancePayload> searchCapacityLimitedBall(const point_t& location,
                                                               Scalar maxRadius,
                                                               std::size_t maxPoints) const
        {
            return searcher().search(location, maxRadius, maxPoints);
        }

        DistancePayload search(const point_t& location) const
        {
            DistancePayload result;
            result.distance = std::numeric_limits<Scalar>::infinity();

            if (m_nodes[0].m_entries > 0)
            {
                std::vector<std::size_t> searchStack;
                searchStack.reserve(1 + std::size_t(1.5 * std::log2(1 + m_nodes[0].m_entries / BucketSize)));
                searchStack.push_back(0);

                while (searchStack.size() > 0)
                {
                    std::size_t nodeIndex = searchStack.back();
                    searchStack.pop_back();
                    const Node& node = m_nodes[nodeIndex];
                    if (result.distance > pointRectDist(nodeIndex, location))
                    {
                        if (node.m_splitDimension == Dimensions)
                        {
                            for (std::size_t i = node.m_dataBegin; i < node.m_dataEnd; ++i)
                            {
                                const auto& lp = m_data[i];
                                Scalar nodeDist = Distance::distance(location, lp.location);
                                if (nodeDist < result.distance)
                                {
                                    result = DistancePayload {nodeDist, lp.payload};
                                }
                            }
                        }
                        else
                        {
                            node.queueChildren(location, searchStack);
                        }
                    }
                }
            }
            return result;
        }

        class Searcher
        {
        public:
            Searcher(const tree_t& tree) : m_tree(tree) { }
            Searcher(const Searcher& searcher) : m_tree(searcher.m_tree) { }

            // NB! this method is not const. Do not call this on same instance from different threads simultaneously.
            const std::vector<DistancePayload>& search(const point_t& location, Scalar maxRadius, std::size_t maxPoints)
            {
                // clear results from last time
                m_results.clear();

                // reserve capacities
                m_searchStack.reserve(1 + std::size_t(1.5 * std::log2(1 + m_tree.m_nodes[0].m_entries / BucketSize)));
                if (m_prioqueueCapacity < maxPoints && maxPoints < m_tree.m_nodes[0].m_entries)
                {
                    std::vector<DistancePayload> container;
                    container.reserve(maxPoints);
                    m_prioqueue = std::priority_queue<DistancePayload, std::vector<DistancePayload>>(
                        std::less<DistancePayload>(), std::move(container));
                    m_prioqueueCapacity = maxPoints;
                }

                m_tree.searchCapacityLimitedBall(location, maxRadius, maxPoints, m_searchStack, m_prioqueue, m_results);

                m_prioqueueCapacity = std::max(m_prioqueueCapacity, m_results.size());
                return m_results;
            }

        private:
            const tree_t& m_tree;

            std::vector<std::size_t> m_searchStack;
            std::priority_queue<DistancePayload, std::vector<DistancePayload>> m_prioqueue;
            std::size_t m_prioqueueCapacity = 0;
            std::vector<DistancePayload> m_results;
        };

        // NB! returned class has no const methods. Get one instance per thread!
        Searcher searcher() const { return Searcher(*this); }

    private:
        void expandBounds(std::size_t nodeIndex, const point_t& location)
        {
            bounds_t& bounds = m_nodeBounds[nodeIndex];
            Loop<Dimensions>::run(
                [&](std::size_t i)
                {
                    if (bounds[i].min > location[i])
                    {
                        bounds[i].min = location[i];
                    }
                    if (bounds[i].max < location[i])
                    {
                        bounds[i].max = location[i];
                    }
                });
            m_nodes[nodeIndex].m_entries++;
        }

        Scalar pointRectDist(std::size_t nodeIndex, const point_t& location) const
        {
            auto clamp = [](Scalar v, Range r) { return std::max(r.min, std::min(r.max, v)); };
            const bounds_t& bounds = m_nodeBounds[nodeIndex];
            point_t closestBoundsPoint;

            Loop<Dimensions>::run([&](std::size_t i) { closestBoundsPoint[i] = clamp(location[i], bounds[i]); });
            return Distance::distance(closestBoundsPoint, location);
        }

        void searchCapacityLimitedBall(const point_t& location,
                                       Scalar maxRadius,
                                       std::size_t maxPoints,
                                       std::vector<std::size_t>& searchStack,
                                       std::priority_queue<DistancePayload, std::vector<DistancePayload>>& prioqueue,
                                       std::vector<DistancePayload>& results) const
        {
            std::size_t numSearchPoints = std::min(maxPoints, m_data.size());

            if (numSearchPoints > 0)
            {
                searchStack.push_back(0);
                while (searchStack.size() > 0)
                {
                    std::size_t nodeIndex = searchStack.back();
                    searchStack.pop_back();
                    const Node& node = m_nodes[nodeIndex];
                    Scalar minDist = pointRectDist(nodeIndex, location);
                    if (maxRadius > minDist
                        && (prioqueue.size() < numSearchPoints || prioqueue.top().distance > minDist))
                    {
                        if (node.m_splitDimension == Dimensions)
                        {
                            node.searchCapacityLimitedBall(m_data, location, maxRadius, numSearchPoints, prioqueue);
                        }
                        else
                        {
                            node.queueChildren(location, searchStack);
                        }
                    }
                }

                results.reserve(prioqueue.size());
                while (prioqueue.size() > 0)
                {
                    results.push_back(prioqueue.top());
                    prioqueue.pop();
                }
                std::reverse(results.begin(), results.end());
            }
        }

        bool split(std::size_t index)
        {
            if (m_nodes.capacity() < m_nodes.size() + 2)
            {
                m_nodes.reserve((m_nodes.capacity() + 1) * 2);
                m_nodeBounds.reserve((m_nodeBounds.capacity() + 1) * 2);
            }
            Node& splitNode = m_nodes[index];
            const bounds_t& splitBounds = m_nodeBounds[index];
            splitNode.m_splitDimension = Dimensions;
            Scalar width(0);
            // select widest dimension
            Loop<Dimensions>::run(
                [&](std::size_t i)
                {
                    Scalar dWidth = splitBounds[i].max - splitBounds[i].min;
                    if (dWidth > width)
                    {
                        splitNode.m_splitDimension = i;
                        width = dWidth;
                    }
                });
            if (splitNode.m_splitDimension == Dimensions)
            {
                return false;
            }

            std::vector<Scalar> splitDimVals;
            splitDimVals.reserve(splitNode.m_entries);
            for (std::size_t i = splitNode.m_dataBegin; i < splitNode.m_dataEnd; ++i)
            {
                splitDimVals.push_back(m_data[i].location[splitNode.m_splitDimension]);
            }
            std::nth_element(
                splitDimVals.begin(), splitDimVals.begin() + splitDimVals.size() / 2 + 1, splitDimVals.end());
            std::nth_element(splitDimVals.begin(),
                             splitDimVals.begin() + splitDimVals.size() / 2,
                             splitDimVals.begin() + splitDimVals.size() / 2 + 1);
            splitNode.m_splitValue
                = (splitDimVals[splitDimVals.size() / 2] + splitDimVals[splitDimVals.size() / 2 + 1]) / Scalar(2);

            auto it = std::partition(m_data.begin() + splitNode.m_dataBegin,
                                     m_data.begin() + splitNode.m_dataEnd,
                                     [&](const LocationPayload& lp)
                                     { return lp.location[splitNode.m_splitDimension] < splitNode.m_splitValue; });
            std::size_t mid = std::distance(m_data.begin(), it);

            splitNode.m_children = std::make_pair(m_nodes.size(), m_nodes.size() + 1);
            m_nodes.emplace_back(splitNode.m_dataBegin, mid);
            m_nodeBounds.emplace_back();
            m_nodeBounds.back().fill({std::numeric_limits<Scalar>::max(), std::numeric_limits<Scalar>::lowest()});

            m_nodes.emplace_back(mid, splitNode.m_dataEnd);
            m_nodeBounds.emplace_back();
            m_nodeBounds.back().fill({std::numeric_limits<Scalar>::max(), std::numeric_limits<Scalar>::lowest()});

            std::size_t leftIndex = splitNode.m_children.first;
            std::size_t rightIndex = splitNode.m_children.second;

            for (std::size_t i = m_nodes[leftIndex].m_dataBegin; i < m_nodes[leftIndex].m_dataEnd; ++i)
            {
                expandBounds(leftIndex, m_data[i].location);
            }
            for (std::size_t i = m_nodes[rightIndex].m_dataBegin; i < m_nodes[rightIndex].m_dataEnd; ++i)
            {
                expandBounds(rightIndex, m_data[i].location);
            }

            if (m_nodes[leftIndex].m_entries == 0) // points with equality to splitValue go in rightNode
            {
                splitNode.m_splitValue = 0;
                splitNode.m_splitDimension = Dimensions;
                splitNode.m_children = std::pair<std::size_t, std::size_t>(0, 0);
                m_nodes.pop_back();
                m_nodes.pop_back();
                m_nodeBounds.pop_back();
                m_nodeBounds.pop_back();
                return false;
            }
            return true;
        }

        struct Node
        {
            Node(std::size_t begin, std::size_t end)
            {
                m_dataBegin = begin;
                m_dataEnd = end;
            }

            bool shouldSplit() const { return m_entries >= BucketSize; }

            void searchCapacityLimitedBall(const std::vector<LocationPayload>& data,
                                           const point_t& location,
                                           Scalar maxRadius,
                                           std::size_t K,
                                           std::priority_queue<DistancePayload>& results) const
            {
                std::size_t i = m_dataBegin;

                // this fills up the queue if it isn't full yet
                for (; results.size() < K && i < m_dataEnd; i++)
                {
                    const auto& lp = data[i];
                    Scalar distance = Distance::distance(location, lp.location);
                    if (distance < maxRadius)
                    {
                        results.emplace(DistancePayload {distance, lp.payload});
                    }
                }

                // this adds new things to the queue once it is full
                for (; i < m_dataEnd; i++)
                {
                    const auto& lp = data[i];
                    Scalar distance = Distance::distance(location, lp.location);
                    if (distance < maxRadius && distance < results.top().distance)
                    {
                        results.pop();
                        results.emplace(DistancePayload {distance, lp.payload});
                    }
                }
            }

            void queueChildren(const point_t& location, std::vector<std::size_t>& searchStack) const
            {
                if (location[m_splitDimension] < m_splitValue)
                {
                    searchStack.push_back(m_children.second);
                    searchStack.push_back(m_children.first); // left is popped first
                }
                else
                {
                    searchStack.push_back(m_children.first);
                    searchStack.push_back(m_children.second); // right is popped first
                }
            }

            std::size_t m_entries = 0; /// size of the tree, or subtree

            std::size_t m_splitDimension = Dimensions; /// split dimension of this node
            Scalar m_splitValue = 0; /// split value of this node

            std::pair<std::size_t, std::size_t> m_children; /// subtrees of this node (if not a leaf)
            std::size_t m_dataBegin = 0; /// start index in the global data vector
            std::size_t m_dataEnd = 0; /// end index in the global data vector
            bool m_isWaitingForSplit = false; /// flag to avoid redundant entries in m_waitingForSplit
        };
    };
}
}
