#pragma once

/**
 * KDTree.h by Julian Kent
 * A single-file, header-only, high performance C++ KD-Tree, allowing
 * dynamic insertions, with a simple API, depending only on the STL.
 *
 * -------------------------------------------------------------------
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * For additional licensing rights, feature requests or questions,
 * please contact Julian Kent <jkflying@gmail.com>
 *
 * -------------------------------------------------------------------
 *
 * Example usage:
 *
 * using tree_t = jk::tree::KDTree<std::string, 2>;
 * using point_t = std::array<double,2>;
 * tree_t tree;
 * tree.addPoint(point_t{{1,2}}, "George");
 * tree.addPoint(point_t{{1,3}}, "Harold");
 * tree.addPoint(point_t{{7,7}}, "Melvin");
 *
 * point_t monsterLocation{{6,6}};
 * const std::size_t monsterHeads = 2; // this monster can eat two people at once
 * auto victims = tree.searchKnn(monsterLocation, monsterHeads); // find the nearest two unlucky people
 * for (const auto& victim : victims)
 * {
 *     std::cout << victim.payload << " eaten by monster, with distance " << victim.distance << "!" << std::endl;
 * }
 *
 * -------------------------------------------------------------------
 *
 * Tuning tips:
 *
 * If you need to add a lot of points before doing any queries, set the optional `autosplit` parameter to false,
 * then call splitOutstanding(). This will reduce temporaries and result in a better balanced tree.
 *
 * Set the bucket size to be roughly twice the K in a KNN query. If you have more dimensions, it is better to have a
 * larger bucket size. 32 is a good starting point.
 *
 * If you experience linear search performance, check that you don't have a bunch of duplicate point locations. This
 * will result in the tree being unable to split the bucket the points are in, degrading search performance.
 *
 * The tree adapts to the parallel-to-axis dimensionality of the problem. Thus, if there is one dimension with a much
 * larger scale than the others, most of the splitting will happen on this dimension. This is achieved by trying to
 * keep the bounding boxes of the buckets equal lengths in all axes. Note, this does not imply that the subspace a
 * subtree is parent of is square, for example if the data distribution in it is unequal.
 *
 * Random data performs worse than 'real world' data with structure. This is because real world data has tighter
 * bounding boxes, meaning more branches of the tree can be eliminated sooner.
 */

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <queue>
#include <set>
#include <vector>

namespace jk
{
namespace tree
{
    struct L1
    {
        template <std::size_t Dimensions, typename Scalar>
        static inline Scalar distance(
            const std::array<Scalar, Dimensions>& location1, const std::array<Scalar, Dimensions>& location2)
        {
            auto abs = [](Scalar v) { return v > 0 ? v : -v; };
            Scalar dist = 0;
            for (std::size_t i = 0; i < Dimensions; i++)
            {
                dist += abs(location1[i] - location2[i]);
            }
            return dist;
        }
    };

    struct L2
    {
        template <std::size_t Dimensions, typename Scalar>
        static inline Scalar distance(
            const std::array<Scalar, Dimensions>& location1, const std::array<Scalar, Dimensions>& location2)
        {
            auto sqr = [](Scalar v) { return v * v; };
            Scalar dist = 0;
            for (std::size_t i = 0; i < Dimensions; i++)
            {
                dist += sqr(location1[i] - location2[i]);
            }
            return dist;
        }
    };

    template <class Payload, std::size_t Dimensions, std::size_t BucketSize = 32, class Metric = L2,
        typename Scalar = double>
    class KDTree
    {
    private:
        struct Node;
        std::vector<Node> m_nodes;
        std::set<std::size_t> waitingForSplit;

    public:
        using metric_t = Metric;
        using scalar_t = Scalar;
        using payload_t = Payload;
        static const std::size_t dimensions = Dimensions;
        static const std::size_t bucketSize = BucketSize;

        KDTree() { m_nodes.emplace_back(BucketSize); } // initialize the root node

        void addPoint(const std::array<Scalar, Dimensions>& location, const Payload& payload, bool autosplit = true)
        {
            std::size_t addNode = 0;

            while (m_nodes[addNode].m_splitDimension != Dimensions)
            {
                m_nodes[addNode].expandBounds(location);
                if (location[m_nodes[addNode].m_splitDimension] < m_nodes[addNode].m_splitValue)
                {
                    addNode = m_nodes[addNode].m_children.first;
                }
                else
                {
                    addNode = m_nodes[addNode].m_children.second;
                }
            }
            m_nodes[addNode].add(LocationPayload{location, payload});

            if (m_nodes[addNode].shouldSplit() && m_nodes[addNode].m_entries % BucketSize == 0)
            {
                if (autosplit)
                {
                    split(addNode);
                }
                else
                {
                    waitingForSplit.insert(addNode);
                }
            }
        }

        void splitOutstanding()
        {
            std::vector<std::size_t> searchStack(waitingForSplit.begin(), waitingForSplit.end());
            while (searchStack.size() > 0)
            {
                std::size_t addNode = searchStack.back();
                searchStack.pop_back();
                if (m_nodes[addNode].m_splitDimension == Dimensions)
                {
                    if (!m_nodes[addNode].shouldSplit())
                    {
                        continue;
                    }

                    if (!split(addNode))
                    {
                        continue;
                    }
                }
                searchStack.push_back(m_nodes[addNode].m_children.first);
                searchStack.push_back(m_nodes[addNode].m_children.second);
            }
            waitingForSplit.clear();
        }

        struct DistancePayload
        {
            Scalar distance;
            Payload payload;
            bool operator<(const DistancePayload& dp) const { return distance < dp.distance; }
        };

        std::vector<DistancePayload> searchK(
            const std::array<Scalar, Dimensions>& location, std::size_t numNeighbours) const
        {

            using VecDistPay = std::vector<DistancePayload>;
            if (m_nodes[0].m_entries < numNeighbours)
            {
                numNeighbours = m_nodes[0].m_entries;
            }
            VecDistPay returnResults;
            if (numNeighbours > 0)
            {
                VecDistPay container;
                container.reserve(numNeighbours);
                std::priority_queue<DistancePayload, VecDistPay> results(
                    std::less<DistancePayload>(), std::move(container));
                std::vector<std::size_t> searchStack;
                searchStack.reserve(1 + std::size_t(1.5 * std::log2(1 + m_nodes[0].m_entries / BucketSize)));
                searchStack.push_back(0);
                while (searchStack.size() > 0)
                {
                    std::size_t nodeIndex = searchStack.back();
                    searchStack.pop_back();
                    const Node& node = m_nodes[nodeIndex];
                    if (results.size() < numNeighbours || results.top().distance > node.pointRectDist(location))
                    {
                        if (node.m_splitDimension == Dimensions)
                        {
                            node.searchBucket(location, numNeighbours, results);
                        }
                        else
                        {
                            node.addChildren(location, searchStack);
                        }
                    }
                }

                returnResults.reserve(results.size());
                while (results.size() > 0)
                {
                    returnResults.push_back(results.top());
                    results.pop();
                }
                std::reverse(returnResults.begin(), returnResults.end());
            }
            return returnResults;
        }

        DistancePayload search(const std::array<Scalar, Dimensions>& location) const
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
                    if (result.distance > node.pointRectDist(location))
                    {
                        if (node.m_splitDimension == Dimensions)
                        {
                            for (const auto& lp : node.m_locationPayloads)
                            {
                                Scalar nodeDist = Metric::distance(location, lp.location);
                                if (nodeDist < result.distance)
                                {
                                    result = DistancePayload{nodeDist, lp.payload};
                                }
                            }
                        }
                        else
                        {
                            node.addChildren(location, searchStack);
                        }
                    }
                }
            }
            return result;
        }

    private:
        struct LocationPayload
        {
            std::array<Scalar, Dimensions> location;
            Payload payload;
        };

        std::vector<LocationPayload> m_bucketRecycle;

        bool split(std::size_t index)
        {
            Node* splitNode = &m_nodes[index];
            splitNode->m_splitDimension = Dimensions;
            Scalar width(0);
            // select widest dimension
            for (std::size_t i = 0; i < Dimensions; i++)
            {
                auto diff = [](std::array<Scalar, 2> vals) { return vals[1] - vals[0]; };
                Scalar dWidth = diff(splitNode->m_bounds[i]);
                if (dWidth > width)
                {
                    splitNode->m_splitDimension = i;
                    width = dWidth;
                }
            }
            if (splitNode->m_splitDimension == Dimensions)
            {
                return false;
            }
            auto avg = [](std::array<Scalar, 2> vals) { return (vals[0] + vals[1]) / Scalar(2); };
            splitNode->m_splitValue = avg(splitNode->m_bounds[splitNode->m_splitDimension]);

            splitNode->m_children = std::pair<std::size_t, std::size_t>(m_nodes.size(), m_nodes.size() + 1);
            std::size_t entries = splitNode->m_entries;
            m_nodes.emplace_back(m_bucketRecycle, entries);
            m_nodes.emplace_back(entries);
            splitNode = &m_nodes[index]; // in case the vector resized
            Node* leftNode = &m_nodes[splitNode->m_children.first];
            Node* rightNode = &m_nodes[splitNode->m_children.second];

            for (const auto& lp : splitNode->m_locationPayloads)
            {
                if (lp.location[splitNode->m_splitDimension] < splitNode->m_splitValue)
                {
                    leftNode->add(lp);
                }
                else
                {
                    rightNode->add(lp);
                }
            }

            if (leftNode->m_entries == 0 || rightNode->m_entries == 0)
            {
                splitNode->m_splitValue = 0;
                splitNode->m_splitDimension = Dimensions;
                splitNode->m_children = std::pair<std::size_t, std::size_t>(0, 0);
                std::swap(leftNode->m_locationPayloads, m_bucketRecycle);
                m_nodes.pop_back();
                m_nodes.pop_back();
                return false;
            }
            else
            {
                splitNode->m_locationPayloads.clear();
                if (splitNode->m_locationPayloads.capacity() == BucketSize)
                {
                    std::swap(splitNode->m_locationPayloads, m_bucketRecycle);
                }
                else
                {
                    std::vector<LocationPayload> empty;
                    std::swap(splitNode->m_locationPayloads, empty);
                }
                return true;
            }
        }

        struct Node
        {
            Node(std::size_t capacity) { init(capacity); }

            Node(std::vector<LocationPayload>& recycle, std::size_t capacity)
            {
                std::swap(m_locationPayloads, recycle);
                init(capacity);
            }

            void init(std::size_t capacity)
            {
                m_bounds.fill({{std::numeric_limits<Scalar>::infinity(), -std::numeric_limits<Scalar>::infinity()}});
                m_locationPayloads.reserve(std::max(BucketSize, capacity));
            }

            void expandBounds(const std::array<Scalar, Dimensions>& location)
            {
                for (std::size_t i = 0; i < Dimensions; i++)
                {
                    if (m_bounds[i][0] > location[i])
                    {
                        m_bounds[i][0] = location[i];
                    }
                    if (m_bounds[i][1] < location[i])
                    {
                        m_bounds[i][1] = location[i];
                    }
                }
                m_entries++;
            }

            void add(const LocationPayload& lp)
            {
                expandBounds(lp.location);
                m_locationPayloads.push_back(lp);
            }

            bool shouldSplit() const { return m_entries >= BucketSize; }

            void searchBucket(const std::array<Scalar, Dimensions>& location, std::size_t K,
                std::priority_queue<DistancePayload>& results) const
            {
                std::size_t i = 0;

                // this fills up the queue if it isn't full yet
                for (std::size_t max_i = K - results.size(); i < max_i && i < m_entries; i++)
                {
                    const auto& lp = m_locationPayloads[i];
                    Scalar distance = Metric::distance(location, lp.location);
                    results.emplace(DistancePayload{distance, lp.payload});
                }

                // this adds new things to the queue once it is full
                for (; i < m_entries; i++)
                {
                    const auto& lp = m_locationPayloads[i];
                    Scalar distance = Metric::distance(location, lp.location);
                    if (distance < results.top().distance)
                    {
                        results.pop();
                        results.emplace(DistancePayload{distance, lp.payload});
                    }
                }
            }

            void addChildren(
                const std::array<Scalar, Dimensions>& location, std::vector<std::size_t>& searchStack) const
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

            Scalar pointRectDist(const std::array<Scalar, Dimensions>& location) const
            {
                std::array<Scalar, Dimensions> closestBoundsPoint;

                for (std::size_t i = 0; i < Dimensions; i++)
                {
                    if (m_bounds[i][0] > location[i])
                    {
                        closestBoundsPoint[i] = m_bounds[i][0];
                    }
                    else if (m_bounds[i][1] < location[i])
                    {
                        closestBoundsPoint[i] = m_bounds[i][1];
                    }
                    else
                    {
                        closestBoundsPoint[i] = location[i];
                    }
                }
                return Metric::distance(closestBoundsPoint, location);
            }

            std::size_t m_entries = 0; /// size of the tree, or subtree

            std::size_t m_splitDimension = Dimensions; /// split dimension of this node
            Scalar m_splitValue = 0; /// split value of this node

            std::array<std::array<Scalar, 2>, Dimensions> m_bounds; /// bounding box of this node

            std::pair<std::size_t, std::size_t> m_children; /// subtrees of this node (if not a leaf)
            std::vector<LocationPayload> m_locationPayloads; /// data held in this node (if a leaf)
        };
    };
}
}
