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
 * jk.tree.KDTree<std::string, 2> tree;
 * tree.addPoint(std::array<double,2>{{1,2}}, "George");
 * tree.addPoint(std::array<double,2>{{1,3}}, "Harold");
 * tree.addPoint(std::array<double,2>{{7,7}}, "Melvin");
 *
 * std::array<double,2> monsterLocation{{6,6}};
 * const std::size_t monsterHeads = 2;
 * auto nearestTwo = tree.searchKnn(monster, monsterHeads);
 * for (const auto& victim : nearestTwo)
 * {
 *     std::cout << victim.payload << " was eaten by the monster!" << std::endl;
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
 */

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <queue>
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
        Node m_root;

    public:
        using metric_t = Metric;
        using scalar_t = Scalar;
        using payload_t = Payload;

        KDTree() {}

        void addPoint(const std::array<Scalar, Dimensions>& location, const Payload& payload, bool autosplit = true)
        {
            Node* addNode = &m_root;

            while (addNode->m_children != nullptr)
            {
                addNode->expandBounds(location);
                if (location[addNode->m_splitDimension] < addNode->m_splitValue)
                {
                    addNode = &(addNode->m_children->first);
                }
                else
                {
                    addNode = &(addNode->m_children->second);
                }
            }
            addNode->add(location, payload);

            if (autosplit && addNode->shouldSplit() && addNode->m_entries % BucketSize == 0)
            {
                addNode->split();
            }
        }

        void splitOutstanding()
        {
            std::deque<Node*> searchStack;
            searchStack.push_back(&m_root);
            while (searchStack.size() > 0)
            {
                Node* node = searchStack.back();
                searchStack.pop_back();
                if (node->m_children == nullptr)
                {
                    if (!node->shouldSplit())
                    {
                        continue;
                    }

                    if (!node->split())
                    {
                        continue;
                    }
                }

                searchStack.push_front(&(node->m_children->first));
                searchStack.push_front(&(node->m_children->second));
            }
        }

        struct DistancePayload
        {
            Scalar distance;
            Payload payload;
            bool operator<(const DistancePayload& dp) const { return distance < dp.distance; }
        };

        std::vector<DistancePayload> searchKnn(
            const std::array<Scalar, Dimensions>& location, std::size_t numNeighbours) const
        {

            using VecDistPay = std::vector<DistancePayload>;
            if (m_root.m_entries < numNeighbours)
            {
                numNeighbours = m_root.m_entries;
            }
            VecDistPay returnResults;
            if (numNeighbours > 0)
            {
                VecDistPay container;
                container.reserve(numNeighbours);
                std::priority_queue<DistancePayload, VecDistPay> results(
                    std::less<DistancePayload>(), std::move(container));
                std::vector<const Node*> searchStack;

                searchStack.push_back(&m_root);
                while (searchStack.size() > 0)
                {
                    const Node* const node = searchStack.back();
                    searchStack.pop_back();
                    if (results.size() < numNeighbours || results.top().distance > node->pointRectDist(location))
                    {
                        if (node->m_children == nullptr)
                        {
                            node->searchBucket(location, numNeighbours, results);
                        }
                        else
                        {
                            node->addChildren(location, searchStack);
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

    private:
        struct LocationPayload
        {
            std::array<Scalar, Dimensions> location;
            Payload payload;
        };

        struct Node
        {
            Node()
            {
                for (std::size_t i = 0; i < Dimensions; i++)
                {
                    m_bounds[i][0] = std::numeric_limits<Scalar>::infinity();
                    m_bounds[i][1] = -std::numeric_limits<Scalar>::infinity();
                }
                m_locationPayloads.reserve(BucketSize);
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

            void add(const std::array<Scalar, Dimensions>& location, const Payload& payload)
            {
                m_locationPayloads.push_back(LocationPayload{location, payload});
                expandBounds(location);
            }

            bool shouldSplit() const { return m_entries >= BucketSize; }

            bool split()
            {
                m_splitDimension = Dimensions;
                Scalar width(0);
                // select widest dimension
                for (std::size_t i = 0; i < Dimensions; i++)
                {
                    Scalar dWidth = m_bounds[i][1] - m_bounds[i][0];
                    if (dWidth > width)
                    {
                        m_splitDimension = i;
                        width = dWidth;
                    }
                }
                if (m_splitDimension == Dimensions)
                {
                    return false;
                }
                m_splitValue = (m_bounds[m_splitDimension][0] + m_bounds[m_splitDimension][1]) / Scalar(2);

                m_children.reset(new std::pair<Node, Node>());

                for (const auto& lp : m_locationPayloads)
                {
                    if (lp.location[m_splitDimension] < m_splitValue)
                    {
                        m_children->first.add(lp.location, lp.payload);
                    }
                    else
                    {
                        m_children->second.add(lp.location, lp.payload);
                    }
                }

                if (m_children->first.m_entries == 0 || m_children->second.m_entries == 0)
                {
                    m_splitValue = 0;
                    m_splitDimension = Dimensions;
                    m_children.reset();
                    return false;
                }
                else
                {
                    m_locationPayloads.clear();
                    m_locationPayloads.shrink_to_fit();
                    return true;
                }
            }

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
                const std::array<Scalar, Dimensions>& location, std::vector<const Node*>& searchStack) const
            {
                if (location[m_splitDimension] < m_splitValue)
                {
                    searchStack.push_back(&(m_children->second));
                    searchStack.push_back(&(m_children->first)); // left is popped first
                }
                else
                {
                    searchStack.push_back(&(m_children->first));
                    searchStack.push_back(&(m_children->second)); // right is popped first
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

            std::vector<LocationPayload> m_locationPayloads; /// data held in this node (if a leaf)
            std::unique_ptr<std::pair<Node, Node>> m_children; /// subtrees held in this node (if not a leaf)
        };
    };
}
}
