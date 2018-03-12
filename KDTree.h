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
        using tree_t = KDTree<Payload, Dimensions, BucketSize, Metric, Scalar>;

    public:
        using metric_t = Metric;
        using scalar_t = Scalar;
        using payload_t = Payload;

        KDTree()
        {
            for (std::size_t i = 0; i < Dimensions; i++)
            {
                m_bounds[i][0] = std::numeric_limits<Scalar>::infinity();
                m_bounds[i][1] = -std::numeric_limits<Scalar>::infinity();
            }
            m_locationPayloads.reserve(BucketSize);
        }

        void addPoint(const std::array<Scalar, Dimensions>& location, const Payload& payload, bool autosplit = true)
        {
            tree_t* addNode = this;

            while (addNode->m_splitDimension != Dimensions)
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
            std::deque<tree_t*> searchStack;
            searchStack.push_back(this);
            while (searchStack.size() > 0)
            {
                tree_t* node = searchStack.back();
                searchStack.pop_back();
                if (node->m_splitDimension == Dimensions)
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
            if (m_entries < numNeighbours)
            {
                numNeighbours = m_entries;
            }
            VecDistPay returnResults;
            if (numNeighbours > 0)
            {
                VecDistPay container;
                container.reserve(numNeighbours);
                std::priority_queue<DistancePayload, VecDistPay> results(
                    std::less<DistancePayload>(), std::move(container));
                std::vector<const tree_t*> searchStack;

                searchStack.push_back(this);
                while (searchStack.size() > 0)
                {
                    const tree_t* const node = searchStack.back();
                    searchStack.pop_back();
                    if (results.size() < numNeighbours
                        || results.top().distance > pointRectDist(node->m_bounds, location))
                    {
                        if (node->m_splitDimension == Dimensions)
                        {
                            node->searchK(location, results, numNeighbours);
                        }
                        else
                        {
                            node->search(location, searchStack);
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
            Scalar width = 0.0;
            // select widest dimension
            for (std::size_t i = 0; i < Dimensions; i++)
            {
                Scalar dWidth = m_bounds[i][1] - m_bounds[i][0];
                if (dWidth > width)
                {
                    m_splitDimension = static_cast<int64_t>(i);
                    width = dWidth;
                }
            }
            if (m_splitDimension == Dimensions)
            {
                return false;
            }
            m_splitValue = (m_bounds[m_splitDimension][0] + m_bounds[m_splitDimension][1]) / Scalar(2);

            m_children.reset(new std::pair<tree_t, tree_t>());

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

        void searchK(const std::array<Scalar, Dimensions>& location, std::priority_queue<DistancePayload>& results,
            std::size_t K) const
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

        void search(const std::array<Scalar, Dimensions>& location, std::vector<const tree_t*>& searchStack) const
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

        static Scalar pointRectDist(
            const std::array<std::array<Scalar, 2>, Dimensions>& bounds, const std::array<Scalar, Dimensions>& location)
        {
            std::array<Scalar, Dimensions> closestBoundsPoint;

            for (std::size_t i = 0; i < Dimensions; i++)
            {
                if (bounds[i][0] > location[i])
                {
                    closestBoundsPoint[i] = bounds[i][0];
                }
                else if (bounds[i][1] < location[i])
                {
                    closestBoundsPoint[i] = bounds[i][1];
                }
                else
                {
                    closestBoundsPoint[i] = location[i];
                }
            }
            return Metric::distance(closestBoundsPoint, location);
        }

        std::size_t m_entries = 0; /// size of the tree, or subtreebuck

        std::size_t m_splitDimension = Dimensions; /// split dimension of this node
        Scalar m_splitValue = 0; /// split value of this node

        std::array<std::array<Scalar, 2>, Dimensions> m_bounds; /// bounding box of this node

        std::vector<LocationPayload> m_locationPayloads; /// data held in this node (if a leaf)
        std::unique_ptr<std::pair<tree_t, tree_t>> m_children; /// subtrees held in this node (if not a leaf)
    };
}
}
