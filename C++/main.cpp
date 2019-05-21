#include "KDTree.h"

#include <cstdlib>
#include <ctime>
#include <iostream>

double drand() { return (rand() / (RAND_MAX + 1.)); }
void example();
void accuracyTest();
void performanceTest();

int main()
{
    example();
    accuracyTest();
    performanceTest();
    return 0;
}

void example()
{
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
}

void accuracyTest()
{
    static const int dims = 4;
    std::vector<std::array<double, dims>> points;
    using tree_t = jk::tree::KDTree<int, dims>;
    tree_t tree;
    int count = 0;
    std::srand(1234567);

    auto randomPoint = []() {
        std::array<double, dims> loc;
        for (std::size_t j = 0; j < dims; j++)
        {
            loc[j] = drand();
        }
        return loc;
    };

    for (int i = 0; i < 2000; i++)
    {
        std::array<double, dims> loc = randomPoint();
        tree.addPoint(loc, count++);

        points.push_back(loc);
    }
    tree.splitOutstanding();

    auto bruteforceKNN = [&](const std::array<double, dims>& searchLoc, int K) -> std::vector<std::pair<double, int>> {
        std::vector<std::pair<double, int>> dists;
        for (std::size_t i = 0; i < points.size(); i++)
        {
            double distance = tree_t::distance_t::distance(searchLoc, points[i]);
            dists.emplace_back(distance, i);
        }
        std::partial_sort(dists.begin(), dists.begin() + K, dists.end());
        dists.resize(K);
        return dists;
    };

    for (std::size_t j = 0; j < points.size(); j++)
    {
        const std::array<double, dims> loc = randomPoint();
        const std::size_t k = 50;

        auto tnn = tree.searchKnn(loc, k);
        auto bnn = bruteforceKNN(loc, k);
        if (tnn.size() != k)
        {
            std::cout << "Searched for " << k << ", found " << tnn.size() << std::endl;
        }
        for (std::size_t i = 0; i < k; i++)
        {
            if (std::abs(bnn[i].first - tnn[i].distance) > 1e-10)
            {
                std::cout << "distances not equal" << std::endl;
            }
            if (bnn[i].second != tnn[i].payload)
            {
                std::cout << "payloads not equal" << std::endl;
            }
        }
    }

    auto bruteforceRadius
        = [&](const std::array<double, dims>& searchLoc, double radius) -> std::vector<std::pair<double, int>> {
        std::vector<std::pair<double, int>> dists;
        for (std::size_t i = 0; i < points.size(); i++)
        {
            double distance = tree_t::distance_t::distance(searchLoc, points[i]);
            dists.emplace_back(distance, i);
        }
        std::sort(dists.begin(), dists.end());
        auto iter = std::lower_bound(dists.begin(), dists.end(), std::make_pair(radius, int(0)));
        std::size_t inliers = std::distance(dists.begin(), iter);
        dists.resize(inliers);
        return dists;
    };

    for (std::size_t j = 0; j < points.size(); j++)
    {
        const std::array<double, dims> loc{{drand(), drand(), drand(), drand()}};
        const double radius = 0.7;

        auto tnn = tree.searchBall(loc, radius);
        auto bnn = bruteforceRadius(loc, radius);
        if (tnn.size() != bnn.size())
        {
            std::cout << "Brute force results are not the same size as tree results" << std::endl;
            continue;
        }

        if (tnn.size() && tnn.back().distance > radius)
        {
            std::cout << "Searched for max radius " << radius << ", found " << tnn.back().distance << std::endl;
        }
        for (std::size_t i = 0; i < tnn.size(); i++)
        {
            if (std::abs(bnn[i].first - tnn[i].distance) > 1e-10)
            {
                std::cout << "distances not equal" << std::endl;
            }
            if (bnn[i].second != tnn[i].payload)
            {
                std::cout << "payloads not equal" << std::endl;
            }
        }
    }
}

#define DURATION double(((previous = current) * 0 + (current = std::clock()) - previous) / double(CLOCKS_PER_SEC))
void performanceTest()
{
    std::clock_t previous = std::clock(), current = previous;

    static const int dims = 2;
    std::cout << "adding ";
    std::vector<std::array<double, dims>> points;
    jk::tree::KDTree<int, dims, 8> tree;

    int count = 0;
    std::srand(1234567);

    auto randomPoint = []() {
        std::array<double, dims> loc;
        for (std::size_t j = 0; j < dims; j++)
        {
            loc[j] = drand();
        }
        return loc;
    };

    for (int i = 0; i < 4 * 1000 * 10; i++)
    {
        std::array<double, dims> loc = randomPoint();
        tree.addPoint(loc, count++, false);

        points.push_back(loc);
    }

    std::vector<std::array<double, dims>> searchPoints;
    for (int i = 0; i < 5 * 1000 * 1000; i++)
    {
        searchPoints.push_back(randomPoint());
    }

    std::cout << DURATION << "s" << std::endl;
    std::cout << "splitting ";
    tree.splitOutstanding();
    std::cout << DURATION << "s" << std::endl;
    for (int j = 0; j < 3; j++)
    {
        std::cout << "searching " << (j + 1) << " ";

        for (auto p : searchPoints)
        {
            const int k = 3;
            auto nn = tree.searchKnn(p, k);

            if (nn.size() != k)
            {
                std::cout << nn.size() << " instead of " << k << " ERROR" << std::endl;
            }
        }
        std::cout << DURATION << "s" << std::endl;
    }
    for (int j = 0; j < 3; j++)
    {
        std::cout << "bulk searching " << (j + 1) << " ";

        const int k = 3;
        auto searcher = tree.searcher();
        for (auto p : searchPoints)
        {
            const auto& nn = searcher.search(p, std::numeric_limits<double>::max(), k);

            if (nn.size() != k)
            {
                std::cout << nn.size() << " instead of " << k << " ERROR" << std::endl;
            }
        }
        std::cout << DURATION << "s" << std::endl;
    }
}
