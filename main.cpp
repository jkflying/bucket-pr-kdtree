#include "KDTree.h"

#include <cstdlib>
#include <ctime>
#include <iostream>

double drand() { return (rand() / (RAND_MAX + 1.)); }
void accuracyTest();
void performanceTest();

int main()
{
    accuracyTest();
    performanceTest();
    return 0;
}

void accuracyTest()
{
    std::vector<std::array<double, 4>> points;
    using tree_t = jk::tree::KDTree<int, 4>;
    tree_t tree;
    int count = 0;
    std::srand(1234567);

    for (int i = 0; i < 2000; i++)
    {
        std::array<double, 4> loc{{drand(), drand(), drand(), drand()}};
        tree.addPoint(loc, count++);

        points.push_back(loc);
    }
    tree.splitOutstanding();

    auto bruteforce = [&](const std::array<double, 4>& searchLoc, int K) -> std::vector<std::pair<double, int>> {
        std::vector<std::pair<double, int>> dists;
        for (int i = 0; i < points.size(); i++)
        {
            double distance = tree_t::metric_t::distance(searchLoc, points[i]);
            dists.emplace_back(distance, i);
        }
        std::partial_sort(dists.begin(), dists.begin() + K, dists.end());
        dists.resize(K);
        return dists;
    };

    for (int i = 0; i < points.size(); i++)
    {
        std::array<double, 4> loc{{drand(), drand(), drand(), drand()}};
        int k = 50;

        auto tnn = tree.searchK(loc, k);
        auto bnn = bruteforce(loc, k);
        for (int i = 0; i < k; i++)
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

    const std::size_t dims = 4;
    std::cout << "adding ";
    std::vector<std::array<double, dims>> points;
    jk::tree::KDTree<int, dims> tree;

    int count = 0;
    std::srand(1234567);

    for (int i = 0; i < 10 * 1000 * 1000; i++)
    {
        std::array<double, dims> loc;
        for (int j = 0; j < dims; j++)
        {
            loc[j] = drand();
        }
        tree.addPoint(loc, count++, false);

        points.push_back(loc);
    }
    std::cout << DURATION << "s" << std::endl;
    std::cout << "splitting ";
    tree.splitOutstanding();
    std::cout << DURATION << "s" << std::endl;
    std::cout << "searching ";

    //     for (int j = 0; j < 500000; j++)
    for (int i = 0; i < 500 * 1000; i++)
    {
        const int k = 5;
        auto nn = tree.searchK(points[i], k);

        if (nn[0].payload != i)
        {
            std::cout << nn[0].distance << " ERROR" << std::endl;
        }
        if (nn.size() != k)
        {
            std::cout << nn.size() << " instead of " << k << " ERROR" << std::endl;
        }
    }
    std::cout << DURATION << "s" << std::endl;
}
