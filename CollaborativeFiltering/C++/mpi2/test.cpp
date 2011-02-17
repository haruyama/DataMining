#include <boost/mpi.hpp>
#include <iostream>
#include <boost/serialization/set.hpp>
#include <set>
#include <boost/foreach.hpp>

using namespace std;
using namespace boost;

int main(int argc, char* argv[])
{
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;

    if (world.rank() == 0) {
        set<int>* s(new set<int>());
        s->insert(1);
        s->insert(3);
        world.send(1, 0, s);
    } else {

        set<int>* s;
        world.recv(0, 0, s);

        BOOST_FOREACH(int e, *s) {
            std::cout << e << endl;
        }
    }

    return 0;
}

