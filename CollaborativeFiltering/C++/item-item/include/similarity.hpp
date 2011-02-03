#include <set>

using namespace std;

namespace similarity {

    template<typename T> inline double tanimoto(const set<T>* set1, const set<T>* set2) {
        set<T> interset;

        set_intersection(set1->begin(),set1->end(),
                set2->begin(),set2->end(),
                inserter(interset, interset.begin()));

        return (double(interset.size()))/((set1->size())+(set2->size())-interset.size());
    }

}
