#pragma once

#include <set>

using namespace std;

namespace similarity {

    //http://stackoverflow.com/questions/1060648/fast-intersection-of-sets-c-vs-c
    template <typename T, typename OutIter> inline void stl_intersect(const T& set1, const T& set2, OutIter out) {
        std::set_intersection(set1.begin(), set1.end(), set2.begin(), set2.end(), out);
    }

    template<typename T> inline double tanimoto(const set<T>* set1, const set<T>* set2) {
        vector<T> interset;

        set_intersection(set1->begin(),set1->end(),
                set2->begin(),set2->end(),
                back_inserter(interset));

        if (interset.empty()) {
            return 0.0;
        }

        return (double(interset.size()))/((set1->size())+(set2->size())-interset.size());
    }

    template<typename T> inline double tanimoto(const vector<T>* set1, const vector<T>* set2) {
        vector<T> interset;

        set_intersection(set1->begin(),set1->end(),
                set2->begin(),set2->end(),
                back_inserter(interset));

        if (interset.empty()) {
            return 0.0;
        }

        return (double(interset.size()))/((set1->size())+(set2->size())-interset.size());
    }


    // _Rb_treeは要素数を覚えているので意味なさげ
//    template<typename T> inline double tanimoto(const set<T>* set1, const set<T>* set2, const size_t size1) {
//        set<T> interset;

//        set_intersection(set1->begin(),set1->end(),
//                set2->begin(),set2->end(),
//                inserter(interset, interset.begin()));

//        if (interset.empty()) {
//            return 0.0;
//        }

//        return (double(interset.size()))/(size1+(set2->size())-interset.size());
//    }

//    template<typename T> inline double tanimoto(const set<T>* set1, const set<T>* set2, const size_t size1, const size_t size2) {
//        set<T> interset;

//        set_intersection(set1->begin(),set1->end(),
//                set2->begin(),set2->end(),
//                inserter(interset, interset.begin()));

//        if (interset.empty()) {
//            return 0.0;
//        }

//        return (double(interset.size()))/(size1 + size2 - interset.size());
//    }


    template<typename T> inline double tanimoto(const set<T>& set1, const set<T>& set2) {
        set<T> interset;

        set_intersection(set1.begin(),set1.end(),
                set2.begin(),set2.end(),
                inserter(interset, interset.begin()));

        if (interset.empty()) {
            return 0.0;
        }

        return (double(interset.size()))/((set1.size())+(set2.size())-interset.size());
    }
}
