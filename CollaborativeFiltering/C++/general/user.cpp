#include <iostream>
#include <set>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/ptr_container/ptr_map.hpp>
#include <boost/multi_array.hpp>
#include <boost/bind.hpp>
#include <boost/foreach.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>
#include <cstdlib>

#include "csv.hpp"
#include "cf.hpp"

using namespace std;
using namespace boost;

static const size_t MAX_RELATED =10;
typedef long item_type;

struct str2item_type {
    item_type operator()(string s) {
        return atol(s.c_str());
    }
};

int main() {

    //ptr_vector<vector<item_type> > users;
    ptr_map<string, set<item_type> > users;

    /* read data */
    string line;
    while(getline(cin,line)) {
        vector<string> row = csv::parse_line(line);
        set<item_type>* item_set(new set<item_type>);
        transform(row.begin() + 1, row.end(), inserter(*item_set, item_set->begin()), str2item_type());
        item_set->erase(0);
        users.insert(row[0], item_set);
    }

    cf::top_matches(users, MAX_RELATED);
    return 0;
}
