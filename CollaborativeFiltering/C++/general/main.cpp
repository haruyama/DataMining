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

    ptr_vector<vector<item_type> > users;
    ptr_map<item_type, set<size_t> > items;

    /* read data */
    string line;
    while(getline(cin,line)) {
        vector<string> row = csv::parse_line(line);
        vector<item_type>* item_vector(new vector<item_type>);
        transform(row.begin() + 1, row.end(), inserter(*item_vector, item_vector->begin()), str2item_type());
        item_vector->erase(remove(item_vector->begin(), item_vector->end(), 0), item_vector->end());
        users.push_back(item_vector);
    }

    cf::transform_prefs(users, items);

    cf::top_matches(items, MAX_RELATED);
    return 0;
}
