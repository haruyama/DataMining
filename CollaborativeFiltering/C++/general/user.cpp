#include <iostream>
#include <set>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/ptr_container/ptr_map.hpp>
#include <boost/multi_array.hpp>
#include <boost/bind.hpp>
#include <boost/foreach.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>
#include <boost/program_options.hpp>
#include <cstdlib>

#include "csv.hpp"
#include "cf.hpp"

using namespace std;
using namespace boost;
using namespace boost::program_options;

static const size_t MAX_RELATED =10;
typedef long item_type;

struct str2item_type {
    item_type operator()(string s) {
        return atol(s.c_str());
    }
};

int main(int argc, char* argv[]) {

    options_description opt("options");

    opt.add_options()
        ("help,h"                                   , "help")
        ("r"      , value<size_t>()->default_value(10) , "max_related_items")
        ("cache,c"                                  , "enable cache");

    variables_map vm;
    store(parse_command_line(argc, argv, opt), vm);
    notify(vm);


    if (vm.count("help")) {
        cout << opt << endl;
        return 1;
    }


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

    cf::top_matches(users, vm["r"].as<size_t>(), vm.count("cache"));

    return 0;
}
