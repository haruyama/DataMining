#include <set>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/ptr_container/ptr_map.hpp>
#include <boost/program_options.hpp>
#include <cstdlib>

#include "csv.hpp"
#include "cf.hpp"

using namespace std;
using namespace boost;
using namespace boost::program_options;


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

    cf::top_matches(items, vm["r"].as<size_t>(), vm.count("cache"));
    return 0;
}
