#pragma once

#include <vector>
#include <boost/tokenizer.hpp>
#include <string>

using namespace std;
using namespace boost;

namespace csv {

    vector<string> parse_line(const string& line) {

        tokenizer<escaped_list_separator<char> > tok(line);

        vector<string> row;

        for (tokenizer<escaped_list_separator<char> >::iterator 
                it = tok.begin();
                it != tok.end(); 
                ++it) {
            row.push_back(*it);
        }
        return row;
    }

    //手でdeleteが必要
//    vector<string>* parse_line_p(const string& line) {

//        tokenizer<escaped_list_separator<char> > tok(line);

//        vector<string>* row(new vector<string>);

//        for (tokenizer<escaped_list_separator<char> >::iterator 
//                it = tok.begin();
//                it != tok.end(); 
//                ++it) {
//            row->push_back(*it);
//        }
//        return row;
//    }
}
