#!/usr/bin/env ruby
require 'rubygems'
require 'json/ext'


# item_length = 10000
# user_length = 100000
item_length = ARGV[0].to_i
user_length = ARGV[1].to_i

items = []

data_length = 0
(0..item_length-1).each { |i|
  users = []
  num = rand(ARGV[2].to_i) + 5
  data_length += num
  (0..num-1).each {
    users << rand(user_length)
  }
  users = users.uniq.sort
  users.unshift 'item_' + (i + user_length).to_s
  items << users
}

header = []
header << item_length
header << user_length
header << data_length
items = header + items
print items.to_json
