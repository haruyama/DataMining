#!/usr/bin/env ruby
require 'json/ext'
require 'set'

items = Set.new
users = []
item_user = {}
user = 0
entries = 0
while line = gets
  (sute, *tmp_items) = line.chomp.split(',')
  users << user
  tmp_items.each { |item|
    items.add(item)
    unless item_user[item]
      item_user[item] = {}
    end
    item_user[item][user] = 1.0
    entries += 1
  }
  user += 1
end
items = items.to_a

result = []
result << items.size
result << users.size
result << entries
i = 0
items.each { |item|
  tmp = []
  tmp << item
  #tmp << item_user[item].size
  item_user[item].sort.each { |k,v|
    tmp << k
    i += 1
  }
  result << tmp
}
print result.to_json
