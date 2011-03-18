#!/usr/bin/env ruby
require 'json/ext'
require 'set'

items = Set.new
users = []
item_user = {}
user = 0
while line = gets
  (sute, *tmp_items) = line.chomp.split(',')
  users << user
  tmp_items.each { |item|
    items.add(item)
    unless item_user[item]
      item_user[item] = {}
    end
    item_user[item][user] = 1.0
  }
  user += 1
end
items = items.to_a

result = []
result << items.size
result << users.size

items.each { |item|
  tmp = []
  tmp << item
  item_user[item].each { |k,v|
    tmp << k
  }
  result << tmp
}
print result.to_json

