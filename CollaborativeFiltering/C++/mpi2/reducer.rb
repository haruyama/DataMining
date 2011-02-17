#!/usr/bin/env ruby

scores = Hash.new
oldid = nil


def print_score(id, scores)

  print id, ':'
  print scores.to_a.sort{ |a, b|
    (b[1] <=> a[1]) * 2 + (a[0] <=> b[0])
  }[0, 10].map { |simid, s|
    s.to_s + '|' + simid
  }.join(',')
  print ",\n"
end

$stdin.each { |l|
  id,simid,score= l.chomp.split(/\s+/)
  if oldid and id != oldid
    print_score(oldid, scores)
    scores = Hash.new
  end
  oldid = id
  scores[simid] =  score.to_f
}

print_score(oldid, scores)
