#!/usr/bin/env ruby
# -*- encoding: utf-8 -*-
# ruby implementation of SVM
# porting of https://code.google.com/p/tsubomi/source/browse/trunk/src/futaba/futaba_make_svm.pl

loop  = ARGV[0].to_i
alpha = ARGV[1].to_f

def predict(w, x)
  y = 0
  x.keys.each { |f|
    y += w[f] * x[f]
  }
  y
end

def train(w, x, t, a, r)
  if predict(w, x) * t < 1
    x.keys.each { |f|
      w[f] += (r * ((t * x[f]) - a * w[f]))
    }
  end
end

data = []
class_list = Hash.new(0.0)

$stdin.each { |l|
  a = l.chomp.split("\t")
  raise 'line ' + l.chomp + ' is invalid.' if a.size != 2
  class_key = a[0].sub(/:.+/, '')
  class_list[class_key] += 1

  v = Hash.new(0.0)
  a[1].split(',').each { |e|
    c = e.split(':')
    raise 'line ' + l.chomp + ' is invalid.' if c.size != 2
    v [c[0]] += c[1].to_f
  }
  data << [class_key, v]
}

class_keys = class_list.keys

ws = Hash.new
class_keys.each { |c|
  ws[c] = Hash.new(0.0)
}

(1 .. loop).each { |r|
  data.each { |d|
    class_keys.each { |c|
      train(ws[c], d[1], d[0] == c ? 1 : -1, alpha, 1.0/r)
    }
  }
}

ii = Hash.new {
  |h, k| h[k] = Hash.new(0.0)
}
class_keys.each { |c|
  ws[c].keys.each { |f|
    ii[f][c] += ws[c][f]
  }
}

ii.keys.each { |f|
  print f, "\t"
  scores = ii[f].keys.map{ |c|
    score = (ii[f][c] * 10000).to_i / 10000.0
    c + ":" + score.to_s
  }
  print scores.join(','), "\n"
}
