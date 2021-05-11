require 'test/unit/assertions'
require 'daru'
require 'distribution'
require 'json'

include Test::Unit::Assertions

## Loads data files
def read_sparse_data_from_csv prefix
  data = []
  classes = Hash.new {|h,k| h[k] = 0}
  header = File.read(prefix + ".header").chomp.split(",")  
  
  File.open(prefix + ".csv").each_line.with_index do |l, i|
    a = l.chomp.split ","
    next if a.empty?
    row = {"features" => Hash.new}
    
    header.each.with_index do |k, i|
      v = a[i].to_f
      if k == "label"
        row["label"] = v.to_f
      else
        next if v.zero?
        row["features"][k] = v
      end
    end

    row["features"]["1"] = 1
    data << row
  end
  return {"classes" => classes, "features" => header[0,header.size - 1] + ["1"], "data" => data}
end

def mean x
  sum = x.inject(0.0) {|u,v| u += v}
  sum / x.size
end

def stdev x
  m = mean x
  sum = x.inject(0.0) {|u,v| u += (v - m) ** 2.0}
  Math.sqrt(sum / (x.size - 1))
end
