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
        v = v.to_i
        row["label"] = v
        classes[v] = 1
      else
        next if v.zero?
        row["features"][k] = v
      end
    end

    row["labels"] 
    data << row
  end
  return {"labels" => classes.keys, "features" => header[0,header.size - 1], "data" => data}
end

module DecisionTreeHelper
  def to_s
    JSON.pretty_generate(summarize_node(@root))
  end
  
  def summarize_node node
    summary = {
      leaf: node.is_leaf?    
    }
    if node.is_leaf?
      summary[:class_distribution] = node.node_class_distribution
    else
      summary[:split] = node.split
      summary[:children] = node.children
        .sort_by{|kv| kv.first}
        .map do |kv|
          path, child = kv      
          [path, summarize_node(child)]
        end.to_h
    end

    return summary
  end
end
