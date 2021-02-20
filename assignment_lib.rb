require 'test/unit/assertions'
require 'json'
require 'daru'
require 'distribution'
include Test::Unit::Assertions

def plot_clusters(data)
  x1 = []
  x2 = []
  target = []
  data.each do |row|
    x1 << row["features"]["x1"]
    x2 << row["features"]["x2"]
    target << row["cluster"]
  end
  df = Daru::DataFrame.new({x1: x1, x2: x2, target: target})
  df.to_category :target
  df.plot(type: :scatter, x: :x1, y: :x2, categorized: {by: :target, method: :color}) do |plot, diagram|
    plot.xrange [-8,8]
    plot.x_label "X1"
    plot.yrange [-8,8]  
    plot.y_label "X2"
    plot.legend false
  end
end