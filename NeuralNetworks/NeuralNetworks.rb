require 'test/unit/assertions'
require 'json'
require 'distribution'
require 'gnuplotrb'

include Test::Unit::Assertions
include GnuplotRB
include Math

def plot_decision_boundary(data, model)
  pos_x1 = data.select{|r| r["label"] > 0}.collect {|r| r["features"]["x1"]}
  pos_x2 = data.select{|r| r["label"] > 0}.collect {|r| r["features"]["x2"]}
  neg_x1 = data.select{|r| r["label"] <= 0}.collect {|r| r["features"]["x1"]}
  neg_x2 = data.select{|r| r["label"] <= 0}.collect {|r| r["features"]["x2"]}

  f_x1 = []
  f_x2 = []
  f_z = []
  -1.step(1.0, 0.1) do |x1|
    -1.step(1.0, 0.1) do |x2|
      f_x1 << x1
      f_x2 << x2
      f_z += model.predict([{"features" => {"x1" => x1, "x2" => x2}}])
#       puts [x1, x2, f_z.last].join("\t")
    end
  end
    
  Splot.new(
    [[f_x1,f_x2,Array.new(f_x1.size, 0), f_z], with: 'image', using: '1:2:3:4'],
    [[neg_x1,neg_x2, Array.new(neg_x1.size, 0)], with: 'points'], 
    [[pos_x1,pos_x2, Array.new(pos_x1.size, 0)], with: 'points'], 
    xrange: (-1)..1, yrange: (-1)..1, 
    palette: 'rgb 34,35,36',
#     view: [0, 0],
    style: 'data lines',
    term: 'png', key: false
  )
end

def xor_dataset rng
  means = [[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]]
  labels = [-1.0, 1.0, 1.0, -1.0]
  Array.new(500) do |i| 
    mx = means[i % 4]
    lbl = labels[i % 4]
    
    {"features" => {"x1" => (rng.call * 0.2) + mx[0], "x2" => (rng.call * 0.2) + mx[1]}, "label" => lbl}
  end
end

def concentric_dataset
  data = []
  File.open("concentric.tsv").each_line do |l|
    x, y, label = l.chomp.split "\t"
    data << {"features" => {"x1" => (x.to_f / 3), "x2" => (y.to_f / 3)}, "label" => 2 * label.to_f - 1.0} 
  end
  return data
end

def generate_synthetic_data rng
  means = [[-0.5, -0.5], [0.5, 0.5]]
  labels = [-1.0, 1.0]
  Array.new(5000) do |i| 
    mx = means[i % 2]
    lbl = labels[i % 2]
    
    {"features" => {"x1" => (rng.call * 0.2) + mx[0], "x2" => (rng.call * 0.2) + mx[1]}, "label" => lbl}
  end
end


def plot_dataset concentric_dataset
  neg, pos = [-1.0, 1.0].map do |cls|
    d = concentric_dataset.select {|r| r["label"] == cls}

    x = d.map {|r| r["features"]["x1"]}
    y = d.map {|r| r["features"]["x2"]}
    [x, y]
  end

  Plot.new(
      [neg, with: 'points', title: 'Negative'],
      [pos, with: 'points', title: 'Positive'],
      color: 3,
      term: 'png', key: true
    )
end