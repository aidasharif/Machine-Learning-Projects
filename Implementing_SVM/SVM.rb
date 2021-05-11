require 'test/unit/assertions'
require 'json'
require 'daru'
require 'distribution'
require 'libsvm'
include Test::Unit::Assertions

def spiral_dataset
  u = Array.new
  97.times do |i|
    angle = i * Math::PI / 16.0
    radius = 6.5 * (104 - i) / 104.0
    x = radius * Math.sin(angle)
    y = radius * Math.cos(angle)

    u << {"features" => {"x1" => x, "x2" => y}, "label" => 1.0}
    u << {"features" => {"x1" => -x, "x2" => -y}, "label" => 0.0}    
  end
  return {"data" => u, "features" => ["x1", "x2"], "labels" => ["1", "0"]}
end

def circle_dataset
    srand 98501257877
    examples = Array.new(1000) do |i|
        x1 = rand
        x2 = rand
        in_small_circle = ((x1 - 0.5) ** 2.0) + ((x2 - 0.5) ** 2.0) < 0.05
        in_large_circle = ((x1 - 0.5) ** 2.0) + ((x2 - 0.5) ** 2.0) < 0.1
        label = if in_small_circle
            1.0
        elsif in_large_circle and rand < 0.5
            1.0
        else
            -1.0
        end
        {"features" => {"x1" => x1, "x2" => x2}, "label" => label}
    end    
    srand Time.now.to_i
    {"data" => examples, "features" => %w(x1 x2), "labels" => [1, -1]}
end

def two_gaussians_dataset
    examples = []
    rng = Random.new("eifjcchdivlbriiuhnuktntrjkfhgfviklgcckjvvkbk".to_i(26))
    File.open("two_gaussians.tsv").each_line do |l|
        x1, x2, label = l.chomp.split("\t").collect {|x| x.to_f}        
        examples << {"features" => {"x1" => x1, "x2" => x2}, "label" => label.to_i}
    end    
    
    {"data" => examples.shuffle(random: rng), "features" => %w(x1 x2), "labels" => [1, -1]}
end

def two_gaussians_sep_dataset
    examples = []
    rng = Random.new("eifjcchdivlbriiuhnuktntrjkfhgfviklgcckjvvkbk".to_i(26))
    File.open("two_gaussians-sep.tsv").each_line do |l|
        x1, x2, label = l.chomp.split("\t").collect {|x| x.to_f}        
        examples << {"features" => {"x1" => x1, "x2" => x2}, "label" => label.to_i}
    end    
    
    {"data" => examples.shuffle(random: rng), "features" => %w(x1 x2), "labels" => [1, -1]}
end

def two_gaussians_sep_nosv_dataset
    examples = []
    rng = Random.new("eifjcchdivlbriiuhnuktntrjkfhgfviklgcckjvvkbk".to_i(26))
    File.open("two_gaussians-sep-nosv.tsv").each_line do |l|
        x1, x2, label = l.chomp.split("\t").collect {|x| x.to_f}        
        examples << {"features" => {"x1" => x1, "x2" => x2}, "label" => label.to_i}
    end    
    
    {"data" => examples.shuffle(random: rng), "features" => %w(x1 x2), "labels" => [1, -1]}
end


module Learner  
  attr_reader :parameters
  attr_accessor :name
  def name
      @name.nil? ? self.class.name : @name
  end

  def train train_dataset    
  end
  def predict example
  end
  def evaluate eval_dataset
  end
end
  

module Metric
  def apply scores
  end
end

def load_support_vectors(filename, id_to_feature)
  dataset = {
    "labels" => [-1, 1],
    "bias" => 0.0,
    "features" => id_to_feature.keys.sort.map {|id| id_to_feature[id]},
    "data" => []
  }

  
  in_sv = false
  in_kernel = false
  File.open(filename).each_line do |l| 
#     print l
    if not in_sv and l =~ /^SV/
      in_sv = true
      next
    elsif not in_sv  and l =~ /^rho/
      dataset["bias"] = -1.0 * l.chomp.split(" ").last.to_f
      next
    elsif not in_sv
      next
    end
      
    row = l.chomp.split " "
    y_alpha = row.shift.to_f
    features = Hash.new
    label = y_alpha > 0 ? 1 : -1

    row.each do |kv|
      feature_id, v = kv.split ":"
      feature_id = feature_id.to_i
      raise ArgumentError.new("Unexpected feature id: '#{id}'") unless id_to_feature.has_key? feature_id
      feature_name = id_to_feature[feature_id]
      v = v.to_f
      features[feature_name] = v unless v.zero?
    end
    example = {"label" => label, "features" => features, "alpha" => y_alpha.abs}
    dataset["data"] << example
  end
    
  return dataset
end

def create_feature_maps dataset
  feature_to_id = Hash.new
  id_to_feature = Hash.new
  
  dataset["features"].each.with_index do |feature_name, feature_id|
    feature_to_id[feature_name] = feature_id
    id_to_feature[feature_id] = feature_name
  end

  [feature_to_id, id_to_feature]
end

def dataset_to_libsvm dataset, feature_to_id  
  libsvm_examples = dataset["data"].map do |example|
    example_with_ids = Hash.new
    example["features"].each_key do |feature_name|
      raise ArgumentError("Unexpected feature name: '#{feature_name}'") unless feature_to_id.has_key? feature_name
      feature_id = feature_to_id[feature_name]
      example_with_ids[feature_id] = example["features"][feature_name]
    end
    Libsvm::Node.features(example_with_ids)    
  end
  
  libsvm_labels =  dataset["data"].map do |example|
    example["label"]
  end

  [libsvm_examples, libsvm_labels]
end

def train_libsvm dataset, kernel, c, filename
  feature_to_id, id_to_feature = create_feature_maps dataset
  libsvm_examples, libsvm_labels = dataset_to_libsvm dataset, feature_to_id

  problem = Libsvm::Problem.new
  parameter = Libsvm::SvmParameter.new

  parameter.cache_size = 1
  kernel.update_parameter(parameter)
  parameter.eps = 0.001
  parameter.c = c

  problem.set_examples(libsvm_labels, libsvm_examples)

  Libsvm::Model.train(problem, parameter).save(filename)
  svm_model = load_support_vectors(filename, id_to_feature)
  
  return svm_model
end
    
def plot x, y
  Daru::DataFrame.new({x: x, y: y}).plot(type: :line, x: :x, y: :y) do |plot, diagram|
    plot.x_label "X"
    plot.y_label "Y"
  end
end

require 'gnuplotrb'

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
  0.step(1.0, 0.1) do |x1|
    0.step(1.0, 0.1) do |x2|
      f_x1 << x1
      f_x2 << x2
      f_z << model.predict({"features" => {"x1" => x1, "x2" => x2}})
#       puts [x1, x2, f_z.last].join("\t")
    end
  end
    
  Splot.new(
    [[f_x1,f_x2,Array.new(f_x1.size, 0), f_z], with: 'image', using: '1:2:3:4'],
    [[neg_x1,neg_x2, Array.new(neg_x1.size, 0)], with: 'points'], 
    [[pos_x1,pos_x2, Array.new(pos_x1.size, 0)], with: 'points'], 
    xrange: 0..1, yrange: 0..1, 
    palette: 'rgb 34,35,36',
#     view: [0, 0],
    style: 'data lines',
    term: 'png', key: false
  )
end