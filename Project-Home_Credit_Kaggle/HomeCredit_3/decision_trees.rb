##decision_trees.rb

#### KEEP THIS AT THE TOP OF YOUR FILE ####

SEED = 'eifjcchdivlbcbflbgblfgukbtkhvejvtkevfbtetjnl'.to_i(26)
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


### ADD YOUR CODE AFTER THIS LINE ###


def class_distribution dataset
  classes = Hash.new {|h,k| h[k] = 0}
  dataset.each do |item|
    classes[item["label"]]=1+classes[item["label"]]
  end
  
  result={}
  classes.each do |key,array|
    result[key]=array.to_f/dataset.size.to_f
  end
  
  return result
end


def entropy dist
  ent=0
  dist.each do |key,array|
    if array==0
      return 0.0
    end
    ent+=-array*Math.log(array)
  end
  
  ent
end

def information_gain h0, splits
 size = Hash.new {|h,k| h[k] = 0}
  sum = Hash.new {|h,k| h[k] = 0}
  total=0
  
  splits.each do |key, array|
    total+=array.size
  end
  
  splits.each do |key, array|
    sum[key]+=entropy(class_distribution(array))
    size[key]=array.size
  end
  
  result=0
  
  size.each do |key,array|
    size[key]=size[key].to_f/total
  end
  
  splits.each do |key, array|
    result+=sum[key]*size[key]
  end

  return h0-result
end

class CategoricalSplit
  attr_reader :feature_name
  
  def initialize fname
    @feature_name = fname
    @path_pattern = "%s == '%s'"
  end
  
  def to_s
    "Categorical[#{@feature_name}]"
  end

  def split_on_feature examples
    
    splits = Hash.new {|h,k| h[k] = []}
    examples.each do |item|
      item["features"].each do |key2,array2|
        if key2==@feature_name
          @path_pattern = "%s == '%s'" % [key2, array2.to_s]
          splits[@path_pattern]=(splits[@path_pattern]).append(item)
        end
      end
    end

    return splits
  end
end

class CategoricalSplit
  def test example 

    if example["features"][@feature_name]
      return "%s == '%s'" % [@feature_name, example["features"][@feature_name]]
    end 
      
    return nil

  end
end

class NumericSplit
  attr_reader :feature_name, :split_point, :paths
  def initialize fname, value
    @feature_name = fname
    @split_point = value
    @split_point_str = "%.2g" % @split_point
    @paths = ["#{@feature_name} < #{@split_point_str}", "#{@feature_name} >= #{@split_point_str}"]
  end
  
  def to_s
    "Numeric[#{@feature_name} <=> #{@split_point_str}]"
  end

  def split_on_feature examples
    splits = Hash.new 

    splits={@paths[0]=>[], @paths[1]=>[]}

    examples.each do |item|
      if item["features"][@feature_name]
        y=item["features"][@feature_name]
      else
        y=0.0
      end
      if y<@split_point
        splits[@paths[0]] << item
      end
      if y>=@split_point
        splits[@paths[1]] << item
      end
    end
  
    return splits
  end
end

class NumericSplit
  def test example
      
    if example["features"][@feature_name]
      y=example["features"][@feature_name]
      if y==nil
        return nil
      end
      if y<@split_point
        return @paths[0]
      elsif y>=@split_point
        return @paths[1]
      end
    end
  
  end
end

class CategoricalSplitter
  def matches? examples, feature_name
    has_feature = examples.select {|r| r["features"].has_key? feature_name} 
    return false if has_feature.empty?    
    return has_feature.all? do |r| 
      r["features"].fetch(feature_name, 0.0).is_a?(String)
    end
  end
  
  def create_split examples, parent_entropy, feature_name
    
    if(matches?(examples, feature_name))
      categorical_split = CategoricalSplit.new feature_name
      splits = categorical_split.split_on_feature examples
      ig = information_gain parent_entropy, splits
      return {"split" => categorical_split, "information_gain" => ig}
    end
     
  end
  
end

class NumericSplitter
  def matches? examples, feature_name
    has_feature = examples.select {|r| r["features"].has_key? feature_name} 
    return false if has_feature.empty?    
    return has_feature.all? do |r| 
      r["features"].fetch(feature_name, 0.0).is_a?(Numeric)
    end
  end
    
    
  def create_split x, entropy, fname 
    
    if(!matches?(x, fname))
      return nil
    end
      
    max_infogain = 0.0
    max_threshold = nil

    hash_featureValues = x.group_by {|r| r['features'].fetch(fname, 0.0)}

    sum_right = Hash.new {|h,k| h[k] = 0}
    sum_left = Hash.new {|h,k| h[k] = 0}
    total_left =0.0
    total_right = x.size.to_f


    hash_featureValues.each_key do |item|
      number = Hash.new {|h,k| h[k] = 0}
      hash_featureValues[item].each do |r|
        number[r['label']] += 1
        sum_right[r['label']] += 1
      end
      hash_featureValues[item] = number
    end

    thresholds = hash_featureValues.keys.sort
    tre1 = thresholds.shift 
    
    hash_featureValues[tre1].each_key do |key|
      sum_left[key] += hash_featureValues[tre1][key]
      sum_right[key] -= hash_featureValues[tre1][key]
      total_left += hash_featureValues[tre1][key]
      total_right -= hash_featureValues[tre1][key]
    end

    igs=[]
    thresholds.each.with_index do |tre,i|
      ratioL = total_left.to_f / x.size
      ratioR = total_right.to_f / x.size

      left_dist = Hash.new
      right_dist = Hash.new
      sum_left.each_key { |key| left_dist[key] = sum_left[key].to_f / total_left }
      sum_right.each_key { |key| right_dist[key] = sum_right[key].to_f / total_right }

      entropy_left = entropy(left_dist)
      entropy_right = entropy(right_dist)
      ig = entropy - (ratioL * entropy_left + ratioR * entropy_right)
      igs << ig
      if ig > max_infogain

        if igs[i-1]==nil
          max_infogain=igs[i]
        else
          max_infogain = igs[i-1]
        end

        if thresholds[i-1]==nil
          max_threshold=thresholds[i]
        else
          max_threshold = thresholds[i-1]
        end
      end

      hash_featureValues[tre].each_key do |key|
        sum_left[key] += hash_featureValues[tre][key]
        sum_right[key] -= hash_featureValues[tre][key]
        total_left += hash_featureValues[tre][key]
        total_right -= hash_featureValues[tre][key]
      end
    end
       
    if max_threshold==nil
        max_threshold=0
        max_infogain=0
        return nil
    else
        split = NumericSplit.new fname, max_threshold
        return {"split" => split, "information_gain" => max_infogain}
    end
  end
end

class DecisionNode
  attr_reader :children, :examples, :split, :node_entropy, :node_class_distribution
  
  def initialize examples
    @examples = examples
    @node_class_distribution = class_distribution examples    
    @node_entropy = entropy (@node_class_distribution)
    @children = Hash.new
  end
  
  def is_leaf?
    self.children.empty?
  end
  
  def set_split thisSplit
    @split=thisSplit
  end
      
  def score positive_class_label
    
    scores=Hash.new {|h,k| h[k] = 0}

    if @node_class_distribution[positive_class_label]==nil
      return 0
    else return @node_class_distribution[positive_class_label]
    end
  end

end


class DecisionNode
  def all_possible_splits feature_names, splitters
    all_splits = []
    
    splitters.each do |split|
      feature_names.each do |item|

        split_result = split.create_split @examples, @node_entropy, item
        if !split_result or (split_result["split"]==nil and split_result["information_gain"]==nil) 
          split_result=nil
        else 
          all_splits+=[split_result]
        end
      end
    end

    all_splits.delete(nil)
    
    return all_splits
  end
end


class DecisionNode
  def split_node! split    
    @split = split
    
    mySplit=@split.split_on_feature(@examples)

    mySplit.each do |key,array|
      @children[key]=DecisionNode.new(array)
    end

    @examples = nil
  end
end


class DecisionTreeLearner
  include DecisionTreeHelper
  include Learner  
  attr_reader :root
  
  def initialize positive_class_label, min_size: 10, max_depth: 50
    @splitters = [CategoricalSplitter.new, NumericSplitter.new]
    @parameters = {"min_size" => min_size, "max_depth" => max_depth}
    @positive_class_label = positive_class_label
  end
    
  def train dataset
    @feature_names = dataset["features"]
    examples = dataset["data"]
    @root = DecisionNode.new examples
    grow_tree @root, @parameters["max_depth"]
  end

  def grow_tree parent, remaining_depth

    if parent.examples.size<@parameters["min_size"] or !@feature_names.size 
      return
    end
   
    all_splits=parent.all_possible_splits(@feature_names,@splitters)
    sorted_splits=all_splits.sort_by{|split| split["information_gain"]}.reverse
    
    best_split=sorted_splits.first
    return if best_split.nil?
    split=best_split["split"]
    
    decreased_depth=remaining_depth-1
    parent.set_split(split)
    
    if (decreased_depth>0)
      parent.split_node! (split)
      children=parent.children
      children.each do |child|
        decreased_depth=remaining_depth-1
        child_decision_node=child[1]
        grow_tree(child_decision_node,decreased_depth)
      end
    end
  end
    
end


class DecisionTreeLearner
  attr_accessor :positive_class_label
  def predict example
    leaf = find_leaf @root, example
    return leaf.score @positive_class_label
  end

  def evaluate eval_dataset
    examples = eval_dataset["data"]
    examples.map do |example|
      score = predict(example)
      label = example["label"] == @positive_class_label ? 1 : 0
      [score, label]
    end
  end

def find_leaf node, example
    
    if(node.is_leaf?)
      return node
    end
      
    children = node.children
    node_split = node.split

    children.each do |child_node|
      condition = child_node[0]
      child_decision_node = child_node[1]
      test = node_split.test(example)
      if(condition.eql?(test))
        return find_leaf(child_decision_node, example)
      end
    end
    return node
  end
end

def random_features_subset dataset, rng
  #changed it to 4  increase features subset decrease divesity
  num_features = 3
  feature_list = dataset["features"].sample(num_features, random: rng)  
end

def random_forest_dataset dataset, rng
  feature_list = random_features_subset dataset, rng
  examples = dataset["data"]
  new_dataset = dataset.clone  
  bootstrap=[]
  
  dataset["data"].size.times do |i|
    temp=dataset["data"][rng.rand(dataset["data"].size)].clone
    temp["features"]=temp["features"].clone
    temp["features"].each do |key,array|
      if !(feature_list.include?(key))
        temp["features"].delete(key)
      end
    end
    bootstrap << temp
  end
  
  new_dataset["features"]=feature_list
  new_dataset["data"]=bootstrap

  return new_dataset
end


class RandomForestLearner
  include Learner  
  attr_reader :trees
  
  def initialize positive_class_label, num_trees: 10, min_size: 10, max_depth: 50
    @parameters = {"num_trees" => num_trees, "min_size" => min_size, "max_depth" => max_depth}
    @positive_class_label = positive_class_label
    tree_parameters = @parameters.clone.delete :num_trees
    
    @trees = Array.new(num_trees) do |i| 
    DecisionTreeLearner.new @positive_class_label, min_size: min_size, max_depth: max_depth
    end
  end
  
  def to_s
    JSON.pretty_generate(@trees.collect {|t| t.summarize_node t.root})
  end
  
  def train dataset
    rng = Random.new SEED
    @trees.each do |tree|
      train_dataset = random_forest_dataset(dataset, rng)
      tree.train(train_dataset)     
    end
  end
end
  
class RandomForestLearner
  attr_accessor :positive_class_label
  
  def evaluate eval_dataset
    examples = eval_dataset["data"]
    examples.map do |example|
      score = predict(example)
      label = example["label"] == @positive_class_label ? 1 : 0
      [score, label]
    end
  end
  
  def predict example

    scores = @trees.inject(0.0) {|u, t| u += t.predict(example) }
    scores/(@trees.size).to_f
    
  end
end