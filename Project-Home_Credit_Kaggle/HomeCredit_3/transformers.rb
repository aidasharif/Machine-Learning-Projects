##transformers.rb

#### KEEP THIS AT THE TOP OF YOUR FILE ####
class TransformingLearner
  include Learner
  attr_accessor :name
  def initialize transformer, learner
    @parameters = learner.parameters
    @parameters["learner"] = learner.parameters["name"] || learner.class.name
    @transformer = transformer
    @learner = learner
    @name = self.class.name
  end
  def train dataset
    @transformer.train dataset
    transformed_examples = @transformer.apply dataset["data"]
    train_dataset = dataset.clone
    train_dataset["data"] = transformed_examples
    @learner.train train_dataset
  end
  
  def predict example
    transformed_example = @transformer.apply [example]
    @learner.predict transformed_example.first
  end
  
  def evaluate dataset
    transformed_dataset = dataset.clone
    transformed_dataset["data"] = @transformer.apply dataset["data"]
    @learner.evaluate transformed_dataset
  end
end

class CopyingTransformingLearner
  include Learner
  attr_accessor :name
  def initialize transformer, learner
    @parameters = learner.parameters
    @parameters["learner"] = learner.parameters["name"] || learner.class.name
    @transformer = transformer
    @learner = learner
    @name = self.class.name
  end

  def clone_example example
    e = example.clone
    e["features"] = example["features"].clone
    return e
  end
    
  def clone_dataset dataset
    cloned_dataset = dataset.clone
    cloned_dataset["features"] = dataset["features"].clone
    cloned_dataset["data"] = dataset["data"].map {|e| clone_example(e)}
    return cloned_dataset
  end
    
  def train dataset
    @transformer.train clone_dataset(dataset)
      
    train_dataset = clone_dataset(dataset)
    transformed_examples = @transformer.apply train_dataset["data"]
    train_dataset["data"] = transformed_examples
    @learner.train train_dataset
  end
  
  def predict example
    transformed_example = @transformer.apply [clone_example(example)]
    @learner.predict transformed_example.first
  end
  
  def evaluate dataset
    transformed_dataset = clone_dataset dataset
    transformed_examples = @transformer.apply transformed_dataset["data"]
    transformed_dataset["data"] = transformed_examples
    @learner.evaluate transformed_dataset
  end
end

### ADD YOUR CODE AFTER THIS ###

class FeatureTransformPipeline
  def initialize *transformers
    @transformers = transformers
  end
  
  def train dataset
    @transformers.each do |item|
      transform=item
      transform.train dataset
      transform.apply dataset["data"]
    end    
  end
  
  def apply example_batch 
    return @transformers.inject(example_batch) do |u, transform|
      u = transform.apply example_batch
    end
  end
end

class AgeRangeAsVector
  def initialize; end
  def train dataset; end
  def apply(example_batch)
    min_age = 0
    max_age = 100
    feature_name = "days_birth"
    pattern = "age_range_%d"
    example_batch.each do |item|
      next if !(item["features"][feature_name])
      age=5*((-item["features"][feature_name])/(365*5)).floor
      
      if age>100
        age=100
      end
      
      if age<0
        age=0
      end
      item["features"][pattern % [age]]=1
      item["features"].delete(feature_name)
    end
    
    return example_batch
  end
end

class DaysEmployedVector
  def initialize; end
  def train dataset; end
  def apply(example_batch)
    min_age = 0
    max_age = 100
    feature_name = "days_employed"
    pattern = "employed_range_%d"
    example_batch.each do |item|
      next if !(item["features"][feature_name])
      age=5*((-item["features"][feature_name])/(365*5)).floor
      
      if age>100
        age=100
      end
      
      if age<0
        age=0
      end
      item["features"][pattern % [age]]=1
      item["features"].delete(feature_name)
    end
    
    return example_batch
  end
end


class TargetAveraging
  attr_reader :means
  
  def initialize feature_names
    @means = Hash.new {|h,k| h[k] = Hash.new {|h,k| h[k] = 0}}
    @feature_names = feature_names
    @pattern = "avg_%s"
    @total=Hash.new {|h,k| h[k] = Hash.new {|h,k| h[k] = 0}}
  end
  
  def train dataset   
    
    dataset["data"].each do |item|
      item["features"].each do |key,array|
        if (array.is_a? (String))
          @total[key][array]+=1.0
        end
      end
    end
    
    dataset["data"].each do |item|
      item["features"].each do |key,array|
        if (array.is_a? (String)) and item["label"]==1 and @feature_names.include? key
          @means[key][array]+=1.0/(@total[key][array])
        end
      end
    end
    
  end
    
  def apply(example_batch)
    
    example_batch.clone.each do |item|
      item["features"].clone.each do |key,array|
        if (array.is_a? (String)) and (@feature_names.include? key)
          new_key="avg_"+key
          item["features"][new_key] = @means[key][array]
          item["features"].delete(key)
        end
      end
    end
    
    return example_batch
  end

end


class MeanImputation
  attr_reader :means
  
  def initialize feature_names
    @means = Hash.new {|h,k| h[k] = 0}
    @miss = Hash.new {|h,k| h[k] = 0}
    @feature_names = feature_names 
  end
  
  def train dataset    
    data=dataset["data"]

    @feature_names.each do |feature|
      mean=[]
      data.each do |item|
        next if item["features"][feature].nil?
        next if !(item["features"][feature].is_a? Numeric)
        mean << item["features"][feature]           
      end
      @means[feature] = mean(mean)
    end 
    
  end
    
  def apply(example_batch)
        
    example_batch.each do |item|
      @feature_names.each do |feature|
        if item["features"][feature]==nil and @means[feature].is_a? (Numeric)
          item["features"][feature]=@means[feature]
        end
      end
    end  
    return example_batch
    
  end
end


class L2Normalize
  def train dataset; end
  def apply(example_batch)
    
    number=0
    total=Hash.new {|h,k| h[k] = 0}
    example_batch.each do |item|
      item["features"].each do |key,array|
        if (array.is_a? (Numeric))
          total[number]=(array)**2+total[number]
        end
      end
      number+=1
    end
    
    number=0
    example_batch.each do |item|
      item["features"].each do |key,array|
        if (array.is_a? (Numeric))
          item["features"][key]=array.to_f/((total[number])**0.5)
        end
      end
      number+=1
    end

    return example_batch
  end
end



class OneHotEncoding
  def initialize feature_names
    @feature_names = feature_names
    @pattern = "%s=%s"
  end
  
  def train dataset; end
  
  def apply(example_batch)

    example_batch.clone.each do |item|
      @feature_names.each do |feature|
        if (item["features"][feature].is_a? (String))
          new_key=feature+"="+item["features"][feature]
          item["features"][new_key] = 1.0
          item["features"].delete(feature)
        end
      end
    end
    
    return example_batch
  end
end


class LogTransform
  def initialize feature_names
    @feature_names = feature_names
    @pattern = "log_%s"
  end
  
  def train dataset; end
  
  def apply(example_batch)
    
    example_batch.clone.each do |item|
      @feature_names.each do |feature|

        if (item["features"][feature].is_a? (Numeric)) and item["features"][feature]>0
          new_key="log_"+feature
          item["features"][new_key] = Math.log(item["features"][feature])
          item["features"].delete(feature)
        end
      end
    end
    return example_batch
  end
end


class ZScoreTransformer
  attr_reader :means, :stdevs
  
  def initialize feature_names
    @means = Hash.new {|h,k| h[k] = 0}
    @miss = Hash.new {|h,k| h[k] = 0}
    @stdevs = Hash.new {|h,k| h[k] = 0}
    @feature_names = feature_names    
  end
  
  def train dataset
    data = dataset["data"]

    @feature_names.each do |feature|
      mean=[]
      data.each do |item|
        next if item["features"][feature].nil?
        next if !(item["features"][feature].is_a? Numeric)
        mean << item["features"][feature]           
      end
      @means[feature] = mean(mean)
    end 
    
    @feature_names.each do |feature|
      data.each do |item|
        next if item["features"][feature].nil? or !(item["features"][feature].is_a? (Numeric))
        @stdevs[feature]=(item["features"][feature]-@means[feature])**2 +@stdevs[feature]      
      end
      @stdevs[feature] = (@stdevs[feature].to_f/data.size)**0.5
    end 
  end
  
  def apply example_batch
    
    example_batch.each do |item|
      @feature_names.each do |feature|
        next if !(item["features"][feature]) or @stdevs[feature]==0 or !((item["features"][feature]).is_a? (Numeric))
        item["features"][feature]=(item["features"][feature]-@means[feature])/(@stdevs[feature])
      end
    end
    return example_batch
  end

end



