##linear_models.rb

def dot x, weight
  sum=0
  x.each do |key1, array1|
    weight.each do |key2, array2|
      if key1==key2 then
        sum+=array1*array2
      end
    end
  end
  return sum
end


def norm weight
  sum=0
  sum = Math.sqrt(dot(weight,weight))
  return sum
end


class StochasticGradientDescent
  attr_reader :weights
  attr_reader :objective
  def initialize obj, weight, lr = 0.01
    @objective = obj
    @weights = weight
    @n = 1.0
    @lr = lr
  end
  def update x
    dw = @objective.grad(x, @weights)
    learning_rate = @lr/Math.sqrt(@n)
    
    dw.each_key do |key|
      @weights[key] -= learning_rate * dw[key]
    end

    @objective.adjust @weights
    @n += 1.0
  end
end



class LogisticRegressionModelL2
  def initialize reg_param
    @reg_param = reg_param
  end


  def predict example, weight
    x = item["features"]    
    1.0 / (1 + Math.exp(-dot(weight, x)))
  end
  
  def adjust weight
    weight.each_key {|key| weight[key] = 0.0 if weight[key].nan? or weight[key].infinite?}
    weight.each_key {|key| weight[key] = 0.0 if weight[key].abs > 1e5 }
  end
  
  def func data, weight
    loss = data.inject(0.0) do |sum,item| 
      y = item["label"].to_f > 0 ? 1.0 : -1.0
      x = item["features"]
      predict = dot(weight,x)
      
      sum += Math.log(1 + Math.exp(-y * predict))
    end
    
    loss/data.size.to_f + @reg_param * 0.5 * (norm(weight) ** 2.0)
  end
  
  def grad data, weight  

    grad = Hash.new {|h,key| h[key] = 0.0}
    data.each do |item| 
      y = item["label"].to_f > 0 ? 1.0 : 0.0
      x = item["features"]
      predict= x.keys.inject(0.0) {|s, key| s += weight[key] * x[key]}
      sig = 1.0 / (1 + Math.exp(-predict))
      x.each_key do |key|
        grad[key] += (sig - y) * x[key]
      end
    end
    grad.each_key {|key| grad[key] = (grad[key] / data.size) + @reg_param * weight[key]}
    return grad
  end
end


class LogisticRegressionLearner
  attr_reader :parameters
  attr_reader :weights  
  include Learner  
  
  def initialize regularization: 0.0, learning_rate: 0.01, batch_size: 20, epochs: 1
    @parameters = {"regularization" => regularization, 
      "learning_rate" => learning_rate, 
      "epochs" => epochs, "batch_size" => batch_size}
  end
  
  def train dataset

    @weights = Hash.new {|h,key| h[key] = 0.0}
    obj = LogisticRegressionModelL2.new @parameters["regularization"]
    sgd = StochasticGradientDescent.new obj, @weights, @parameters["learning_rate"]
    @parameters["epochs"].times do 
      dataset["data"].each_slice(@parameters["batch_size"]) do |batch| 
        sgd.update batch
      end
    end
  end
  
  def evaluate eval_dataset
    examples = eval_dataset["data"]
    examples.map do |example|
      score = predict(example)
      label = example["label"] > 0 ? 1 : 0
      [score, label]
    end
  end
  
  def predict example
    x = example["features"]    
    1.0 / (1 + Math.exp(-dot(@weights, x))) 
  end
end  
