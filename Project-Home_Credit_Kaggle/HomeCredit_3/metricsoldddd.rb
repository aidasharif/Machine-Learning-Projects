### metrics.rb

#### KEEP THIS AT THE TOP OF YOUR FILE ####
def plot_roc_curve fp, tp, auc
  plot = Daru::DataFrame.new({x: fp, y: tp}).plot(type: :line, x: :x, y: :y) do |plot, diagram|
    plot.x_label "False Positive Rate"
    plot.y_label "True Positive Rate"
    diagram.title("AUC: %.4f" % auc)
    plot.legend(true)
  end
end  



def cross_validation_model_performance dataset, folds, learners, metric    
  learners.map do |learner|
    tr_metrics = []
    te_metrics = []
    puts "#{folds}-fold CV: #{learner.class.name}, parameters: #{learner.parameters}"
    cross_validate dataset, folds do |train_dataset, test_dataset|
      learner.train train_dataset
      train_scores = learner.evaluate train_dataset
      test_scores = learner.evaluate test_dataset      
      tr_metrics << metric.apply(train_scores)
      te_metrics << metric.apply(test_scores)
    end
      

    #Train on full training set
    learner.train dataset
    learner_name = learner.name
    puts mean(te_metrics)
    {
      "learner" => learner_name, "trained_model" => learner, "parameters" => learner.parameters, "folds" => folds,
      "mean_train_metric" => mean(tr_metrics), "stdev_train_metric" => stdev(tr_metrics),
      "mean_test_metric" => mean(te_metrics), "stdev_test_metric" => stdev(te_metrics),
    }
  end
end

def best_performance_by_learner stats  
  stats.group_by {|s| s["learner"]}.map do |g_s|
    learner, learner_stats = g_s
    best_parameters = learner_stats.max_by {|l| l["mean_test_metric"]}    
    [learner, best_parameters]
  end.to_h
end

def parameter_search learners, dataset, folds = 5

  metric = AUCMetric.new  
  stats = cross_validation_model_performance dataset, folds, learners, metric
  best_by_learner = best_performance_by_learner stats  
    summary = Hash.new
    best_by_learner.each_key do |k|
        summary[k] = best_by_learner[k].clone
        summary[k].delete "trained_model"
    end
  puts JSON.pretty_generate(summary)

  assert_equal learners.size, stats.size
  assert_true(stats.all? {|s| a = s["mean_train_metric"]; a >= 0.0 and a <= 1.0}, "0 <= Train AUC <= 1")
  assert_true(stats.all? {|s| a = s["mean_test_metric"]; a >= 0.0 and a <= 1.0}, "0 <= Train AUC <= 1")
  
  stats.map! {|s| t = s.clone; t.delete "trained_model"; t}
  df = Daru::DataFrame.new(stats) 
    
  return [df, best_by_learner]
end

### ADD YOUR CODE AFTER THIS LINE ###

def mean x
  sum=0
  x.each do |item|
    sum+=item.to_f
  end
  return sum.to_f/(x.size)
end

def stdev x
  sum=0
  mean1=mean(x)
  x.each do |item|
    sum+=(item-mean1)**2
  end
  return (sum.to_f/(x.size-1))**0.5
end


class AUCMetric 
  include Metric
  def roc_curve(scores)
    print "NEw3 it's here in auc metric"
    totalP = 0.0
    totalN = 0.0
      
    scores.each do |score|
      if score[1]==1
        totalP+=1.0
      end
        
      if score[1]<1
        totalN+=1.0
      end
        
    end

    falseP = totalN 
    trueP = totalP
      
    fp=[1.0]
    tp=[1.0]
      
    sorted= scores.sort_by {|score| score[0]}
    auc = 0.0

    sorted.each do |score|
      if score[1]!=1.0
        falseP-=1.0
      end
        
      if score[1]==1.0
        trueP-=1.0
      end
        
      fpratio=falseP/totalN
      tpratio=trueP/totalP
      
      auc += -0.5*(tpratio + tp.last)*(fpratio - fp.last)
      fp << fpratio
      tp << tpratio
        
    end
    return [fp, tp, auc]
  end
  
  def apply scores
    fp, tp, auc = roc_curve scores
    return auc
  end
  
end
 





def cross_validate dataset, folds, &block
  examples = dataset["data"]
  slice=(dataset["data"].size)/folds

  folds.times do |fold|

    train_data = dataset.clone
    train_data["data"] = train_data["data"]-train_data["data"][fold*slice..((fold+1)*slice-1)]
    
    test_data = dataset.clone
    test_data["data"] = test_data["data"][fold*slice..((fold+1)*slice-1)]
    
    ## Call the callback like this:
    yield train_data, test_data, fold
  end
end