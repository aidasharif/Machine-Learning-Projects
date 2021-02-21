require 'open-uri'
require 'json'
require 'daru'
require 'distribution'
require 'sqlite3'
require 'test/unit/assertions'
require 'timeout'

include Test::Unit::Assertions

$dir = "/home/dataset"

def load_db path
  db = SQLite3::Database.new path, results_as_hash: true, readonly: true
  db.execute "pragma temp_store = 1;"
  db.execute "pragma temp_store_directory = '#{$dir}/temp';"
  return db
end

def train_db
  load_db "#{$dir}/credit_risk_data_train.db"
end

def dev_db
  load_db "#{$dir}/credit_risk_data_dev.db"
end

def create_dataset db, sql
  examples = []
  feature_names = Hash.new
  
  db.execute sql do |row|
    features = Hash.new      
    unless row.has_key?("TARGET") and row.has_key?("SK_ID_CURR")
        raise ArgumentError.new("Query must include 'target' and 'sk_id_curr'") 
    end
      
    fields = row.keys.select {|k| k.is_a? String}.map{|k| k.downcase} - ["target", "sk_id_curr"]
    fields.each do |k| 
      v = row[k.upcase]
      next if v.is_a? String and v == ""
      features[k] = v
    end
    fields.each {|k| feature_names[k] = 1}
    
    u = {"label" => row["TARGET"], "id" => row["SK_ID_CURR"]}
    u["features"] = features
    examples << u
  end
  dataset = {
    "features" => feature_names.keys,
    "data" => examples,
    }
  return dataset
end

module Learner  
  attr_reader :parameters
  def name
      self.class.name
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

module FeatureTransformer
    def train dataset
        ## Calculate any statistics
    end

    def apply example_batch
        ## Apply transform to a batch of examples
    end
end

def get_labels_for db, predictions
  ids = predictions.keys.join(", ")
  sql = "select sk_id_curr, target from application_train"
  scores = Array.new
  db.execute(sql) do |row|
      id = row["SK_ID_CURR"].to_i
      unless predictions.has_key? id
          raise ArgumentError.new("There is no prediction for #{id}. Make sure you are not removing any records.") 
      end
    y_hat = predictions[id]
    y = row["TARGET"]
    scores << [y_hat, y]
  end
  return scores
end

module FinalProjectClassifier
  ## Perform any SQL queries / transformations you need to do 
  ## Return a dataset to be used to train models
  ## This may be called multiple times during testing
  def create_training_dataset training_db
    
  end
  
  ## Run whatever SQL queries / transformations you need.
  ## Assume this database is different that the training set
  ## For example, if you are doing any sampling on the training set
  ## don't do it here.
  def create_evaluation_dataset evaluation_db
    
  end
  
  ## Return an array of Learners which be evaluated on 5-fold cross-validation
  def create_learners dataset
    
  end
  
  ## Returns predictions in the correct format
  def create_predictions learner, dataset
    dataset["data"].map do |example|
      score = learner.predict example
      [example["id"], score]
    end.to_h
  end
end

def test_basics e
  assert_not_nil e[:classifier]  
  dev_training_set = e[:classifier].create_training_dataset dev_db()
  
  puts "Dev Set:", dev_training_set["features"], dev_training_set["data"][0]
  assert_true dev_training_set["data"].size > 1, "> 1 examples on dev training set"
  
  dev_evaluation_set = e[:classifier].create_evaluation_dataset dev_db()
  assert_true dev_evaluation_set["data"].size > 1, "> 1 examples on dev evaluation set"
  puts "\nEvaluation Set:", dev_evaluation_set["features"], dev_training_set["data"][0]
  
  small_training_set = dev_training_set.clone
  small_training_set["data"] = small_training_set["data"].sample(100)
  
  dev_learners = e[:classifier].create_learners small_training_set
  assert_not_nil dev_learners
  assert_true dev_learners.size > 0, "At least 1 learner"
  
  dev_learner = dev_learners.first
  dev_learner.train small_training_set
  puts "\nModel trained on dev set:", dev_learner.name
  
  test_example = small_training_set["data"][0]
  puts "\nTesting on", test_example, dev_learner.predict(test_example)
end


def run_cross_validation_performance e
  puts "Creating training dataset"
  training_set = e[:classifier].create_training_dataset train_db()
  puts "Creating learners"
  learners = e[:classifier].create_learners training_set
  
  # Summary contains the cross-validation results
  puts "Running #{e[:folds]}-fold cross validation"
  summary = nil
    
  Timeout::timeout(10800) do
      df, summary = parameter_search learners, training_set, e[:folds]
      e[:summary] = summary
      e[:cross_validation_results] = df      
  end
end

def test_cross_validation_performance e
    summary = e[:summary]
  # Best learner summary contains the cross-validation results for the best learner
  best_learner_name = summary.keys.max_by {|k| summary[k]["mean_test_metric"]}  
  best_learner_summary = summary[best_learner_name]
  
  # Best learner contains a learner trained on the FULL training set"
  e[:best_learner] = best_learner_summary["trained_model"]  
  
  cv_auc = best_learner_summary["mean_test_metric"]
  puts "Best Learner", best_learner_name, "AUC = #{cv_auc}"
  assert_true(cv_auc > e[:min_auc], "Cross-Validation AUC #{cv_auc} > #{e[:min_auc]}")
  assert_true(cv_auc <= e[:max_auc], "Cross-Validation AUC #{cv_auc} <= #{e[:max_auc]}") 
end


def test_evaluation_set_performance e
  puts "Creating evaluation dataset"
  eval_dataset = e[:classifier].create_evaluation_dataset e[:db]  
  puts "Evaluating classifier"
  predictions = e[:classifier].create_predictions e[:best_learner], eval_dataset
  puts predictions.entries[0,5]
  
  #Scores on evaluation database
  puts "Validating predictions against labels from database"
  scores_on_evaluation_set = get_labels_for e[:db], predictions
  assert_equal e[:db_size], scores_on_evaluation_set.size, "Returns a score for every example in evaluation set"
  
  puts "Plotting ROC curve"
  metric = AUCMetric.new
  fp, tp, auc = metric.roc_curve scores_on_evaluation_set
  puts "#{e[:name]} set AUC: #{auc}"
  
  plot_roc_curve(fp, tp, auc).show()  

  assert_equal(e[:db_size] + 1, fp.size, "Get all the points")
  assert_true(auc > e[:min_auc], "#{e[:name]} set AUC: #{auc} > #{e[:min_auc]}")
  assert_true(auc <= e[:max_auc], "#{e[:name]} set AUC: #{auc} <= #{e[:max_auc]}")
end