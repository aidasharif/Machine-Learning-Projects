require 'open-uri'
require 'json'
require 'daru'
require 'distribution'
require 'sqlite3'
require 'test/unit/assertions'

include Test::Unit::Assertions

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

