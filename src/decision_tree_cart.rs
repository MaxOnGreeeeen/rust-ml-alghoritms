use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::thread;
use std::thread::JoinHandle;
use std::time::SystemTime;
use rand::{Rng};

const LOC_DATASET_FILE_PATH: &str = "datasets/diabetes.csv";

#[derive(Clone, Debug)]
pub struct Feature {
    index: usize,
    values: Vec<FeatureType>,
}

impl Feature {
    fn of_features_array(
        values: &DataSet<FeatureType>,
        start_index: Option<usize>,
    ) -> Vec<Feature> {
        let shift = start_index.unwrap_or(0);
        let features_array = values.iter().enumerate().map(|(index, item)| {
            return Feature {
                index: index + shift,
                values: item.clone(),
            };
        }).collect();
        return features_array;
    }
}

mod helpers {
    use std::error::Error;
    use std::num::ParseFloatError;
    use crate::decision_tree_cart::{DataSet, FeatureType};

    fn validate_parsed_value(match_value: Result<FeatureType, ParseFloatError>) -> Option<FeatureType> {
        return match match_value {
            Ok(match_value) => {
                Some(match_value)
            }
            Err(error) => {
                println!("Error: Failed to parse record: {:?}", error);
                None
            }
        };
    }

    pub fn read_dataset(
        file_path: &str,
        include_args_range: [i32; 2],
        target_attribute: i32,
    ) -> Result<(DataSet<FeatureType>, Vec<FeatureType>, Vec<String>), Box<dyn Error>> {
        let mut reader = csv::Reader::from_path(file_path)?;
        let mut res_vec = vec![];
        let mut target_vec = vec![];

        let binding = reader.headers()?.clone();
        let mut headers = binding.into_iter().collect::<Vec<_>>();

        for (_record_id, result) in reader.records().enumerate() {
            let record = result?;
            let mut records: Vec<FeatureType> = vec![];
            for (column_index, value) in record.iter().enumerate() {
                if column_index >= include_args_range[0] as usize
                    && column_index <= include_args_range[1] as usize
                {
                    let record_column_val = value.parse::<FeatureType>();
                    if let Some(validated) = validate_parsed_value(record_column_val) {
                        records.push(validated);
                    }
                }
                if column_index == target_attribute as usize {
                    let record_column_val = value.parse::<FeatureType>();
                    if let Some(validated) = validate_parsed_value(record_column_val) {
                        target_vec.push(validated);
                    }
                }
            }
            res_vec.push(records);
        }
        let mut headers_to_string = vec![];
        ;
        headers.clone()
            .iter()
            .for_each(|title| {
                headers_to_string.push(title.to_string());
            });
        println!("Parsed dataset with: {:?} features", res_vec.len());
        Ok((res_vec, target_vec, headers_to_string))
    }
}

type DataSet<T> = Vec<Vec<T>>;

// Поддерживается только числовой тип данных
type FeatureType = f32;

#[derive(Debug, Clone)]
enum Node {
    Leaf {
        outcome: f32,
    },
    Interior {
        // Пороговое значение (threshold)
        value: f32,
        // Индекс фичи
        feature_index: usize,
        // Имя фичи
        feature_name: String,
        left_child: Box<Node>,
        right_child: Box<Node>,
    },
}

pub struct DecisionTreeClassifier {
    // Максимальное количество слоёв дерева
    max_depth: i32,
    min_samples_split: i32,
    // Не используется
    min_samples_leaf: i32,
    root: Option<Node>,
}

impl DecisionTreeClassifier {
    fn new(max_depth: i32, min_samples_split: i32, min_samples_leaf: i32) -> Self {
        return Self {
            max_depth,
            min_samples_split,
            min_samples_leaf,
            root: None,
        };
    }

    // Обучение модели
    pub fn fit(&mut self,
               data: &DataSet<FeatureType>,
               target: &Vec<f32>,
               _features_names: &Vec<String>) {
        let features = Feature::of_features_array(&data, Some(0));
        self.root = Some(Self::build_tree(&features, target, 1, self.max_depth, self.min_samples_split, self.min_samples_leaf));

    }

    // Параллельная версия алгоритма
    pub fn fit_parallel(&mut self,
                        data: &DataSet<FeatureType>,
                        target: &Vec<f32>,
                        _features_names: &Vec<String>,
                        num_threads: usize,
    ) {
        let features = Feature::of_features_array(&data, Some(0));
        let threads_counter: Arc<Mutex<i32>> = Arc::new(Mutex::new(1));
        self.root = Some(Self::build_tree_parallel(&features, target, 1, self.max_depth, self.min_samples_split, self.min_samples_leaf,num_threads, &threads_counter));
        drop(threads_counter);
    }

    fn build_tree_parallel(data: &Vec<Feature>,
                           target: &Vec<f32>,
                           depth: i32,
                           max_depth: i32,
                           min_samples_split: i32,
                           min_samples_leaf: i32,
                           num_threads_max: usize,
                           threads_counter: &Arc<Mutex<i32>>,
    ) -> Node {
        let outcome = Self::get_outcome(data, target);
        if depth > max_depth || data.len() < min_samples_split as usize {
            return Node::Leaf {
                outcome
            };
        }

        let (_, best_feature_index, best_threshold) = Self::calculate_split(data, target);
        let (left, right) = Self::split_data(best_feature_index as i32, best_threshold, data);
        if left.len() > 0 && right.len() > 0 {
            let mut threads_pool: Vec<JoinHandle<Node>> = Vec::with_capacity(2);
            let mut nodes_result = vec![];

            [left, right].into_iter().for_each(|item| {
                let target_copy = target.clone();
                let thread_counter_value = threads_counter.lock().unwrap().clone();
                let thread_counter_copy = Arc::clone(threads_counter);

                if thread_counter_value < num_threads_max as i32 {
                    let mut threads_counter_lock = threads_counter.lock().unwrap();
                    *threads_counter_lock += 1;

                    threads_pool.push(thread::spawn(move || {
                        return Self::build_tree_parallel(&item.clone(), &target_copy, depth + 1, max_depth, min_samples_split, min_samples_leaf, num_threads_max, &thread_counter_copy)
                    }));
                } else {
                    nodes_result.push(Self::build_tree(&item, &target_copy, depth + 1, max_depth, min_samples_split, min_samples_leaf))
                };
            });

            threads_pool.into_iter().for_each(|handle| {
                let handle_result = handle.join().unwrap();
                let mut threads_counter_lock = threads_counter.lock().unwrap();
                *threads_counter_lock -= 1;

                nodes_result.push(handle_result);
            });

            let left_child_clone = nodes_result[0].clone();
            let right_child_clone =  nodes_result[1].clone();

            return Node::Interior {
                value: best_threshold,
                feature_index: best_feature_index,
                feature_name: "Not implemented".to_owned(),
                left_child: Box::new(left_child_clone),
                right_child: Box::new(right_child_clone),
            };
        }

        return Node::Leaf {
            outcome,
        };
    }

    fn build_tree(data: &Vec<Feature>,
                  target: &Vec<f32>,
                  depth: i32,
                  max_depth: i32,
                  min_samples_split: i32,
                  min_samples_leaf: i32,
    ) -> Node {
        let outcome = Self::get_outcome(data, target);
        if depth > max_depth || data.len() < min_samples_split as usize {
            return Node::Leaf {
                outcome
            };
        }

        let (_, best_feature_index, best_threshold) = Self::calculate_split(data, target);
        let (left, right) = Self::split_data(best_feature_index as i32, best_threshold, data);
        if left.len() > 0 && right.len() > 0 {
            let left_child = Self::build_tree(&left, target, depth + 1, max_depth, min_samples_split, min_samples_leaf);
            let right_child = Self::build_tree(&right, target, depth + 1, max_depth, min_samples_split, min_samples_leaf);

            return Node::Interior {
                value: best_threshold,
                feature_index: best_feature_index,
                feature_name: "Not implemented".to_owned(),
                left_child: Box::new(left_child),
                right_child: Box::new(right_child),
            };
        }

        return Node::Leaf {
            outcome,
        };
    }

    fn calculate_split(data: &Vec<Feature>, target: &Vec<f32>) -> (f32, usize, FeatureType) {
        let mut best_gini = f32::INFINITY;
        let mut best_feature_index = 0;
        let mut best_threshold = 0.0;

        for index in 0..data.first().unwrap().values.len() {
            data.iter().for_each(|item| {
                let (left, right) = Self::split_data(index as i32, item.values[index], data);
                let gini = Self::proxy_gini_impurity(&left, &right, &Self::filter_uniq(&target), target);

                if gini < best_gini {
                    best_gini = gini;
                    best_feature_index = index;
                    best_threshold = item.values[index];
                }
            });
        }

        return (best_gini, best_feature_index, best_threshold);
    }

    fn filter_uniq(vec: &Vec<f32>) -> Vec<f32> {
        let unique_set: HashSet<i32> = vec.iter().map(|&x| x as i32).collect();
        let unique_vec: Vec<f32> = unique_set.iter().map(|&x| x as f32).collect();
        unique_vec
    }

    fn get_outcome(data: &Vec<Feature>,
                   target: &Vec<f32>,
    ) -> f32 {
        let mut features_targets = Self::get_target_features_classes(data, target);

        let mut frequencies_hashmap: HashMap<String, i32> = HashMap::new();
        features_targets.iter().for_each(|item| {
            let mut feature_target: i32;
            match frequencies_hashmap.get(&item.to_string()) {
                Some(value) => feature_target = *value,
                None => feature_target = 0
            }
            frequencies_hashmap.insert(item.to_string(), feature_target + 1);
        });

        return frequencies_hashmap
            .iter()
            .max_by(|a, b| a.1.cmp(&b.1))
            .map(|(k, _v)| k).unwrap().parse::<f32>().unwrap();
    }


    // Функция для разделения данных на основе лучшего признака и порога
    fn split_data(
        attribute_index: i32,
        value: FeatureType,
        data: &Vec<Feature>,
    ) -> (Vec<Feature>, Vec<Feature>) {
        let mut right: Vec<Feature> = Vec::new();
        let mut left: Vec<Feature> = Vec::new();
        for item in data.iter() {
            if item.values[attribute_index as usize] < value {
                left.push(item.clone());
            } else {
                right.push(item.clone());
            }
        }
        (right, left)
    }

    pub fn predict(&self, train_data: &Vec<FeatureType>) -> f32 {
        return self.query_tree(&self.root.as_ref().unwrap(), train_data);
    }

    // Возвращает предсказание класса для переданной фичи
    fn query_tree(&self, ref node: &Node, train_data: &Vec<FeatureType>) -> f32 {
        return match *node {
            Node::Interior {
                value,
                feature_index,
                feature_name: _feature_name,
                ref left_child,
                ref right_child,
            } => {
                return if train_data[*feature_index] < *value {
                    self.query_tree(left_child, train_data)
                } else {
                    self.query_tree(right_child, train_data)
                };
            }
            Node::Leaf { outcome } => *outcome,
        };
    }

    fn get_target_features_classes(data: &Vec<Feature>, target: &Vec<f32>) -> Vec<f32> {
        return data.iter().map(|item| {
            return target[item.index];
        }).collect();
    }

    // Возвращает индекс чистоты разбиения
    fn proxy_gini_impurity(
        left: &Vec<Feature>,
        right: &Vec<Feature>,
        classes: &Vec<f32>,
        target: &Vec<f32>,
    ) -> f32 {
        let n_instances = (left.len() + right.len()) as f32;

        let left_child_proportion = left.len() as f32 / n_instances;
        let right_child_proportion = right.len() as f32 / n_instances;

        let left_group_features_targets = Self::get_target_features_classes(left, target);
        let right_group_features_targets = Self::get_target_features_classes(right, target);

        let mut left_child_gini = 0.0;
        let mut right_child_gini = 0.0;

        classes.iter().for_each(|class| {
            let left_p = left_group_features_targets.iter().filter(|&value| value == class).count() as f32 / left.len() as f32;
            let right_p = right_group_features_targets.iter().filter(|&value| value == class).count() as f32 / right.len() as f32;

            left_child_gini += 1.0 - left_p.powi(2);
            right_child_gini += 1.0 - right_p.powi(2);
        });

        let gini_impurity = left_child_proportion * left_child_gini + right_child_proportion * right_child_gini;
        gini_impurity
    }
}

mod bench {
    use crate::decision_tree_cart::{DecisionTreeClassifier, Feature};

    pub fn get_precision_metric(decision_tree: &DecisionTreeClassifier, data: &Vec<Feature>, target: &Vec<f32>) -> f32 {
        let mut true_positive = 0;
        let mut false_positive = 0;
        let mut cloned_data = data.clone();
        cloned_data.iter_mut().for_each(|item| {
            let result = decision_tree.predict(&item.values);
            let item_class = target[item.index];

            if item_class == 1.0 && result == 1.0 {
                true_positive += 1;
            } else if item_class == 0.0 && result == 1.0 {
                false_positive += 1;
            }
        });
        return (true_positive as f32) / ((true_positive + false_positive) as f32);
    }

    pub fn get_accuracy_metric(decision_tree: &DecisionTreeClassifier, data: &Vec<Feature>, target: &Vec<f32>) -> f32 {
        let mut true_positive = 0;
        let mut true_negative = 0;
        let mut false_negative = 0;
        let mut false_positive = 0;
        let mut cloned_data = data.clone();
        cloned_data.iter_mut().for_each(|item| {
            let result = decision_tree.predict(&item.values);
            let item_class = target[item.index];
            if item_class == 0.0 {
                if result == item_class {
                    true_negative += 1;
                } else {
                    false_negative += 1;
                }
            }
            if item_class == 1.0 {
                if result == item_class {
                    true_positive += 1;
                } else {
                    false_positive += 1;
                }
            }
        });
        return (true_positive + true_negative) as f32 / (true_positive + false_positive + false_negative + true_negative) as f32;
    }

    pub fn get_recall_metric(decision_tree: &DecisionTreeClassifier, data: &Vec<Feature>, target: &Vec<f32>) -> f32 {
        let mut true_positive = 0;
        let mut false_negative = 0;
        let mut cloned_data = data.clone();
        cloned_data.iter_mut().for_each(|item| {
            let result = decision_tree.predict(&item.values);
            let item_class = target[item.index];

            if item_class == 1.0 {
                if result == 1.0 {
                    true_positive += 1;
                } else {
                    false_negative += 1;
                }
            }
        });
        return true_positive as f32 / (true_positive + false_negative) as f32;
    }

    pub fn get_metrics(decision_tree_classifier: &DecisionTreeClassifier, data: &Vec<Feature>, target: &Vec<f32>){
        let accuracy = get_accuracy_metric(&decision_tree_classifier, &data, &target);
        let precision = get_precision_metric(&decision_tree_classifier, &data, &target);
        let recall = get_recall_metric(&decision_tree_classifier, &data, &target);

        println!("Accuracy result {:?}", accuracy);
        println!("Precision result {:?}", precision);
        println!("Recall result {:?}", recall);
    }
}

pub fn benchmark_cart_parallel() {
    let (x,
        target,
        labels
    ) = helpers::read_dataset(LOC_DATASET_FILE_PATH, [1, 7], 8).unwrap();
    let train_end_index = (x.len() as f32 * 0.8) as usize;

    let train_data_slice: DataSet<FeatureType> = x[0..train_end_index].to_vec();
    let test_data_slice = Feature::of_features_array(&x[train_end_index..x.len()].to_vec(), Some(train_end_index));

    let mut decision_tree_classifier =
        DecisionTreeClassifier::new(10, 10, 10);

    let now = SystemTime::now();
    decision_tree_classifier.fit_parallel(&train_data_slice, &target, &labels, 16);
    println!("Time elapsed parallel: {} ms", now.elapsed().unwrap().as_millis());

    bench::get_metrics(&decision_tree_classifier, &test_data_slice, &target);
}

pub fn benchmark_cart_linear() {
    let (x,
        target,
        labels
    ) = helpers::read_dataset(LOC_DATASET_FILE_PATH, [1, 7], 8).unwrap();
    let train_end_index = (x.len() as f32 * 0.8) as usize;

    let train_data_slice: DataSet<FeatureType> = x[0..train_end_index].to_vec();
    let test_data_slice = Feature::of_features_array(&x[train_end_index..x.len()].to_vec(), Some(train_end_index));

    let mut decision_tree_classifier =
        DecisionTreeClassifier::new(10, 10, 10);

    let now = SystemTime::now();
    decision_tree_classifier.fit(&train_data_slice, &target, &labels);
    println!("Time elapsed linear: {} ms", now.elapsed().unwrap().as_millis());

    bench::get_metrics(&decision_tree_classifier, &test_data_slice, &target);
}