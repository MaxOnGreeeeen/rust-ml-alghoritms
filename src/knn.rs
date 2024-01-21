use std::error::Error;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::SystemTime;
use rand::{Rng, thread_rng};

const DATASET_FILE_PATH: &str = "datasets/diabetes.csv";
const CLASSES: usize = 2;
const THREADS: usize = 4;

#[derive(Clone, Debug)]
pub struct Point {
    x: f32,
    y: f32,
    distance: f32,
    class: i32,
}

pub type Arr = [i32; CLASSES];

pub fn resolve_data_ranges(points: &Vec<Point>) -> [Vec<Point>; THREADS] {
    let range_points_amount = (points.len() as i32 / THREADS as i32) as f32;
    const VAL: Vec<Point> = Vec::new();
    let mut chunks: [Vec<Point>; THREADS] = [VAL; THREADS];

    let mut start = 0;
    let mut end = 0;
    for range_idx in 0..THREADS {
        if range_idx == THREADS - 1 {
            end = points.len() as i32 - 1;
            chunks[range_idx] = points[start as usize..end as usize].to_vec();
            continue;
        }
        end = start + range_points_amount as i32;
        chunks[range_idx] = points[start as usize..end as usize].to_owned();
        start = end + 1;
    }
    return chunks;
}

pub fn read_dataset(file_path: &str) -> Result<Vec<Point>, Box<dyn Error>> {
    let reader = csv::Reader::from_path(file_path);
    let mut points: Vec<Point> = vec![];
    for (_record_id, result) in reader?.records().enumerate() {
        let record = result?;
        let x: Result<f32, _> = record.get(1).unwrap().parse();
        let y: Result<f32, _> = record.get(4).unwrap().parse();
        let class: Result<i32, _> = record.get(8).unwrap().parse();
        match (x, y, class) {
            (Ok(x), Ok(y), Ok(class)) => {
                points.push(Point { x, y, distance: 0.0, class });
            }
            error => {
                println!("Error: Failed to parse x and/or y values: {:?}", error);
            }
        }
    }
    println!("Parsed dataset with: {:?} features", points.len());
    Ok(points)
}

pub fn distance<'a>(point_a: &'a Point, point_b: &'a Point) -> f32 {
    ((point_a.x - point_b.x).powi(2) + (point_a.y - point_b.y).powi(2)).sqrt()
}

pub fn classify_point(points: &mut Vec<Point>, k_neighbours: i32, test_point: &Point) -> i32 {
    points.into_iter().enumerate().for_each(|(point_id, point)| {
        point.distance = distance(point, test_point);
    });
    points.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

    let mut classes_vec: Vec<Arr> = vec![[0, 0], [0, 1]];
    for point_index in 0..k_neighbours {
        classes_vec[points[point_index as usize].class as usize][0] += 1;
    }
    return classes_vec.iter().max_by(|class_a, class_b| class_a[0].partial_cmp(&class_b[0]).unwrap()).unwrap()[1];
}

pub fn knn_thread(points: &mut Vec<Point>, test_point: &Point, classes_vec: &Arc<Mutex<Vec<Arr>>>, k_neighbours: &i32) {
    points.into_iter().enumerate().for_each(|(point_id, point)| {
        point.distance = distance(point, test_point);
    });
    points.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
    for point_index in 0..*k_neighbours {
        classes_vec.lock().unwrap()[points[point_index as usize].class as usize][0] += 1;
    }
}

pub fn normalize(points: &mut Vec<Point>) {
    let param_min_x = points.iter().min_by(|point_a, point_b| point_a.x.partial_cmp(&point_b.x).unwrap()).unwrap().x;
    let param_min_y = points.iter().min_by(|point_a, point_b| point_a.y.partial_cmp(&point_b.y).unwrap()).unwrap().y;

    let param_max_x = points.iter().max_by(|point_a, point_b| point_a.x.partial_cmp(&point_b.x).unwrap()).unwrap().x;
    let param_max_y = points.iter().max_by(|point_a, point_b| point_a.y.partial_cmp(&point_b.y).unwrap()).unwrap().y;

    for point in points.iter_mut() {
        point.x = ((point.x - param_min_x) / (param_max_x - param_min_x));
        point.y = ((point.y - param_min_y) / (param_max_y - param_min_y));
    }
}

pub fn get_precision_metric(points: &mut Vec<Point>) -> f32 {
    let mut true_positive = 0;
    let mut false_positive = 0;
    let mut cloned_points = points.clone(); // Clone the points vector
    cloned_points.iter_mut().for_each(|point| {
        let result = knn_parallel(points, Some(&point.clone()));
        if point.class == 1 && result == 1 {
            true_positive += 1;
        } else if point.class == 0 && result == 1 {
            false_positive += 1;
        }
    });
    return (true_positive as f32) / ((true_positive + false_positive) as f32);
}

pub fn get_accuracy_metric(points: &mut Vec<Point>) -> f32 {
    let mut true_positive = 0;
    let mut true_negative = 0;
    let mut false_negative = 0;
    let mut false_positive = 0;
    let mut cloned_points = points.clone(); // Clone the points vector
    cloned_points.iter_mut().for_each(|point| {
        let result = knn_parallel(points, Some(&point.clone()));
        if point.class == 0 {
            if result == point.class {
                true_negative += 1;
            } else {
                false_negative += 1;
            }
        }
        if point.class == 1 {
            if result == point.class {
                true_positive += 1;
            } else {
                false_positive += 1;
            }
        }
    });
    return (true_positive + true_negative) as f32 / (true_positive + false_positive + false_negative + true_negative) as f32;
}

pub fn get_recall_metric(points: &mut Vec<Point>) -> f32 {
    let mut true_positive = 0;
    let mut false_negative = 0;
    let mut cloned_points = points.clone();
    cloned_points.iter_mut().for_each(|point| {
        let result = knn_parallel(points, Some(point));
        if point.class == 1 {
            if result == 1 {
                true_positive += 1;
            } else {
                false_negative += 1;
            }
        }
    });
    return true_positive as f32 / (true_positive + false_negative) as f32;
}

pub fn benchmark_knn() {
    let mut points = read_dataset(DATASET_FILE_PATH).unwrap();
    normalize(&mut points);

    let mut rng = thread_rng();
    let now = SystemTime::now();
    let points_len = points.len();
    let test_point = points.get_mut(rng.gen_range(0..points_len)).unwrap().clone();

    let predicted_point_class = classify_point(&mut points, 10, &test_point);

    println!("Time elapsed: {}", now.elapsed().unwrap().as_millis());
    println!("Test point: {:?}", &test_point);
    println!("Predicted point class: {:?}", predicted_point_class);
}

pub fn knn_parallel(points: &mut Vec<Point>, test_point: Option<&Point>) -> i32 {
    let points_len = points.len();

    let mut rng = thread_rng();

    let test_point_resolved = match test_point {
        Some(test_point) => test_point.clone(),
        None => points.get_mut(rng.gen_range(0..points_len)).unwrap().clone(),
    };

    let test_point = Arc::new(test_point_resolved);
    let chunks = resolve_data_ranges(&points);
    let classes_vec: Arc<Mutex<Vec<Arr>>> = Arc::new(Mutex::new(vec![[0, 0], [0, 1]]));
    let mut threads = Vec::with_capacity(THREADS);

    chunks.into_iter().for_each(|chunk| {
        let cloned_classes = Arc::clone(&classes_vec);
        let test_point_clone = Arc::clone(&test_point);
        threads.push(thread::spawn(move || {
            return knn_thread(&mut chunk.to_vec(), &test_point_clone, &cloned_classes, &10);
        }));
    });

    threads.into_iter().for_each(|thread| {
        thread.join().unwrap();
    });

    let predicted_point_class = classes_vec.lock().unwrap().iter().max_by(|class_a, class_b| class_a[0].partial_cmp(&class_b[0]).unwrap()).unwrap()[1];

    return predicted_point_class;
}

pub fn benchmark_knn_parallel() {
    let mut points = read_dataset(DATASET_FILE_PATH).unwrap();
    let points_len = points.len();
    let mut rng = thread_rng();

    normalize(&mut points);

    let now = SystemTime::now();
    let test_point = points.get_mut(rng.gen_range(0..points_len)).unwrap().clone();
    let predicted_class = knn_parallel(&mut points, None);

    println!("Time elapsed: {}", now.elapsed().unwrap().as_millis());
    println!("Test point: {:?}", &test_point);
    println!("Predicted point class: {:?}", predicted_class);

    println!("Precision metric: {}", get_precision_metric(&mut points));
    println!("Recall metric: {}", get_recall_metric(&mut points));
    println!("Accuracy: {}", get_accuracy_metric(&mut points));
}