use std::{f64, fmt};
use rand::prelude::*;
use plotters::prelude::*;
use std::{error::Error};
use std::collections::HashMap;
use std::sync::{Arc, Barrier, mpsc, Mutex};
use std::sync::mpsc::{Receiver, Sender};
use std::time::SystemTime;
use csv;
use std::thread;

// Algorithm description:
// initialize k centroids
// Distance calculation
// Assign each point closest centroid
// Compute the new centroids for each cluster

const DATASET_FILE_PATH: &str = "datasets/Cust_Segmentation.csv";
const PLOT_OUTPUT_FILE_PATH: &str = "target/kmeans.png";
const CLUSTERS: usize = 3;
const MAX_ITERATIONS: i32 = 10;
const THREADS_NUM: i32 = 16;

#[derive(Clone, Debug)]
pub struct Point {
    id: i32,
    x: f32,
    y: f32,
    cluster: i32,
    min_dist: f64,
}

pub type ArcMutex<T> = Arc<Mutex<T>>;

trait IPointDistance {
    fn distance(&self, point: Point) -> f32;
}

impl Point {
    fn new() -> Self {
        Self { id: 0, x: 0.0, y: 0.0, cluster: -1, min_dist: f64::INFINITY }
    }

    fn with_default_coord(id: i32, x: f32, y: f32) -> Self {
        Point { id, x, y, cluster: -1, min_dist: f64::INFINITY }
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Point {{ id: {}, x: {:?}, y: {:?}, cluster: {:?}, min_dist: {:?} }}",
            self.id, self.x, self.y, self.cluster, self.min_dist
        )
    }
}

impl IPointDistance for Point {
    fn distance(&self, point: Point) -> f32 {
        let x = self.x;
        let y = self.y;
        let point_x = point.x;
        let point_y = point.y;

        ((point_x - x).powi(2) + (point_y - y).powi(2)).sqrt()
    }
}

#[derive(Debug)]
pub struct IterCentroid {
    sum_x: Vec<f32>,
    sum_y: Vec<f32>,
    n_points: Vec<i32>
}

#[derive(Copy, Clone, Debug)]
pub struct Range {
    id: i32,
    start: i32,
    end: i32,
}

pub struct KMeans {
    points: Vec<Point>,
    centroids: Vec<Point>,
    clusters: usize,
    iterations: i32,
    threads: i32,
    ranges: Vec<Range>,
    chunks: Vec<Vec<Point>>,
}

pub trait IKMeans {
    fn assign_random_centroids(&mut self);
    fn read_dataset(file_path: &str) -> Result<Vec<Point>, Box<dyn Error>>;
    fn k_means_multi_threaded(&mut self);
    fn k_means_single_threaded(&mut self);
    fn init_plot(&mut self);
    fn resolve_data_ranges(&mut self);
    fn k_means_thread(max_iterations: i32, centroids: &Arc<Mutex<Vec<Point>>>, range: &mut Vec<Point>, barrier: &Arc<Barrier>, sender: &Arc<Sender<IterCentroid>>) -> Vec<Point>;
}

impl KMeans {
    fn new(clusters: usize, iterations: i32) -> Self {
        Self { points: Vec::new(), centroids: Vec::new(), clusters, iterations, threads: 0, ranges: vec![], chunks: vec![] }
    }
}

impl IKMeans for KMeans {
    fn assign_random_centroids(&mut self) {
        let mut rng = thread_rng();

        for cluster in 0..self.clusters {
            let mut random_point = self.points[rng.gen_range(0..self.points.len())].clone();
            random_point.cluster = cluster as i32;
            self.centroids.push(random_point);
        }
    }

    fn read_dataset(file_path: &str) -> Result<Vec<Point>, Box<dyn Error>> {
        let reader = csv::Reader::from_path(file_path);
        let mut points: Vec<Point> = vec![];
        for (record_id, result) in reader?.records().enumerate() {
            let record = result?;
            let x: Result<f32, _> = record.get(1).unwrap().parse();
            let y: Result<f32, _> = record.get(4).unwrap().parse();
            match (x, y) {
                (Ok(x_val), Ok(y_val)) => {
                    points.push(Point::with_default_coord(record_id as i32, x_val, y_val));
                }
                error => {
                    println!("Error: Failed to parse x and/or y values: {:?}", error);
                }
            }
        }
        println!("Parsed dataset with: {:?} features", points.len());
        Ok(points)
    }

    fn k_means_multi_threaded(&mut self) {
        let now = SystemTime::now();
        self.assign_random_centroids();

        let max_iterations = self.iterations;

        let centroids = Arc::new(Mutex::new(self.centroids.clone()));
        let points_chunks = self.chunks.clone();

        let (sender, receiver): (Sender<IterCentroid>, Receiver<IterCentroid>) = mpsc::channel();
        let sender = Arc::new(sender);

        let barrier = Arc::new(Barrier::new(self.threads as usize));
        let mut threads = Vec::with_capacity(self.threads as usize);

        for mut data_range in points_chunks {
            let centroids_clone = Arc::clone(&centroids);
            let cloned_barrier = Arc::clone(&barrier);
            let sender_cloned = Arc::clone(&sender);
            threads.push(thread::spawn(move || {
                return Self::k_means_thread(max_iterations, &centroids_clone, &mut data_range, &cloned_barrier, &sender_cloned);
            }));
        }

        // Drop sender value
        drop(sender);

        let mut iter_centroids_global_cache: Arc<Mutex<Vec<IterCentroid>>> = Arc::new(Mutex::new(vec![]));
        loop {
            match receiver.recv() {
                Ok(received) => {
                    iter_centroids_global_cache.lock().unwrap().push(received);
                    let cache_len = iter_centroids_global_cache.lock().unwrap().len();

                    // Находим среднее значение среди вычисленных после итерации в каждом потоке центроидов
                    //
                    // Формула: centroid.x = Sum(sum(xi)) / Sum(n_points(i))
                    //          centroid.y = Sum(sum(yi)) / Sum(n_points(i))
                    if cache_len == (self.threads - 1) as usize {
                        let iter_centroids_global_cache_clone = Arc::clone(&iter_centroids_global_cache);
                        let clusters = centroids.lock().unwrap().len();
                        let mut sum_x = vec![0.0; clusters];
                        let mut sum_y = vec![0.0; clusters];
                        let mut n_points = vec![0; clusters];

                        for iter_centroids in iter_centroids_global_cache_clone.lock().unwrap().iter() {
                            for (centroid_id, sum) in iter_centroids.sum_x.iter().enumerate() {
                                sum_x[centroid_id] += sum;
                            }

                            for (centroid_id, sum) in iter_centroids.sum_y.iter().enumerate() {
                                sum_y[centroid_id] += sum;
                            }

                            for (centroid_id, n) in iter_centroids.n_points.iter().enumerate() {
                                n_points[centroid_id] += n;
                            }
                        }

                        let mut centroids_lock = centroids.lock().unwrap();
                        for (centroid_id, centroid) in centroids_lock.iter_mut().enumerate() {
                            centroid.x = sum_x[centroid_id] / n_points[centroid_id] as f32;
                            centroid.y = sum_y[centroid_id] / n_points[centroid_id] as f32;
                        }

                        // Clear the cache
                        iter_centroids_global_cache.lock().unwrap().clear();
                    }
                }
                _ => {
                    break;
                }
            }
        }

        let mut date_range_collected: Vec<Point> = vec![];
        threads.into_iter().for_each(|thread| {
            date_range_collected.extend(thread.join().unwrap());
        });

        self.points = date_range_collected.clone();
        println!("Time elapsed: {}", now.elapsed().unwrap().as_millis());
    }

    fn k_means_single_threaded(&mut self) {
        let now = SystemTime::now();
        self.assign_random_centroids();

        for _ in 0..self.iterations {
            for (cluster_id, centroid) in self.centroids.iter_mut().enumerate() {
                self.points.iter_mut().for_each(|point| {
                    let dist = centroid.distance(point.clone());
                    if dist < point.min_dist as f32 {
                        point.min_dist = dist as f64;
                        point.cluster = cluster_id as i32;
                    }
                });
            }

            let mut n_points = vec![0; self.clusters];
            let mut sum_x = vec![0.0; self.clusters];
            let mut sum_y = vec![0.0; self.clusters];

            for point in self.points.iter_mut() {
                let cluster_id = point.cluster as usize;
                n_points[cluster_id] += 1;
                sum_x[cluster_id] += point.x;
                sum_y[cluster_id] += point.y;
                point.min_dist = f64::INFINITY;
            }
            for (cluster_id, centroid) in self.centroids.iter_mut().enumerate() {
                let n = n_points[cluster_id];
                centroid.x = sum_x[cluster_id] / n as f32;
                centroid.y = sum_y[cluster_id] / n as f32;
            }
        }
        println!("Time elapsed: {}", now.elapsed().unwrap().as_millis());
    }

    fn init_plot(&mut self) {
        let root = BitMapBackend::new(PLOT_OUTPUT_FILE_PATH, (600, 400)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        let x_lim = 0.0..100.0f32;
        let y_lim = 0.0..450.0f32;

        let mut ctx = ChartBuilder::on(&root)
            .set_label_area_size(LabelAreaPosition::Left, 40) // Put in some margins
            .set_label_area_size(LabelAreaPosition::Right, 40)
            .set_label_area_size(LabelAreaPosition::Bottom, 40)
            .caption("KMeans Demo", ("sans-serif", 25)) // Set a caption and font
            .build_cartesian_2d(x_lim, y_lim)
            .expect("Couldn't build our ChartBuilder");

        ctx.configure_mesh().draw().unwrap();
        let root_area = ctx.plotting_area();

        const AVAILABLE_COLORS: i32 = 6;
        let available_colors = HashMap::from([
            (0, &RED),
            (1, &GREEN),
            (2, &BLUE),
            (3, &MAGENTA),
            (4, &CYAN),
            (5, &YELLOW),
        ]);

        for point in self.points.iter() {
            let point = match point.cluster {
                0..=AVAILABLE_COLORS => Circle::new(
                    (point.x, point.y),
                    3,
                    ShapeStyle::from(available_colors.get(&point.cluster).unwrap()).filled(),
                ),
                _ => Circle::new(
                    (point.x, point.y),
                    3,
                    ShapeStyle::from(&BLACK).filled(),
                ),
            };

            root_area
                .draw(&point)
                .expect("An error occurred while drawing the point!");
        }
    }

    fn resolve_data_ranges(&mut self) {
        let range_points_amount = (self.points.len() as i32 / self.threads) as f32;
        let mut start = 0;
        let mut end = 0;

        for range in 0..self.threads {
            if range == self.threads - 1 {
                end = self.points.len() as i32 - 1;
                let range = Range { start, end, id: range };
                self.ranges.push(range);
                self.chunks.push(self.points[range.start as usize..range.end as usize].to_owned());
                return;
            }
            end = start + range_points_amount as i32;
            let range = Range { start, end, id: range };
            self.ranges.push(range);
            self.chunks.push(self.points[range.start as usize..range.end as usize].to_owned());
            start = end + 1;
        }
        self.ranges.iter().for_each(|range| {
            println!("{:?}", range);
        });
    }

    fn k_means_thread(
        max_iterations: i32,
        centroids: &Arc<Mutex<Vec<Point>>>,
        range: &mut Vec<Point>,
        barrier: &Arc<Barrier>,
        sender: &Arc<Sender<IterCentroid>>
    ) -> Vec<Point> {
        let mut range_local: Vec<Point> = range.to_vec();

        let clusters = centroids.lock().unwrap().len();
        let mut iter_counter = 0;

        let mut n_points = vec![0; clusters];
        let mut sum_x = vec![0.0; clusters];
        let mut sum_y = vec![0.0; clusters];

        while iter_counter < max_iterations {
            range_local.iter_mut().for_each(|mut point| {
                centroids.lock().unwrap().iter().enumerate().for_each(|(cluster_id, centroid)| {
                    let dist = centroid.distance(point.clone()) as f64;
                    if dist < point.min_dist {
                        point.min_dist = dist;
                        point.cluster = cluster_id as i32;
                    }
                });
            });

            range_local.iter_mut().enumerate().for_each(|(point_id, mut point)| {
                let cluster_id = point.cluster as usize;
                n_points[cluster_id] += 1;

                sum_x[cluster_id] += point.x;
                sum_y[cluster_id] += point.y;

                point.min_dist = f64::INFINITY;
            });

            barrier.wait();

            sender.send(IterCentroid {
                sum_y: sum_y.clone(),
                sum_x: sum_x.clone(),
                n_points: n_points.clone()
            }).unwrap();

            for i in 0..clusters {
                n_points[i] = 0;
                sum_x[i] = 0.0;
                sum_y[i] = 0.0;
            }

            iter_counter += 1;
        }

        range_local
    }
}

pub fn benchmark_k_means(){
    let mut struct_k_means = KMeans::new(CLUSTERS, MAX_ITERATIONS);

    struct_k_means.points = KMeans::read_dataset(DATASET_FILE_PATH).unwrap();
    struct_k_means.threads = THREADS_NUM;

    struct_k_means.resolve_data_ranges();
    struct_k_means.k_means_multi_threaded();
    struct_k_means.init_plot();
}