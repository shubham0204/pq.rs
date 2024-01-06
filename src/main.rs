use std::vec;

use rand::seq::SliceRandom ; 

enum DistanceMetric {
    L2 , 
    Dot
}

struct ProductQuantizer {
    m: usize ,
    ks: usize , 
    d: usize , 
    codewords: Vec<Vec<Vec<f32>>> , 
    metric: DistanceMetric
}

impl ProductQuantizer {

    fn new(
        m: usize ,      //   M
        ks: usize ,     //   Ks
        d: usize ,      //   D
        metric: DistanceMetric
    ) -> Self {
        ProductQuantizer{ m , ks , d , codewords: Vec::new() , metric }
    }

    fn fit(
        self: &mut ProductQuantizer , 
        vectors: Vec<Vec<f32>> , 
        iterations: usize
    ) {
        let ds: usize = self.d / self.m ; 
        self.codewords = Vec::new() ; 
        for m in 0..self.m {
            let mut vectors_sub: Vec<Vec<f32>> = Vec::new() ; 
            for vec in &vectors {
                vectors_sub.push( vec[ (m * ds)..((m+1) * ds) ].to_vec() ) ; 
            }
            self.codewords.push( self.kmeans( vectors_sub , self.ks , iterations ) ) ; 
        }
    }

    fn kmeans(
        self: &ProductQuantizer , 
        data: Vec<Vec<f32>> , 
        k: usize , 
        iter: usize
    ) -> Vec<Vec<f32>> {
        let mut assigned_centroids: Vec<Vec<f32>> = vec![ Vec::new() ; data.len() ] ; 
        let mut centroids: Vec<Vec<f32>> = data.choose_multiple( &mut rand::thread_rng() , k )  
                                            .map( | vec | vec.to_vec() )
                                            .collect() ; 
        let vec_dims: usize = data[0].len() ; 
        for _ in 0..iter {
            for i in 0..data.len() {
                let mut min_centroid_distance: f32 = f32::MAX ; 
                let mut min_centroid: Vec<f32> = centroids[0].clone() ; 
                for centroid in &centroids {
                    let distance: f32 = self.euclid_distance( &data[i] , centroid ) ;
                    if distance < min_centroid_distance {
                        min_centroid_distance = distance ; 
                        min_centroid = centroid.clone() ; 
                    }
                }
                assigned_centroids[ i ] = min_centroid ; 
            }
            for i in 0..k {
                let mut vec_sum: Vec<f32> = vec![ 0.0 ; vec_dims ] ; 
                let mut count: usize = 0 ; 
                for j in 0..assigned_centroids.len() {
                    if assigned_centroids[ j ] == centroids[ i ] {
                        self.vec_add( &mut vec_sum , &data[j] ) ; 
                        count += 1 ; 
                    }
                }
                self.vec_scale( &mut vec_sum , count as f32 ) ; 
                centroids[ i ] = Vec::from( vec_sum ) ; 
            }
        }    
        centroids    
    }

    fn vec_scale(
        self: &ProductQuantizer , 
        vec: &mut Vec<f32> , 
        scale: f32
    ) {
        for i in 0..vec.len() {
            vec[ i ] = vec[ i ] / scale ; 
        }
    }

    fn vec_add(
        self: &ProductQuantizer , 
        vec1: &mut Vec<f32> , 
        vec2: &Vec<f32>
    ) {
        println!( "{} {}" , vec1.len() , vec2.len() ) ;
        for i in 0..vec1.len() {
            println!( "{}" , i ) ; 
            vec1[ i ] = vec1[ i ] + vec2[ i ] ; 
        }
    }

    fn euclid_distance(
        self: &ProductQuantizer ,
        vec1: &Vec<f32> , 
        vec2: &Vec<f32>
    ) -> f32 {
        let mut squared_diff_sum = 0.0 ; 
        for i in 0..vec1.len() {
            squared_diff_sum += ( vec1[i] - vec2[i] ).powi( 2 ) ; 
        }
        squared_diff_sum.sqrt()
    }

}


fn main() {
    let mut quantizer = ProductQuantizer::new( 4 , 2 , 8 , DistanceMetric::Dot ) ; 
    let codewords = quantizer.fit(
        vec![ 
            vec![ 1.2 , 3.4 , 1.2 , 3.4 , 3.4 , 3.4 , 3.4 , 3.4 ] , 
            vec![ 1.2 , 3.4 , 1.2 , 3.4 , 3.4 , 3.4 , 3.4 , 3.4 ] ,
            vec![ 1.2 , 3.4 , 1.2 , 3.4 , 3.4 , 3.4 , 3.4 , 3.4 ] , 
            vec![ 1.2 , 3.4 , 1.2 , 3.4 , 3.4 , 3.4 , 3.4 , 3.4 ] ,
            vec![ 1.2 , 3.4 , 1.2 , 3.4 , 3.4 , 3.4 , 3.4 , 3.4 ] , 
            vec![ 1.2 , 3.4 , 1.2 , 3.4 , 3.4 , 3.4 , 3.4 , 3.4 ] ,
            vec![ 1.2 , 3.4 , 1.2 , 3.4 , 3.4 , 3.4 , 3.4 , 3.4 ] , 
            vec![ 1.2 , 3.4 , 1.2 , 3.4 , 3.4 , 3.4 , 3.4 , 3.4 ] 
        ] , 100
    ) ; 
    println!( "{:?}" , codewords ) ;
}
