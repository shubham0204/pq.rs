pub mod pq ; 
use pq::{ ProductQuantizer , DistanceMetric };

fn main() {
    let n_subvectors = 4 ; 
    let n_codes = 8 ; 
    let src_vec_dims = 8 ; 

    let mut quantizer = ProductQuantizer::new( n_subvectors , n_codes , src_vec_dims , DistanceMetric::Euclidean ) ; 
    let vectors: Vec<Vec<f32>> =  vec![ 
        vec![ 5.2 , 3.4 , 1.5 , 3.4 , -3.4 , 3.4 , 0.0, 3.4 ] , 
        vec![ 1.2 , 3.4 , 1.2 , 3.4 , 3.4 , -3.4 , 10.4 , 3.4 ] ,
        vec![ -1.2 , 3.4 , 1.2 , 3.4 , 3.4 , 3.4 , 3.4 , 3.4 ] , 
        vec![ 1.2 , 3.4 , 0.0 , 3.4 , 23.4 , 3.4 , 2.4 , 3.4096 ] ,
        vec![ 1.2 , 3.4 , 2.2 , 3.42 , 3.4 , 3.4 , 3.4 , 3.4 ] , 
        vec![ -1.2 , 3.4 , 1.2 , 3.4 , 3.4 , 4.4 , 3.4 , 3.4 ] ,
        vec![ 1.2 , 3.4 , 1.2 , 3.9 , 3.4 , -10.4 , 3.4 , 3.4 ] , 
        vec![ 1.2 , 3.4 , 1.2 , 3.4 , 0.0 , 3.4 , 3.4 , 3.4 ] 
    ] ; 
    quantizer.fit( &vectors , 100 ) ;
    let vector_codes = quantizer.encode::<u32>( &vectors ) ;
    let closed_vec_indexes = quantizer.search::<u32>( &vectors , &vector_codes ) ;

    println!( "{:?}" , vector_codes ) ;
    println!( "{:?}" , closed_vec_indexes ) ; 
}