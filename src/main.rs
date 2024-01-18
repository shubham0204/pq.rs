use std::vec;

use rand::seq::SliceRandom ; 

pub enum DistanceMetric {
    L2 , 
    Dot
}

pub struct ProductQuantizer {
    n_subvectors: usize ,
    n_codes: usize , 
    src_vec_dims: usize , 
    codewords: Vec<Vec<Vec<f32>>> , 
    metric: DistanceMetric
}

impl ProductQuantizer {

    pub fn new(
        n_subvectors: usize , 
        n_codes: usize ,
        src_vec_dims: usize ,
        metric: DistanceMetric
    ) -> Self {
        ProductQuantizer{ n_subvectors , n_codes , src_vec_dims , codewords: Vec::new() , metric }
    }

    pub fn fit(
        self: &mut ProductQuantizer , 
        src_vectors: &Vec<Vec<f32>> , 
        iterations: usize
    ) {
        // `sub_vec_dims` is the dimensionality of the subvector
        // A subvector is a part of the source vector, obtained by dividing the source 
        // vector equally into `n_subvectors` parts
        let sub_vec_dims: usize = self.src_vec_dims / self.n_subvectors ; 

        // `codewords` has dims ( n_subspaces , n_codes , sub_vec_dims )
        self.codewords = Vec::new() ; 
        for m in 0..self.n_subvectors {

            // Store mth subvector in `sub_vectors_m`, for each vector in `src_vectors`
            // `sub_vectors_m` has dims ( n_src_vectors , sub_vec_dims )
            let mut sub_vectors_m: Vec<Vec<f32>> = Vec::new() ; 
            for vec in src_vectors {
                sub_vectors_m.push( vec[ (m * sub_vec_dims)..((m+1) * sub_vec_dims) ].to_vec() ) ; 
            }

            // Perform K means clustering on `sub_vectors_m`
            // with K = `n_codes` and data points `sub_vectors_m`
            // It returns centroid having dims ( n_codes , sub_vec_dims )
            self.codewords.push( self.kmeans( &sub_vectors_m , self.n_codes , iterations ) ) ; 
        }
    }

    /// <func description>
    /// # Arguments
    /// * `vectors` - Vectors to be quantized
    /// # Returns
    /// The quantized vectors
    pub fn encode(
        self: &ProductQuantizer , 
        vectors: &Vec<Vec<f32>>
    ) -> Vec<Vec<usize>> {
        let sub_vec_dims: usize = self.src_vec_dims / self.n_subvectors ; 
        let mut vector_codes: Vec<Vec<usize>> = Vec::new() ; 
        for vec in vectors {
            let mut subvectors: Vec<Vec<f32>> = Vec::new() ; 
            for m in 0..self.n_subvectors {
                subvectors.push( vec[  (m * sub_vec_dims)..((m+1) * sub_vec_dims) ].to_vec() )
            }
            vector_codes.push( self.vector_quantize( &subvectors ) ) ; 
        }

        /*
        for m in 0..self.n_subvectors {
            let mut vectors_sub: Vec<Vec<f32>> = Vec::new() ; 
            for vec in vectors {
                vectors_sub.push( vec[ (m * sub_vec_dims)..((m+1) * sub_vec_dims) ].to_vec() ) ; 
            }
            vector_codes.push( self.vector_quantize( &vectors_sub , &self.codewords[m] ) )
        }*/ 

        vector_codes
    }

    pub fn dtable(
        self: &ProductQuantizer , 
        query: &Vec<f32>
    ) -> Vec<Vec<f32>> {
        let sub_vec_dims: usize = self.src_vec_dims / self.n_subvectors ; 
        let mut dtable: Vec<Vec<f32>> = Vec::new() ; 
        for m in 0..self.n_subvectors {
            let query_sub: Vec<f32> = query[ (m * sub_vec_dims)..((m+1) * sub_vec_dims) ].to_vec() ; 
            let mut m_sub_distances: Vec<f32> = Vec::new() ; 
            for k in 0..self.n_codes {
                m_sub_distances.push( self.euclid_distance( &self.codewords[m][k] , &query_sub ) )
            }
            dtable.push( m_sub_distances ) ; 
        }
        dtable
    }

    /// Given vectors and a codebook,
    /// return the index of the centroid in the codebook to 
    /// which each vector is the nearest
    fn vector_quantize(
        self: &ProductQuantizer , 
        vector: &Vec<Vec<f32>> 
    ) -> Vec<usize> {

        let mut codes: Vec<usize> = Vec::new() ; 
        for ( m , subvector ) in vector.iter().enumerate() {
            let mut min_distance: f32 = f32::MAX ; 
            let mut min_distance_code_index: usize = 0 ; 
            for ( k , code ) in self.codewords[m].iter().enumerate() {
                let distance = self.euclid_distance( subvector , code ) ; 
                if distance < min_distance {
                    min_distance = distance ; 
                    min_distance_code_index = k ; 
                } 
            }
            codes.push( min_distance_code_index ) ;
        }

        codes
    }

    /// Given a set of vectors, return `n_clusters` vectors
    /// which represent the centroids of the clusters
    /// derived using the standard algorithm of K-means clusterings
    fn kmeans(
        self: &ProductQuantizer , 
        vecs: &Vec<Vec<f32>> , 
        n_clusters: usize , 
        iter: usize
    ) -> Vec<Vec<f32>> {
        let mut assigned_centroids: Vec<Vec<f32>> = vec![ Vec::new() ; vecs.len() ] ; 

        // Choose random samples from `vecs` as initial centroids
        let mut centroids: Vec<Vec<f32>> = vecs.choose_multiple( &mut rand::thread_rng() , n_clusters )  
                                            .map( | vec | vec.to_vec() )
                                            .collect() ; 

        let vec_dims: usize = vecs[0].len() ; 

        // For each iteration, 
        // `centroids` gets updated
        for _ in 0..iter {

            // For each vector, calculate its nearest centroid
            // from the available `centroids`
            // For the ith vector, the nearest centroid is stored in 
            // assigned_centroids[i]
            for i in 0..vecs.len() {
                let mut min_centroid_distance: f32 = f32::MAX ; 
                let mut min_centroid: Vec<f32> = centroids[0].clone() ; 
                for centroid in &centroids {
                    // Calculate distance between ith vector and `centroid`
                    let distance: f32 = self.euclid_distance( &vecs[i] , centroid ) ;
                    if distance < min_centroid_distance {
                        min_centroid_distance = distance ; 
                        min_centroid = centroid.clone() ; 
                    }
                }
                assigned_centroids[ i ] = min_centroid ; 
            }

            // For each centroid, take the average/mean of all vectors to which it was
            // the nearest
            // Replace the current centroid with the computed mean
            for i in 0..n_clusters {

                let mut vec_sum: Vec<f32> = vec![ 0.0 ; vec_dims ] ; 
                let mut count: usize = 0 ; 

                // Add only those vectors for which 
                // nearest_centroid = assigned_centroids[i]
                for j in 0..assigned_centroids.len() {
                    if assigned_centroids[ j ] == centroids[ i ] {
                        self.vec_add( &mut vec_sum , &vecs[j] ) ; 
                        count += 1 ; 
                    }
                }
                self.vec_scale( &mut vec_sum , 1.0 / (count as f32) ) ; 

                // Update the centroid with 
                // the average of the vectors
                centroids[ i ] = Vec::from( vec_sum ) ; 
            }
        }    

        centroids    
    }

    /// For given vector `vec`,
    /// multiply each element of `vec` with `scale`, inplace
    fn vec_scale(
        self: &ProductQuantizer , 
        vec: &mut Vec<f32> , 
        scale: f32
    ) {
        for i in 0..vec.len() {
            vec[ i ] = vec[ i ] * scale ; 
        }
    }

    /// Add `vec1` and `vec2` elementwise
    /// and store the result in `vec1` (through a mutable reference)
    fn vec_add(
        self: &ProductQuantizer , 
        vec1: &mut Vec<f32> , 
        vec2: &Vec<f32>
    ) {
        for i in 0..vec1.len() {
            vec1[ i ] = vec1[ i ] + vec2[ i ] ; 
        }
    }

    /// Compute the Euclidean distance between vectors `vec1` and `vec2`
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

    fn dot( 
        self: &ProductQuantizer , 
        vec1: &Vec<f32> , 
        vec2: &Vec<f32>
    ) -> f32 {
        let mut dot_product = 0.0 ; 
        for i in 0..vec1.len() {
            dot_product += vec1[i] * vec2[i] ; 
        }
        dot_product
    }

}


fn main() {
    let n_subvectors = 4 ; 
    let n_codes = 8 ; 
    let src_vec_dims = 8 ; 

    let mut quantizer = ProductQuantizer::new( n_subvectors , n_codes , src_vec_dims , DistanceMetric::Dot ) ; 
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
    let codewords = quantizer.fit(
        &vectors , 100
    ) ;
    let vector_codes = quantizer.encode( &vectors ) ;
    let dtable = quantizer.dtable( &vectors[4] ) ; 
    println!( "{:?}" , codewords ) ;
    println!( "{:?}" , vector_codes ) ;
    println!( "{:?}" , dtable ) ; 
}
