pub trait IntoArrow {
    fn into_arrow(vec: &[Self]) -> arrow::array::ArrayRef
    where
        Self: Sized;
}

// primitive_impls.rs - Add this to your trait_def crate or wherever you define the IntoArrow trait

use arrow::array::*;
use std::sync::Arc;

// Implement IntoArrow for primitive types
impl crate::IntoArrow for String {
    fn into_arrow(vec: &[Self]) -> arrow::array::ArrayRef {
        Arc::new(StringArray::from(vec.to_vec())) as ArrayRef
    }
}

impl crate::IntoArrow for i32 {
    fn into_arrow(vec: &[Self]) -> arrow::array::ArrayRef {
        Arc::new(Int32Array::from(vec.to_vec())) as ArrayRef
    }
}

impl crate::IntoArrow for i64 {
    fn into_arrow(vec: &[Self]) -> arrow::array::ArrayRef {
        Arc::new(Int64Array::from(vec.to_vec())) as ArrayRef
    }
}

impl crate::IntoArrow for i16 {
    fn into_arrow(vec: &[Self]) -> arrow::array::ArrayRef {
        Arc::new(Int16Array::from(vec.to_vec())) as ArrayRef
    }
}

impl crate::IntoArrow for i8 {
    fn into_arrow(vec: &[Self]) -> arrow::array::ArrayRef {
        Arc::new(Int8Array::from(vec.to_vec())) as ArrayRef
    }
}

impl crate::IntoArrow for u64 {
    fn into_arrow(vec: &[Self]) -> arrow::array::ArrayRef {
        Arc::new(UInt64Array::from(vec.to_vec())) as ArrayRef
    }
}

impl crate::IntoArrow for u32 {
    fn into_arrow(vec: &[Self]) -> arrow::array::ArrayRef {
        Arc::new(UInt32Array::from(vec.to_vec())) as ArrayRef
    }
}

impl crate::IntoArrow for u16 {
    fn into_arrow(vec: &[Self]) -> arrow::array::ArrayRef {
        Arc::new(UInt16Array::from(vec.to_vec())) as ArrayRef
    }
}

impl crate::IntoArrow for u8 {
    fn into_arrow(vec: &[Self]) -> arrow::array::ArrayRef {
        Arc::new(UInt8Array::from(vec.to_vec())) as ArrayRef
    }
}

impl crate::IntoArrow for bool {
    fn into_arrow(vec: &[Self]) -> arrow::array::ArrayRef {
        Arc::new(BooleanArray::from(vec.to_vec())) as ArrayRef
    }
}

impl crate::IntoArrow for f32 {
    fn into_arrow(vec: &[Self]) -> arrow::array::ArrayRef {
        Arc::new(Float32Array::from(vec.to_vec())) as ArrayRef
    }
}

impl crate::IntoArrow for f64 {
    fn into_arrow(vec: &[Self]) -> arrow::array::ArrayRef {
        Arc::new(Float64Array::from(vec.to_vec())) as ArrayRef
    }
}

// For string references
impl crate::IntoArrow for &str {
    fn into_arrow(vec: &[Self]) -> arrow::array::ArrayRef {
        Arc::new(StringArray::from(vec.to_vec())) as ArrayRef
    }
}