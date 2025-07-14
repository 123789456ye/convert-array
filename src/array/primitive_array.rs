use arrow::array::{
    Array, ArrowPrimitiveType, BinaryArray, BinaryBuilder, BooleanArray, BooleanBuilder, Decimal128Array, PrimitiveArray, StringArray, StringBuilder
};
use arrow::datatypes::*;

use bit_vec::BitVec;

use crate::array::time_array::Decimal128Value;
use crate::{for_all_primitivetype};

/// Trait for native array implementations that can be converted to Arrow arrays.
/// Provides methods for manipulating arrays and converting them to Arrow format.
pub trait NativeArray
where
    for<'a> Self: 'a,
{
    type Item;
    type ItemRef<'a>;
    type ArrowArray: Array;

    /// Adds an item to the array.
    fn push(&mut self, item: Self::Item);
    /// Gets a reference to the item at the specified index.
    fn get(&self, index: usize) -> Self::ItemRef<'_>;
    /// Converts this native array to an Arrow array.
    fn to_arrow_array(&self) -> Self::ArrowArray;
}

// ========== Primitive类型 ============

/// A typed vector wrapper for Arrow primitive types.
/// Stores native values and provides conversion to Arrow arrays.
pub struct TypedVec<P: ArrowPrimitiveType> {
    pub data: Vec<P::Native>,
}

impl<P> TypedVec<P>
where
    P: ArrowPrimitiveType,
    PrimitiveArray<P>: From<Vec<P::Native>>,
{
    /// Creates a new TypedVec from a vector of native values.
    pub fn from_vec(vec: Vec<P::Native>) -> Self {
        Self { data: vec }
    }
}
impl<P> NativeArray for TypedVec<P>
where
    P: ArrowPrimitiveType,
    PrimitiveArray<P>: From<Vec<P::Native>>,
{
    type Item = P::Native;
    type ItemRef<'a> = &'a P::Native;
    type ArrowArray = PrimitiveArray<P>;

    fn push(&mut self, v: Self::Item) {
        self.data.push(v)
    }
    fn get(&self, idx: usize) -> Self::ItemRef<'_> {
        &self.data[idx]
    }
    fn to_arrow_array(&self) -> Self::ArrowArray {
        self.data.clone().into()
    }
}


/// Rust类型到Arrow Primitive类型的映射
/// Maps Rust types to their corresponding Arrow primitive types.
/// 
/// This trait enables generic conversion from Vec<T> to TypedVec<T::ArrowType>
/// allowing usage like: Vec<i32>.into() -> Int32Vec
pub trait ArrowTyped {
    type ArrowType: ArrowPrimitiveType;
}

/// Implements ArrowTyped trait for primitive types using a macro.
/// Maps common Rust primitive types to their Arrow counterparts.
macro_rules! impl_arrowtyped {
    (
        $(($rust_ty:ty, $arrow_ty:ty)),*
    ) => {
        $(
            impl ArrowTyped for $rust_ty {
                type ArrowType = $arrow_ty;
            }
        )*
    }
}
for_all_primitivetype!(impl_arrowtyped);

/// Generic conversion from Vec<T> to TypedVec<T::ArrowType>
/// 
/// This enables convenient usage like:
/// let vec: Vec<i32> = vec![1, 2, 3];
/// let typed_vec: Int32Vec = vec.into();
impl<T> From<Vec<T>> for TypedVec<<T as ArrowTyped>::ArrowType>
where
    T: ArrowTyped,
    <<T as ArrowTyped>::ArrowType as ArrowPrimitiveType>::Native: From<T>,
{
    fn from(vec: Vec<T>) -> Self {
        let native_vec = vec
            .into_iter()
            .map(<<T as ArrowTyped>::ArrowType as ArrowPrimitiveType>::Native::from)
            .collect();
        TypedVec { data: native_vec }
    }
}

pub type Int8Vec = TypedVec<Int8Type>;
pub type Int16Vec = TypedVec<Int16Type>;
pub type Int32Vec = TypedVec<Int32Type>;
pub type Int64Vec = TypedVec<Int64Type>;
pub type UInt8Vec = TypedVec<UInt8Type>;
pub type UInt16Vec = TypedVec<UInt16Type>;
pub type UInt32Vec = TypedVec<UInt32Type>;
pub type UInt64Vec = TypedVec<UInt64Type>;
pub type Float32Vec = TypedVec<Float32Type>;
pub type Float64Vec = TypedVec<Float64Type>;

// ========== Boolean 类型（使用BitVec） ==========

/// A vector for storing boolean values using BitVec for efficient storage.
pub struct BoolVec {
    pub data: BitVec,
}

impl BoolVec {
    /// Creates a new BoolVec from a BitVec.
    pub fn from_bitvec(data: BitVec) -> Self {
        Self { data }
    }
    
    /// Creates a new BoolVec from a vector of boolean values.
    pub fn from_vec(vec: Vec<bool>) -> Self {
        let mut bitvec = BitVec::new();
        for b in vec {
            bitvec.push(b);
        }
        Self { data: bitvec }
    }
}

impl NativeArray for BoolVec {
    type Item = bool;
    type ItemRef<'a> = bool;
    type ArrowArray = BooleanArray;

    fn push(&mut self, item: Self::Item) {
        self.data.push(item);
    }
    
    fn get(&self, idx: usize) -> Self::ItemRef<'_> {
        self.data[idx]
    }
    
    fn to_arrow_array(&self) -> Self::ArrowArray {
        let mut builder = BooleanBuilder::new();
        for i in 0..self.data.len() {
            builder.append_value(self.data[i]);
        }
        builder.finish()
    }
}

impl From<BitVec> for BoolVec {
    fn from(bitvec: BitVec) -> Self {
        BoolVec::from_bitvec(bitvec)
    }
}

impl From<Vec<bool>> for BoolVec {
    fn from(vec: Vec<bool>) -> Self {
        BoolVec::from_vec(vec)
    }
}

/// A vector for storing optional boolean values.
pub struct BoolOptVec {
    pub data: Vec<Option<bool>>,
}

impl BoolOptVec {
    /// Creates a new BoolOptVec from a vector of optional boolean values.
    pub fn from_vec(vec: Vec<Option<bool>>) -> Self {
        Self { data: vec }
    }
}

impl NativeArray for BoolOptVec {
    type Item = Option<bool>;
    type ItemRef<'a> = &'a Option<bool>;
    type ArrowArray = BooleanArray;

    fn push(&mut self, item: Self::Item) {
        self.data.push(item);
    }
    
    fn get(&self, idx: usize) -> Self::ItemRef<'_> {
        &self.data[idx]
    }
    
    fn to_arrow_array(&self) -> Self::ArrowArray {
        let mut builder = BooleanBuilder::new();
        for opt_b in &self.data {
            match opt_b {
                Some(b) => builder.append_value(*b),
                None => builder.append_null(),
            }
        }
        builder.finish()
    }
}

impl From<Vec<Option<bool>>> for BoolOptVec {
    fn from(vec: Vec<Option<bool>>) -> Self {
        BoolOptVec::from_vec(vec)
    }
}


/// Macro to implement vector types with builders for string and binary data.
/// Generates both non-optional and optional vector implementations.
macro_rules! impl_builder_vec {
    (
        $vec_name:ident,
        $opt_vec_name:ident,
        $item_type:ty,
        $arrow_array:ty,
        $builder:ty
    ) => {
        pub struct $vec_name {
            pub data: Vec<$item_type>,
        }

        /// Creates a new vector from the provided data.
        impl $vec_name {
            pub fn from_vec(vec: Vec<$item_type>) -> Self {
                Self { data: vec }
            }
        }

        impl NativeArray for $vec_name {
            type Item = $item_type;
            type ItemRef<'a> = &'a $item_type;
            type ArrowArray = $arrow_array;

            fn push(&mut self, item: Self::Item) {
                self.data.push(item)
            }
            
            fn get(&self, idx: usize) -> Self::ItemRef<'_> {
                &self.data[idx]
            }
            
            fn to_arrow_array(&self) -> Self::ArrowArray {
                let mut builder = <$builder>::new();
                for item in &self.data {
                    builder.append_value(item);
                }
                builder.finish()
            }
        }

        impl From<Vec<$item_type>> for $vec_name {
            fn from(vec: Vec<$item_type>) -> Self {
                Self::from_vec(vec)
            }
        }

        pub struct $opt_vec_name {
            pub data: Vec<Option<$item_type>>,
        }

        /// Creates a new optional vector from the provided data.
        impl $opt_vec_name {
            pub fn from_vec(vec: Vec<Option<$item_type>>) -> Self {
                Self { data: vec }
            }
        }

        impl NativeArray for $opt_vec_name {
            type Item = Option<$item_type>;
            type ItemRef<'a> = &'a Option<$item_type>;
            type ArrowArray = $arrow_array;

            fn push(&mut self, item: Self::Item) {
                self.data.push(item)
            }
            
            fn get(&self, idx: usize) -> Self::ItemRef<'_> {
                &self.data[idx]
            }
            
            fn to_arrow_array(&self) -> Self::ArrowArray {
                let mut builder = <$builder>::new();
                for item in &self.data {
                    match item {
                        Some(v) => builder.append_value(v),
                        None => builder.append_null(),
                    }
                }
                builder.finish()
            }
        }

        impl From<Vec<Option<$item_type>>> for $opt_vec_name {
            fn from(vec: Vec<Option<$item_type>>) -> Self {
                Self::from_vec(vec)
            }
        }
    };
}

impl_builder_vec!(
    StringVec,
    StringOptVec,
    String,
    StringArray,
    StringBuilder
);

impl_builder_vec!(
    BinaryVec,
    BinaryOptVec,
    Vec<u8>,
    BinaryArray,
    BinaryBuilder
);

// ========== Decimal128 类型 ==========

/// A vector for storing Decimal128 values with precision and scale.
pub struct Decimal128Vec {
    pub data: Vec<Decimal128Value>,
}

impl Decimal128Vec {
    /// Creates a new Decimal128Vec from a vector of Decimal128Value.
    pub fn from_vec(vec: Vec<Decimal128Value>) -> Self {
        Self { data: vec }
    }
}

impl NativeArray for Decimal128Vec {
    type Item = Decimal128Value;
    type ItemRef<'a> = &'a Decimal128Value;
    type ArrowArray = Decimal128Array;

    fn push(&mut self, item: Self::Item) {
        self.data.push(item);
    }
    
    fn get(&self, idx: usize) -> Self::ItemRef<'_> {
        &self.data[idx]
    }
    
    fn to_arrow_array(&self) -> Self::ArrowArray {
        if self.data.is_empty() {
            return Decimal128Array::from(vec![] as Vec<Option<i128>>)
                .with_precision_and_scale(10, 0)
                .unwrap();
        }
        
        let precision = self.data[0].precision;
        let scale = self.data[0].scale;
        let values: Vec<Option<i128>> = self.data.iter()
            .map(|d| Some(d.value))
            .collect();
            
        Decimal128Array::from(values)
            .with_precision_and_scale(precision, scale)
            .unwrap()
    }
}

impl From<Vec<Decimal128Value>> for Decimal128Vec {
    fn from(vec: Vec<Decimal128Value>) -> Self {
        Decimal128Vec::from_vec(vec)
    }
}

/// A vector for storing optional Decimal128 values.
pub struct Decimal128OptVec {
    pub data: Vec<Option<Decimal128Value>>,
}

impl Decimal128OptVec {
    /// Creates a new Decimal128OptVec from a vector of optional Decimal128Value.
    pub fn from_vec(vec: Vec<Option<Decimal128Value>>) -> Self {
        Self { data: vec }
    }
}

impl NativeArray for Decimal128OptVec {
    type Item = Option<Decimal128Value>;
    type ItemRef<'a> = &'a Option<Decimal128Value>;
    type ArrowArray = Decimal128Array;

    fn push(&mut self, item: Self::Item) {
        self.data.push(item);
    }
    
    fn get(&self, idx: usize) -> Self::ItemRef<'_> {
        &self.data[idx]
    }
    
    fn to_arrow_array(&self) -> Self::ArrowArray {
        if self.data.is_empty() {
            return Decimal128Array::from(vec![] as Vec<Option<i128>>)
                .with_precision_and_scale(10, 0)
                .unwrap();
        }
        
        // Find the first non-None value to get precision and scale
        let (precision, scale) = self.data.iter()
            .find_map(|opt| opt.as_ref().map(|d| (d.precision, d.scale)))
            .unwrap_or((10, 0));
            
        let values: Vec<Option<i128>> = self.data.iter()
            .map(|opt| opt.as_ref().map(|d| d.value))
            .collect();
            
        Decimal128Array::from(values)
            .with_precision_and_scale(precision, scale)
            .unwrap()
    }
}

impl From<Vec<Option<Decimal128Value>>> for Decimal128OptVec {
    fn from(vec: Vec<Option<Decimal128Value>>) -> Self {
        Decimal128OptVec::from_vec(vec)
    }
}

/// A typed optional vector wrapper for Arrow primitive types.
/// Stores optional native values and provides conversion to Arrow arrays.
pub struct TypedOptVec<P: ArrowPrimitiveType> {
    pub data: Vec<Option<P::Native>>,
}

impl<P> TypedOptVec<P>
where
    P: ArrowPrimitiveType,
    PrimitiveArray<P>: From<Vec<Option<P::Native>>>,
{
    /// Creates a new TypedOptVec from a vector of optional native values.
    pub fn from_vec(vec: Vec<Option<P::Native>>) -> Self {
        Self { data: vec }
    }
}

impl<P> NativeArray for TypedOptVec<P>
where
    P: ArrowPrimitiveType,
    PrimitiveArray<P>: From<Vec<Option<P::Native>>>,
{
    type Item = Option<P::Native>;
    type ItemRef<'a> = &'a Option<P::Native>;
    type ArrowArray = PrimitiveArray<P>;

    fn push(&mut self, v: Self::Item) {
        self.data.push(v)
    }
    
    fn get(&self, idx: usize) -> Self::ItemRef<'_> {
        &self.data[idx]
    }
    
    fn to_arrow_array(&self) -> Self::ArrowArray {
        self.data.clone().into()
    }
}

impl<P> From<Vec<Option<P::Native>>> for TypedOptVec<P>
where
    P: ArrowPrimitiveType,
    PrimitiveArray<P>: From<Vec<Option<P::Native>>>,
{
    fn from(vec: Vec<Option<P::Native>>) -> Self {
        Self::from_vec(vec)
    }
}

// Explicit type aliases for common optional primitive types
pub type Int8OptVec = TypedOptVec<Int8Type>;
pub type Int16OptVec = TypedOptVec<Int16Type>;
pub type Int32OptVec = TypedOptVec<Int32Type>;
pub type Int64OptVec = TypedOptVec<Int64Type>;
pub type UInt8OptVec = TypedOptVec<UInt8Type>;
pub type UInt16OptVec = TypedOptVec<UInt16Type>;
pub type UInt32OptVec = TypedOptVec<UInt32Type>;
pub type UInt64OptVec = TypedOptVec<UInt64Type>;
pub type Float32OptVec = TypedOptVec<Float32Type>;
pub type Float64OptVec = TypedOptVec<Float64Type>;

// ========== TESTS =============

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{BinaryArray, Float64Array, Int32Array, StringArray};

    #[test]
    fn test_i32_nativearray_to_arrowarray_eq() {
        let data = vec![1i32, 2, 3, 4, 5, -10, 100];
        let native: Int32Vec = data.clone().into();
        let arrow_from_native = native.to_arrow_array();
        let arrow_direct = Int32Array::from(data);
        assert_eq!(arrow_from_native, arrow_direct);
    }

    #[test]
    fn test_f64_nativearray_to_arrowarray_eq() {
        let data = vec![1.0f64, 3.14, 2.72, -7.618, 42.0];
        let native: Float64Vec = data.clone().into();
        let arrow_from_native = native.to_arrow_array();
        let arrow_direct = Float64Array::from(data);
        assert_eq!(arrow_from_native, arrow_direct);
    }

    #[test]
    fn test_string_nativearray() {
        let data = vec!["abc".to_string(), "def".to_string()];
        let array = StringVec::from(data.clone()).to_arrow_array();
        let refdata = StringArray::from(data);
        assert_eq!(array, refdata);
    }

    #[test]
    fn test_option_string_nativearray() {
        let data = vec![Some("abc".to_string()), None, Some("def".to_string())];
        let arr = StringOptVec::from(data.clone()).to_arrow_array();
        let refdata = {
            let mut builder = StringBuilder::new();
            for x in data {
                match x {
                    Some(s) => builder.append_value(s),
                    None => builder.append_null(),
                }
            }
            builder.finish()
        };
        assert_eq!(arr, refdata);
    }

    #[test]
    fn test_binary_nativearray() {
        let data = vec![b"abc".to_vec(), b"def".to_vec()];
        let array = BinaryVec::from(data.clone()).to_arrow_array();
        let refdata = {
            let refs: Vec<&[u8]> = data.iter().map(|x| x.as_slice()).collect();
            BinaryArray::from(refs)
        };
        assert_eq!(array, refdata);
    }

    #[test]
    fn test_option_binary_nativearray() {
        let data = vec![Some(b"abc".to_vec()), None, Some(b"def".to_vec())];
        let arr = BinaryOptVec::from(data.clone()).to_arrow_array();
        let refdata = {
            let mut builder = BinaryBuilder::new();
            for x in data {
                match x {
                    Some(s) => builder.append_value(&s),
                    None => builder.append_null(),
                }
            }
            builder.finish()
        };
        assert_eq!(arr, refdata);
    }
    #[test]
    fn test_bool_vec() {
        let mut bool_vec = BoolVec::from_vec(vec![true, false, true]);
        bool_vec.push(false);
        assert_eq!(bool_vec.get(0), true);
        
        let arrow_array = bool_vec.to_arrow_array();
        let expected = BooleanArray::from(vec![true, false, true, false]);
        assert_eq!(arrow_array, expected);
    }

    #[test]
    fn test_string_vec() {
        let mut string_vec = StringVec::from_vec(vec!["hello".to_string(), "world".to_string()]);
        string_vec.push("test".to_string());
        assert_eq!(string_vec.get(0), &"hello".to_string());
        
        let arrow_array = string_vec.to_arrow_array();
        assert_eq!(arrow_array.len(), 3);
    }

    #[test]
    fn test_decimal128() {
        let decimal_values = vec![
            Decimal128Value { value: 12345, precision: 10, scale: 2 },
            Decimal128Value { value: 67890, precision: 10, scale: 2 },
        ];
        let decimal_vec = Decimal128Vec::from(decimal_values);
        let arrow_array = decimal_vec.to_arrow_array();
        assert_eq!(arrow_array.len(), 2);
        assert_eq!(arrow_array, Decimal128Array::from(vec![Some(12345), Some(67890)])
            .with_precision_and_scale(10, 2)
            .unwrap());
    }

    #[test]
    fn test_typed_opt_vec() {
        let opt_values = vec![Some(1i32), None, Some(3i32), Some(4i32)];
        let typed_opt_vec = Int32OptVec::from(opt_values);
        let arrow_array = typed_opt_vec.to_arrow_array();
        
        assert_eq!(arrow_array.len(), 4);
        assert_eq!(arrow_array.value(0), 1);
        assert!(arrow_array.is_null(1));
        assert_eq!(arrow_array.value(2), 3);
        assert_eq!(arrow_array.value(3), 4);
    }

    #[test]
    fn test_typed_opt_vec_f64() {
        let opt_values = vec![Some(1.5f64), None, Some(3.14f64)];
        let typed_opt_vec = Float64OptVec::from(opt_values);
        let arrow_array = typed_opt_vec.to_arrow_array();
        
        assert_eq!(arrow_array.len(), 3);
        assert_eq!(arrow_array.value(0), 1.5);
        assert!(arrow_array.is_null(1));
        assert_eq!(arrow_array.value(2), 3.14);
    }

    #[test]
    fn test_typed_opt_vec_push_get() {
        let mut typed_opt_vec = Int32OptVec::from_vec(vec![Some(42)]);
        typed_opt_vec.push(None);
        typed_opt_vec.push(Some(100));
        
        assert_eq!(typed_opt_vec.get(0), &Some(42));
        assert_eq!(typed_opt_vec.get(1), &None);
        assert_eq!(typed_opt_vec.get(2), &Some(100));
    }

}
