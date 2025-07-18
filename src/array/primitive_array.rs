use arrow::array::{
    Array, ArrowPrimitiveType, BinaryArray, BinaryBuilder, BooleanArray, BooleanBuilder,
    Decimal128Array, PrimitiveArray, StringArray, StringBuilder,
};
use arrow::datatypes::*;

use bit_vec::BitVec;

use crate::for_all_primitivetype;

/// Trait for native array implementations that can be converted to Arrow arrays.
/// Provides methods for manipulating arrays and converting them to Arrow format.
pub trait NativeArray: 'static {
    type Item;
    type ItemRef;
    type ArrowArray: Array;

    /// Adds an item to the array.
    fn push(&mut self, item: Self::Item);
    /// Gets a reference to the item at the specified index.
    fn get(&self, index: usize) -> Self::ItemRef;
    /// Converts this native array to an Arrow array.
    fn to_arrow_array(&self) -> Self::ArrowArray;
}

// ========== Primitive Type ============

/// A typed vector wrapper for Arrow primitive types.
/// Stores native values and provides conversion to Arrow arrays.
pub struct TypedVec<P: ArrowPrimitiveType> {
    pub data: Vec<Option<P::Native>>,
    pub len: usize,
}

impl<P> TypedVec<P>
where
    P: ArrowPrimitiveType,
    PrimitiveArray<P>: From<Vec<Option<P::Native>>>,
{
    /// Creates a new TypedVec from a vector of optional native values.
    pub fn from_vec(vec: Vec<Option<P::Native>>) -> Self {
        Self {
            len: vec.len(),
            data: vec,
        }
    }
}
impl<P> NativeArray for TypedVec<P>
where
    P: ArrowPrimitiveType,
    PrimitiveArray<P>: From<Vec<Option<P::Native>>>,
{
    type Item = Option<P::Native>;
    type ItemRef = Option<P::Native>;
    type ArrowArray = PrimitiveArray<P>;

    fn push(&mut self, v: Self::Item) {
        self.len += 1;
        self.data.push(v)
    }
    fn get(&self, idx: usize) -> Self::ItemRef {
        if idx >= self.len {
            None
        } else {
            self.data[idx]
        }
    }
    fn to_arrow_array(&self) -> Self::ArrowArray {
        self.data.clone().into()
    }
}

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
impl<T> From<Vec<Option<T>>> for TypedVec<<T as ArrowTyped>::ArrowType>
where
    T: ArrowTyped,
    <<T as ArrowTyped>::ArrowType as ArrowPrimitiveType>::Native: From<T>,
{
    fn from(vec: Vec<Option<T>>) -> Self {
        let len = vec.len();
        let native_vec = vec
            .into_iter()
            .map(|opt| opt.map(<<T as ArrowTyped>::ArrowType as ArrowPrimitiveType>::Native::from))
            .collect();
        TypedVec {
            len,
            data: native_vec,
        }
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

// ========== Boolean Type (Using BitVec) ==========

/// A vector for storing boolean values using BitVec for efficient storage.
pub struct BoolVec {
    pub data: Vec<Option<bool>>,
    pub len: usize,
}

impl BoolVec {
    /// Creates a new BoolVec from a vector of optional boolean values.
    pub fn from_vec(vec: Vec<Option<bool>>) -> Self {
        Self {
            len: vec.len(),
            data: vec,
        }
    }
}

impl NativeArray for BoolVec {
    type Item = Option<bool>;
    type ItemRef = Option<bool>;
    type ArrowArray = BooleanArray;

    fn push(&mut self, item: Self::Item) {
        self.len += 1;
        self.data.push(item)
    }

    fn get(&self, idx: usize) -> Self::ItemRef {
        if idx >= self.len {
            None
        } else {
            self.data[idx]
        }
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

impl From<BitVec> for BoolVec {
    fn from(bitvec: BitVec) -> Self {
        let mut data = Vec::with_capacity(bitvec.len());
        for elem in bitvec {
            data.push(Some(elem));
        }
        Self {
            len: data.len(),
            data,
        }
    }
}

impl From<Vec<bool>> for BoolVec {
    fn from(vec: Vec<bool>) -> Self {
        let mut data = Vec::with_capacity(vec.len());
        for elem in vec {
            data.push(Some(elem));
        }
        Self {
            len: data.len(),
            data,
        }
    }
}

/// Macro to implement vector types with builders for string and binary data.
/// Generates both non-optional and optional vector implementations.
macro_rules! impl_builder_vec {
    (
        $vec_name:ident,
        $item_type:ty,
        $arrow_array:ty,
        $builder:ty
    ) => {
        pub struct $vec_name {
            pub data: Vec<$item_type>,
            pub len: usize,
        }

        impl $vec_name {
            pub fn from_vec(vec: Vec<$item_type>) -> Self {
                Self {
                    len: vec.len(),
                    data: vec,
                }
            }
        }

        impl NativeArray for $vec_name {
            type Item = $item_type;
            type ItemRef = $item_type;
            type ArrowArray = $arrow_array;

            fn push(&mut self, item: Self::Item) {
                self.len += 1;
                self.data.push(item)
            }

            fn get(&self, idx: usize) -> Self::ItemRef {
                if idx >= self.len {
                    None
                } else {
                    self.data[idx].clone()
                }
            }

            fn to_arrow_array(&self) -> Self::ArrowArray {
                let mut builder = <$builder>::new();
                for item in &self.data {
                    match item {
                        Some(item) => builder.append_value(item),
                        None => builder.append_null(),
                    }
                }
                builder.finish()
            }
        }

        impl From<Vec<$item_type>> for $vec_name {
            fn from(vec: Vec<$item_type>) -> Self {
                Self::from_vec(vec)
            }
        }
    };
}

impl_builder_vec!(StringVec, Option<String>, StringArray, StringBuilder);

impl_builder_vec!(BinaryVec, Option<Vec<u8>>, BinaryArray, BinaryBuilder);

// ========== Decimal128 Vector ==========

/// A vector for storing Decimal128 values with precision and scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Decimal128Value {
    pub value: i128,
    pub precision: u8,
    pub scale: i8,
}
pub struct Decimal128Vec {
    pub data: Vec<Option<Decimal128Value>>,
    pub len: usize,
}

impl Decimal128Vec {
    pub fn from_vec(vec: Vec<Option<Decimal128Value>>) -> Self {
        Self {
            len: vec.len(),
            data: vec,
        }
    }
}

impl NativeArray for Decimal128Vec {
    type Item = Option<Decimal128Value>;
    type ItemRef = Option<Decimal128Value>;
    type ArrowArray = Decimal128Array;

    fn push(&mut self, item: Self::Item) {
        self.data.push(item);
    }

    fn get(&self, idx: usize) -> Self::ItemRef {
        self.data[idx]
    }

    fn to_arrow_array(&self) -> Self::ArrowArray {
        if self.data.is_empty() {
            return Decimal128Array::from(vec![] as Vec<Option<i128>>)
                .with_precision_and_scale(10, 0)
                .unwrap();
        }

        let precision = self.data[0].unwrap().precision;
        let scale = self.data[0].unwrap().scale;
        let values: Vec<Option<i128>> = self
            .data
            .clone()
            .into_iter()
            .map(|d| match d.is_some() {
                true => Some(d.unwrap().value),
                false => None,
            })
            .collect();

        Decimal128Array::from(values)
            .with_precision_and_scale(precision, scale)
            .unwrap()
    }
}

impl From<Vec<Decimal128Value>> for Decimal128Vec {
    fn from(vec: Vec<Decimal128Value>) -> Self {
        Decimal128Vec::from_vec(vec.into_iter().map(|x| Some(x)).collect())
    }
}

// ========== TESTS =============

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{BinaryArray, Float64Array, Int32Array, StringArray};

    #[test]
    fn test_i32_nativearray_to_arrowarray_eq() {
        let data: Vec<Option<i32>> = vec![1i32, 2, 3, 4, 5, -10, 100]
            .into_iter()
            .map(|x| Some(x))
            .collect();
        let native: Int32Vec = data.clone().into();
        let arrow_from_native = native.to_arrow_array();
        let arrow_direct = Int32Array::from(data);
        assert_eq!(arrow_from_native, arrow_direct);
    }

    #[test]
    fn test_f64_nativearray_to_arrowarray_eq() {
        let data: Vec<Option<f64>> = vec![1.0f64, 3.14, 2.72, -7.618, 42.0]
            .into_iter()
            .map(|x| Some(x))
            .collect();
        let native: Float64Vec = data.clone().into();
        let arrow_from_native = native.to_arrow_array();
        let arrow_direct = Float64Array::from(data);
        assert_eq!(arrow_from_native, arrow_direct);
    }

    #[test]
    fn test_string_nativearray() {
        let data: Vec<Option<String>> = vec!["abc".to_string(), "def".to_string()]
            .into_iter()
            .map(|x| Some(x))
            .collect();
        let array = StringVec::from(data.clone()).to_arrow_array();
        let refdata = StringArray::from(data);
        assert_eq!(array, refdata);
    }

    #[test]
    fn test_binary_nativearray() {
        let data: Vec<Option<Vec<u8>>> = vec![b"abc".to_vec(), b"def".to_vec()]
            .into_iter()
            .map(|x| Some(x))
            .collect();
        let array = BinaryVec::from(data.clone()).to_arrow_array();
        let refdata = {
            let refs: Vec<&[u8]> = data
                .iter()
                .map(|x| x.as_ref().unwrap().as_slice())
                .collect();
            BinaryArray::from(refs)
        };
        assert_eq!(array, refdata);
    }

    #[test]
    fn test_bool_vec() {
        let mut bool_vec = BoolVec::from_vec(vec![Some(true), Some(false), Some(true)]);
        bool_vec.push(Some(false));
        assert_eq!(bool_vec.get(0), Some(true));

        let arrow_array = bool_vec.to_arrow_array();
        let expected = BooleanArray::from(vec![true, false, true, false]);
        assert_eq!(arrow_array, expected);
    }

    #[test]
    fn test_string_vec() {
        let mut string_vec = StringVec::from_vec(
            vec!["hello".to_string(), "world".to_string()]
                .into_iter()
                .map(|x| Some(x))
                .collect(),
        );
        string_vec.push(Some("test".to_string()));
        assert_eq!(string_vec.get(0), Some("hello".to_string()));

        let arrow_array = string_vec.to_arrow_array();
        assert_eq!(arrow_array.len(), 3);
    }

    #[test]
    fn test_decimal128() {
        let decimal_values = vec![
            Decimal128Value {
                value: 12345,
                precision: 10,
                scale: 2,
            },
            Decimal128Value {
                value: 67890,
                precision: 10,
                scale: 2,
            },
        ];
        let decimal_vec = Decimal128Vec::from(decimal_values);
        let arrow_array = decimal_vec.to_arrow_array();
        assert_eq!(arrow_array.len(), 2);
        assert_eq!(
            arrow_array,
            Decimal128Array::from(vec![Some(12345), Some(67890)])
                .with_precision_and_scale(10, 2)
                .unwrap()
        );
    }

    #[test]
    fn test_typed_opt_vec_push_get() {
        let mut typed_opt_vec = Int32Vec::from_vec(vec![Some(42)]);
        typed_opt_vec.push(None);
        typed_opt_vec.push(Some(100));

        assert_eq!(typed_opt_vec.get(0), Some(42));
        assert_eq!(typed_opt_vec.get(1), None);
        assert_eq!(typed_opt_vec.get(2), Some(100));
    }
}
