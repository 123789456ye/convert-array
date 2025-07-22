use std::any::Any;
use std::sync::Arc;

use arrow::array::{ArrayRef};
use arrow::datatypes::*;
use bit_vec::BitVec;

use crate::array::primitive_array::{
    BinaryVec, BoolVec, Decimal128Value, Decimal128Vec, NativeArray, StringVec, TypedVec,
};
use crate::array::time_array::*;
use crate::datatype::DynScalar;
use crate::{for_all_primitivetype_with_variant, for_all_timetypes};

/// Trait for dynamic native arrays that can handle DynScalar values.
/// Provides type-erased operations for arrays of different types.
pub trait DynNativeArray<T> {
    /// Creates a new empty dynamic native array.
    fn new() -> Box<dyn DynNativeArray<T>>
    where
        Self: Sized;
    /// Pushes a DynScalar value into the array.
    fn push(&mut self, value: DynScalar) -> Result<(), String>;
    /// Gets a DynScalar value at the specified index.
    fn get(&self, index: usize) -> Option<DynScalar>;
    /// Returns a reference to the underlying native array as Any.
    fn as_native_array(&self) -> &dyn Any;
    /// Returns a mutable reference to the underlying native array as Any.
    fn as_native_array_mut(&mut self) -> &mut dyn Any;
    /// Converts the array to an Arrow ArrayRef.
    fn to_arrow_array(&self) -> ArrayRef;
}

/// Implements DynNativeArray for primitive types using a macro.
/// Generates implementations for all supported primitive types.
macro_rules! impl_dynnativearray_for_primitive {
    ($(($native_type:ty, $arrow_type:ty, $dyn_variant:ident)),*) => {
        $(
            impl DynNativeArray<Option<$native_type>> for TypedVec<$arrow_type> {
                fn new() -> Box<dyn DynNativeArray<Option<$native_type>>> {
                    Box::new(TypedVec::<$arrow_type>::from_vec(Vec::new()))
                }
                fn push(&mut self, value: DynScalar) -> Result<(), String> {
                    match value {
                        DynScalar::$dyn_variant(x) => {
                            <Self as NativeArray>::push(self, Some(x));
                            Ok(())
                        },
                        DynScalar::Null => {
                            <Self as NativeArray>::push(self, None);
                            Ok(())
                        },
                        _ => Err(format!("Type mismatch")),
                    }
                }
                fn get(&self, index: usize) -> Option<DynScalar> {
                    let res = <Self as NativeArray>::get(self, index);
                    if let Some(res) = res {
                        Some(DynScalar::$dyn_variant(res))
                    } else {
                        None
                    }
                }
                fn as_native_array(&self) -> &dyn Any {
                    self
                }
                fn as_native_array_mut(&mut self) -> &mut dyn Any {
                    self
                }
                fn to_arrow_array(&self) -> ArrayRef {
                    Arc::new(<Self as NativeArray>::to_arrow_array(self))
                }
            }
        )*
    }
}

/// Implements DynNativeArray for time types using a macro.
/// Generates implementations for all supported time types.
macro_rules! impl_dynnativearray_for_time {
    ($(($native_type:ty, $arrow_type:ty, $dyn_variant:ident, $inner_type:ty)),*) => {
        $(
            impl DynNativeArray<Option<$native_type>> for crate::array::primitive_array::TypedVec<$arrow_type> {
                fn new() -> Box<dyn DynNativeArray<Option<$native_type>>> {
                    Box::new(crate::array::primitive_array::TypedVec::<$arrow_type>::from_vec(Vec::new()))
                }
                fn push(&mut self, value: DynScalar) -> Result<(), String> {
                    match value {
                        DynScalar::$dyn_variant(x) => {
                            <Self as NativeArray>::push(self, Some(x.into()));
                            Ok(())
                        },
                        DynScalar::Null => {
                            <Self as NativeArray>::push(self, None);
                            Ok(())
                        },
                        _ => Err(format!("Type mismatch")),
                    }
                }
                fn get(&self, index: usize) -> Option<DynScalar> {
                    let res = <Self as NativeArray>::get(self, index);
                    if let Some(res) = res {
                        Some(DynScalar::$dyn_variant(<$native_type>::from(res)))
                    } else {
                        None
                    }
                }
                fn as_native_array(&self) -> &dyn Any {
                    self
                }
                fn as_native_array_mut(&mut self) -> &mut dyn Any {
                    self
                }
                fn to_arrow_array(&self) -> ArrayRef {
                    Arc::new(<Self as NativeArray>::to_arrow_array(self))
                }
            }
        )*
    }
}

for_all_primitivetype_with_variant!(impl_dynnativearray_for_primitive);
for_all_timetypes!(impl_dynnativearray_for_time);

impl DynNativeArray<Option<String>> for StringVec {
    fn new() -> Box<dyn DynNativeArray<Option<String>>> {
        Box::new(StringVec::from_vec(Vec::new()))
    }
    fn push(&mut self, value: DynScalar) -> Result<(), String> {
        match value {
            DynScalar::String(x) => {
                <Self as NativeArray>::push(self, Some(x));
                Ok(())
            }
            DynScalar::Null => {
                <Self as NativeArray>::push(self, None);
                Ok(())
            }
            _ => Err(format!("Type mismatch")),
        }
    }
    fn get(&self, index: usize) -> Option<DynScalar> {
        let res = <Self as NativeArray>::get(self, index);
        if let Some(res) = res {
            Some(DynScalar::String(res))
        } else {
            None
        }
    }
    fn as_native_array(&self) -> &dyn Any {
        self
    }
    fn as_native_array_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn to_arrow_array(&self) -> ArrayRef {
        Arc::new(<Self as NativeArray>::to_arrow_array(self))
    }
}

impl DynNativeArray<Option<Vec<u8>>> for BinaryVec {
    fn new() -> Box<dyn DynNativeArray<Option<Vec<u8>>>> {
        Box::new(BinaryVec::from_vec(Vec::new()))
    }
    fn push(&mut self, value: DynScalar) -> Result<(), String> {
        match value {
            DynScalar::Binary(x) => {
                <Self as NativeArray>::push(self, Some(x));
                Ok(())
            }
            DynScalar::Null => {
                <Self as NativeArray>::push(self, None);
                Ok(())
            }
            _ => Err(format!("Type mismatch")),
        }
    }
    fn get(&self, index: usize) -> Option<DynScalar> {
        let res = <Self as NativeArray>::get(self, index);
        if let Some(res) = res {
            Some(DynScalar::Binary(res))
        } else {
            None
        }
    }
    fn as_native_array(&self) -> &dyn Any {
        self
    }
    fn as_native_array_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn to_arrow_array(&self) -> ArrayRef {
        Arc::new(<Self as NativeArray>::to_arrow_array(self))
    }
}

impl DynNativeArray<Option<bool>> for BoolVec {
    fn new() -> Box<dyn DynNativeArray<Option<bool>>> {
        Box::new(BoolVec::from_vec(Vec::new()))
    }
    fn push(&mut self, value: DynScalar) -> Result<(), String> {
        match value {
            DynScalar::Bool(x) => {
                <Self as NativeArray>::push(self, Some(x));
                Ok(())
            }
            DynScalar::Null => {
                <Self as NativeArray>::push(self, None);
                Ok(())
            }
            _ => Err(format!("Type mismatch")),
        }
    }
    fn get(&self, index: usize) -> Option<DynScalar> {
        let res = <Self as NativeArray>::get(self, index);
        if let Some(res) = res {
            Some(DynScalar::Bool(res))
        } else {
            None
        }
    }
    fn as_native_array(&self) -> &dyn Any {
        self
    }
    fn as_native_array_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn to_arrow_array(&self) -> ArrayRef {
        Arc::new(<Self as NativeArray>::to_arrow_array(self))
    }
}

impl DynNativeArray<Option<Decimal128Value>> for Decimal128Vec {
    fn new() -> Box<dyn DynNativeArray<Option<Decimal128Value>>> {
        Box::new(Decimal128Vec::from_vec(Vec::new()))
    }
    fn push(&mut self, value: DynScalar) -> Result<(), String> {
        match value {
            DynScalar::Decimal128(x) => {
                <Self as NativeArray>::push(self, Some(x));
                Ok(())
            }
            DynScalar::Null => {
                <Self as NativeArray>::push(self, None);
                Ok(())
            }
            _ => Err(format!("Type mismatch")),
        }
    }
    fn get(&self, index: usize) -> Option<DynScalar> {
        let res = <Self as NativeArray>::get(self, index);
        if let Some(res) = res {
            Some(DynScalar::Decimal128(res))
        } else {
            None
        }
    }
    fn as_native_array(&self) -> &dyn Any {
        self
    }
    fn as_native_array_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn to_arrow_array(&self) -> ArrayRef {
        Arc::new(<Self as NativeArray>::to_arrow_array(self))
    }
}

/// Trait for types that can create dynamic arrays.
pub trait MakeDynArray
where
    Self: Sized,
{
    /// Creates a new dynamic array for this type.
    fn make_array() -> Box<dyn DynNativeArray<Option<Self>>>;
}

/// Implements MakeDynArray for primitive types using a macro.
macro_rules! impl_makedynarray_for_primitive {
    ($(($native_type:ty, $arrow_type:ty, $dyn_variant:ident)),*) => {
        $(
            impl Into<DynScalar> for $native_type {
                fn into(self) -> DynScalar {
                    DynScalar::$dyn_variant(self)
                }
            }
            impl MakeDynArray for $native_type {
                fn make_array() -> Box<dyn DynNativeArray<Option<Self>>> {
                    Box::new(TypedVec::<$arrow_type>::from_vec(Vec::new()))
                }
            }
        )*
    }
}

/// Implements MakeDynArray for time types using a macro.
macro_rules! impl_makedynarray_for_time {
    ($(($native_type:ty, $arrow_type:ty, $dyn_variant:ident, $inner_type:ty)),*) => {
        $(
            impl Into<DynScalar> for $native_type {
                fn into(self) -> DynScalar {
                    DynScalar::$dyn_variant(self)
                }
            }
            impl MakeDynArray for $native_type {
                fn make_array() -> Box<dyn DynNativeArray<Option<Self>>> {
                    Box::new(TypedVec::<$arrow_type>::from_vec(Vec::new()))
                }
            }
        )*
    }
}

for_all_primitivetype_with_variant!(impl_makedynarray_for_primitive);
for_all_timetypes!(impl_makedynarray_for_time);

impl Into<DynScalar> for String {
    fn into(self) -> DynScalar {
        DynScalar::String(self)
    }
}

impl MakeDynArray for String {
    fn make_array() -> Box<dyn DynNativeArray<Option<Self>>> {
        Box::new(StringVec::from_vec(Vec::new()))
    }
}

/// Conflicting with Vec<T> to List.
///
/*impl Into<DynScalar> for Vec<u8> {
    fn into(self) -> DynScalar {
        DynScalar::Binary(self)
    }
}*/

impl MakeDynArray for Vec<u8> {
    fn make_array() -> Box<dyn DynNativeArray<Option<Self>>> {
        Box::new(BinaryVec::from_vec(Vec::new()))
    }
}

impl Into<DynScalar> for bool {
    fn into(self) -> DynScalar {
        DynScalar::Bool(self)
    }
}

impl MakeDynArray for bool {
    fn make_array() -> Box<dyn DynNativeArray<Option<Self>>> {
        Box::new(BoolVec::from_vec(Vec::new()))
    }
}

impl Into<DynScalar> for Decimal128Value {
    fn into(self) -> DynScalar {
        DynScalar::Decimal128(self)
    }
}

impl MakeDynArray for Decimal128Value {
    fn make_array() -> Box<dyn DynNativeArray<Option<Self>>> {
        Box::new(Decimal128Vec::from_vec(Vec::new()))
    }
}
/// Trait for converting collections to dynamic native arrays.
pub trait IntoDynNativeArray {
    type Elem;
    /// Converts this collection to a dynamic native array.
    fn to_dyn_array(&self) -> Result<Box<dyn DynNativeArray<Option<Self::Elem>>>, String>;
}

impl<T> IntoDynNativeArray for [Option<T>]
where
    T: Clone + Into<DynScalar> + MakeDynArray + 'static,
{
    type Elem = T;
    fn to_dyn_array(&self) -> Result<Box<dyn DynNativeArray<Option<T>>>, String> {
        let mut arr = T::make_array();
        for v in self {
            if let Some(v) = v {
                arr.push(v.clone().into())?;
            } else {
                arr.push(DynScalar::Null)?;
            }
        }
        Ok(arr)
    }
}

impl IntoDynNativeArray for BitVec {
    type Elem = bool;
    fn to_dyn_array(&self) -> Result<Box<dyn DynNativeArray<Option<bool>>>, String> {
        let mut arr: Box<dyn DynNativeArray<Option<bool>>> =
            Box::new(BoolVec::from_vec(Vec::new()));
        for v in self.iter() {
            arr.push(DynScalar::Bool(v))?;
        }
        Ok(arr)
    }
}

/// Creates a new dynamic array for the specified type.
pub fn new<T>() -> Box<dyn DynNativeArray<Option<T>>>
where
    T: MakeDynArray + 'static,
{
    T::make_array()
}

#[cfg(test)]
mod tests {
    use arrow::array::{Array, ArrayRef, BooleanArray, Decimal128Array, Int32Array, TimestampSecondArray};
    use bit_vec::BitVec;

    use crate::array::primitive_array::Decimal128Value;

    use super::*;

    #[test]
    fn test_to_dyn_array_i32() {
        let input: Vec<Option<i32>> = vec![1i32, 2, 42].into_iter().map(|x| Some(x)).collect();
        let arr = input.to_dyn_array().unwrap();
        assert_eq!(arr.get(2), Some(DynScalar::Int32(42)));
    }

    #[test]
    fn test_to_dyn_array_f64() {
        let input: Vec<Option<f64>> = vec![-1.5f64, 0.0, 9.81]
            .into_iter()
            .map(|x| Some(x))
            .collect();
        let arr = input.to_dyn_array().unwrap();
        assert_eq!(arr.get(0), Some(DynScalar::Float64(-1.5)));
    }

    #[test]
    fn test_to_dyn_array_string() {
        let input: Vec<Option<String>> =
            vec!["abc".to_string(), "def".to_string(), "114514".to_string()]
                .into_iter()
                .map(|x| Some(x))
                .collect();
        let arr = input.to_dyn_array().unwrap();
        assert_eq!(arr.get(2), Some(DynScalar::String("114514".to_string())));
    }

    #[test]
    fn test_to_dyn_array_bool() {
        let mut input: BitVec = BitVec::from_elem(3, false);
        input.set(1, true);
        let arr = input.to_dyn_array().unwrap();
        assert_eq!(arr.get(1), Some(DynScalar::Bool(true)));
        let arr = arr.to_arrow_array();
        let arr = arr.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(*arr, BooleanArray::from(vec![false, true, false]));
    }

    #[test]
    fn test_to_dyn_array_empty() {
        let input: Vec<Option<i32>> = vec![];
        let mut arr = input.to_dyn_array().unwrap();
        assert!(arr.get(0).is_none());
        arr.push(DynScalar::Int32(100)).unwrap();
        assert_eq!(arr.get(0), Some(DynScalar::Int32(100)));
    }

    #[test]
    fn test_dyn_native_array_to_arrow() {
        let input: Vec<Option<i32>> = vec![114, 514, 1919810]
            .into_iter()
            .map(|x| Some(x))
            .collect();
        let arr: ArrayRef = input.to_dyn_array().unwrap().to_arrow_array();
        let arr = arr.as_any().downcast_ref::<Int32Array>().unwrap();
        let expected = Arc::new(Int32Array::from(input.clone()));
        assert_eq!(arr.values(), expected.values());
    }

    #[test]
    fn test_new_with_type() {
        let mut arr: Box<dyn DynNativeArray<Option<i32>>> = new::<i32>();
        arr.push(DynScalar::Int32(42)).unwrap();
        assert_eq!(arr.get(0), Some(DynScalar::Int32(42)));

        let mut arr: Box<dyn DynNativeArray<Option<f64>>> = new::<f64>();
        arr.push(DynScalar::Float64(3.14)).unwrap();
        assert_eq!(arr.get(0), Some(DynScalar::Float64(3.14)));

        let mut arr: Box<dyn DynNativeArray<Option<String>>> = new::<String>();
        arr.push(DynScalar::String("hello".to_string())).unwrap();
        assert_eq!(arr.get(0), Some(DynScalar::String("hello".to_string())));
    }

    #[test]
    fn test_dyn_native_option_i32_array_to_arrow() {
        let input = vec![Some(1i32), None, Some(42)];
        let arr: ArrayRef = input.to_dyn_array().unwrap().to_arrow_array();
        let arr = arr.as_any().downcast_ref::<Int32Array>().unwrap();
        let expected = Arc::new(Int32Array::from(input.clone()));
        assert_eq!(arr.values(), expected.values());
    }

    #[test]
    fn test_dyn_native_option_time_array_to_arrow() {
        let input = vec![
            Some(TimestampSecond::from(114514)),
            None,
            Some(TimestampSecond::from(1919810)),
        ];
        let arr: ArrayRef = input.to_dyn_array().unwrap().to_arrow_array();
        let arr = arr.as_any().downcast_ref::<TimestampSecondArray>().unwrap();
        let expected = Arc::new(TimestampSecondArray::from(
            input
                .iter()
                .map(|opt| opt.map(|ts| ts.into()))
                .collect::<Vec<_>>(),
        ));
        assert_eq!(arr.values(), expected.values());
    }

    #[test]
    fn test_dyn_native_option_decimal_array_to_arrow() {
        let input = vec![
            Some(Decimal128Value {
                value: 12345,
                precision: 10,
                scale: 2,
            }),
            None,
            Some(Decimal128Value {
                value: 67890,
                precision: 10,
                scale: 2,
            }),
        ];
        let arr: ArrayRef = input.to_dyn_array().unwrap().to_arrow_array();
        let arr = arr.as_any().downcast_ref::<Decimal128Array>().unwrap();
        let expected = Arc::new(
            Decimal128Array::from(vec![Some(12345), None, Some(67890)])
                .with_precision_and_scale(10, 2)
                .unwrap(),
        );
        println!("arr: {:?}, {:?}, {:?}", arr.is_null(0), arr.is_null(1), arr.is_null(2));
        assert_eq!(arr.values(), expected.values());
        assert_eq!(arr.precision(), expected.precision());
        assert_eq!(arr.scale(), expected.scale());
        assert_eq!(arr.is_null(1), expected.is_null(1));
    }
}
