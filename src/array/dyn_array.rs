use std::sync::Arc;
use std::any::Any;

use arrow::array::{Array, ArrayRef};
use arrow::datatypes::*;
use bit_vec::BitVec;

use crate::datatype::{DynScalar};
use crate::array::primitive_array::{BinaryVec, BoolVec, NativeArray, StringVec, TypedVec};
use crate::array::time_array::*;
use crate::{for_all_primitivetype_with_variant, for_all_timetypes};

/// Trait for dynamic native arrays that can handle DynScalar values.
/// Provides type-erased operations for arrays of different types.
pub trait DynNativeArray<T> {
    /// Creates a new empty dynamic native array.
    fn new() -> Box<dyn DynNativeArray<T>> where Self: Sized;
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
            impl DynNativeArray<$native_type> for TypedVec<$arrow_type> {
                fn new() -> Box<dyn DynNativeArray<$native_type>> {
                    Box::new(TypedVec::<$arrow_type>::from_vec(Vec::new()))
                }
                fn push(&mut self, value: DynScalar) -> Result<(), String> {
                    match value {
                        DynScalar::$dyn_variant(x) => {
                            <Self as NativeArray>::push(self, x);
                            Ok(())
                        }
                        _ => Err(format!("Type mismatch")),
                    }
                }
                fn get(&self, index: usize) -> Option<DynScalar> {
                    if index >= <Self as NativeArray>::to_arrow_array(self).len() {
                        None
                    } else {
                        Some(DynScalar::$dyn_variant(*<Self as NativeArray>::get(self, index)))
                    }
                }
                fn as_native_array(&self) -> &dyn std::any::Any {
                    self
                }
                fn as_native_array_mut(&mut self) -> &mut dyn std::any::Any {
                    self
                }
                fn to_arrow_array(&self) -> arrow::array::ArrayRef {
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
            impl DynNativeArray<$native_type> for crate::array::primitive_array::TypedVec<$arrow_type> {
                fn new() -> Box<dyn DynNativeArray<$native_type>> {
                    Box::new(crate::array::primitive_array::TypedVec::<$arrow_type>::from_vec(Vec::new()))
                }
                fn push(&mut self, value: DynScalar) -> Result<(), String> {
                    match value {
                        DynScalar::$dyn_variant(x) => {
                            <Self as NativeArray>::push(self, x.into());
                            Ok(())
                        }
                        _ => Err(format!("Type mismatch")),
                    }
                }
                fn get(&self, index: usize) -> Option<DynScalar> {
                    if index >= <Self as NativeArray>::to_arrow_array(self).len() {
                        None
                    } else {
                        let inner_val = *<Self as NativeArray>::get(self, index);
                        Some(DynScalar::$dyn_variant(<$native_type>::from(inner_val)))
                    }
                }
                fn as_native_array(&self) -> &dyn Any {
                    self
                }
                fn as_native_array_mut(&mut self) -> &mut dyn Any {
                    self
                }
                fn to_arrow_array(&self) -> arrow::array::ArrayRef {
                    Arc::new(<Self as NativeArray>::to_arrow_array(self))
                }
            }
        )*
    }
}

for_all_primitivetype_with_variant!(impl_dynnativearray_for_primitive);
for_all_timetypes!(impl_dynnativearray_for_time);

impl DynNativeArray<String> for StringVec {
    fn new() -> Box<dyn DynNativeArray<String>> {
        Box::new(StringVec::from_vec(Vec::new()))
    }
    fn push(&mut self, value: DynScalar) -> Result<(), String> {
        match value {
            DynScalar::String(x) => {
                <Self as NativeArray>::push(self, x);
                Ok(())
            }
            _ => Err(format!("Type mismatch")),
        }
    }
    fn get(&self, index: usize) -> Option<DynScalar> {
        if index >= <Self as NativeArray>::to_arrow_array(self).len() {
            None
        } else {
            Some(DynScalar::String(<Self as NativeArray>::get(self, index).clone()))
        }
    }
    fn as_native_array(&self) -> &dyn Any {
        self
    }
    fn as_native_array_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn to_arrow_array(&self) -> arrow::array::ArrayRef {
        Arc::new(<Self as NativeArray>::to_arrow_array(self))
    }
}

impl DynNativeArray<Vec<u8>> for BinaryVec {
    fn new() -> Box<dyn DynNativeArray<Vec<u8>>> {
        Box::new(BinaryVec::from_vec(Vec::new()))
    }
    fn push(&mut self, value: DynScalar) -> Result<(), String> {
        match value {
            DynScalar::Binary(x) => {
                <Self as NativeArray>::push(self, x);
                Ok(())
            }
            _ => Err(format!("Type mismatch")),
        }
    }
    fn get(&self, index: usize) -> Option<DynScalar> {
        if index >= <Self as NativeArray>::to_arrow_array(self).len() {
            None
        } else {
            Some(DynScalar::Binary(<Self as NativeArray>::get(self, index).clone()))
        }
    }
    fn as_native_array(&self) -> &dyn Any {
        self
    }
    fn as_native_array_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn to_arrow_array(&self) -> arrow::array::ArrayRef {
        Arc::new(<Self as NativeArray>::to_arrow_array(self))
    }
}

impl DynNativeArray<bool> for BoolVec {
    fn new() -> Box<dyn DynNativeArray<bool>> {
        Box::new(BoolVec::from_vec(Vec::new()))
    }
    fn push(&mut self, value: DynScalar) -> Result<(), String> {
        match value {
            DynScalar::Bool(x) => {
                <Self as NativeArray>::push(self, x);
                Ok(())
            }
            _ => Err(format!("Type mismatch")),
        }
    }
    fn get(&self, index: usize) -> Option<DynScalar> {
        if index >= <Self as NativeArray>::to_arrow_array(self).len() {
            None
        } else {
            Some(DynScalar::Bool(<Self as NativeArray>::get(self, index)))
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
pub trait MakeDynArray {
    /// Creates a new dynamic array for this type.
    fn make_array() -> Box<dyn DynNativeArray<Self>>;
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
                fn make_array() -> Box<dyn DynNativeArray<Self>> {
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
                fn make_array() -> Box<dyn DynNativeArray<Self>> {
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
    fn make_array() -> Box<dyn DynNativeArray<Self>> {
        Box::new(StringVec::from_vec(Vec::new()))
    }
}

impl Into<DynScalar> for Vec<u8> {
    fn into(self) -> DynScalar {
        DynScalar::Binary(self)
    }
}

impl MakeDynArray for Vec<u8> {
    fn make_array() -> Box<dyn DynNativeArray<Self>> {
        Box::new(BinaryVec::from_vec(Vec::new()))
    }
}
impl Into<DynScalar> for bool {
    fn into(self) -> DynScalar {
        DynScalar::Bool(self)
    }
}

impl MakeDynArray for bool {
    fn make_array() -> Box<dyn DynNativeArray<Self>> {
        Box::new(BoolVec::from_vec(Vec::new()))
    }
}

/// Trait for converting collections to dynamic native arrays.
pub trait IntoDynNativeArray {
    type Elem;
    /// Converts this collection to a dynamic native array.
    fn to_dyn_array(&self) -> Result<Box<dyn DynNativeArray<Self::Elem>>, String>;
}

impl<T> IntoDynNativeArray for [T]
where 
    T: Clone + Into<DynScalar> + MakeDynArray + 'static
{
    type Elem = T;
    fn to_dyn_array(&self) -> Result<Box<dyn DynNativeArray<T>>, String> {
        let mut arr = T::make_array();
        for v in self {
            arr.push(v.clone().into())?;
        }
        Ok(arr)
    }
}

impl IntoDynNativeArray for BitVec {
    type Elem = bool;
    fn to_dyn_array(&self) -> Result<Box<dyn DynNativeArray<bool>>, String> {
        let mut arr: Box<dyn DynNativeArray<bool>> = Box::new(BoolVec::from_vec(Vec::new()));
        for v in self.iter() {
            arr.push(DynScalar::Bool(v))?;
        }
        Ok(arr)
    }
}

/// Creates a new dynamic array for the specified type.
pub fn new<T>() -> Box<dyn DynNativeArray<T>>
where
    T: MakeDynArray + 'static,
{
    T::make_array()
}


#[cfg(test)]
mod tests {
    use arrow::array::{ArrayRef, BooleanArray, Int32Array};
    use bit_vec::BitVec;

    use super::*;

    #[test]
    fn test_to_dyn_array_i32() {
        let input = vec![1i32, 2, 42];
        let arr = input.to_dyn_array().unwrap();
        assert_eq!(arr.get(2), Some(DynScalar::Int32(42)));
    }

    #[test]
    fn test_to_dyn_array_f64() {
        let input = vec![-1.5f64, 0.0, 9.81];
        let arr = input.to_dyn_array().unwrap();
        assert_eq!(arr.get(0), Some(DynScalar::Float64(-1.5)));
    }

    #[test]
    fn test_to_dyn_array_string() {
        let input = vec!["abc".to_string(), "def".to_string(), "114514".to_string()];
        let arr = input.to_dyn_array().unwrap();
        assert_eq!(arr.get(2), Some(DynScalar::String("114514".to_string())));
    }

    #[test]
    fn test_to_dyn_array_bool() {
        let mut input:BitVec = BitVec::from_elem(3, false);
        input.set(1, true);
        let arr = input.to_dyn_array().unwrap();
        assert_eq!(arr.get(1), Some(DynScalar::Bool(true)));
        let arr = arr.to_arrow_array();
        let arr = arr.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(*arr, BooleanArray::from(vec![false, true, false]));
    }

    #[test]
    fn test_to_dyn_array_empty() {
        let input: Vec<i32> = vec![];
        let mut arr = input.to_dyn_array().unwrap();
        assert!(arr.get(0).is_none());
        arr.push(DynScalar::Int32(100)).unwrap();
        assert_eq!(arr.get(0), Some(DynScalar::Int32(100)));
    }

    #[test]
    fn test_dyn_native_array_to_arrow() {
        let input = vec![114, 514, 1919810];
        let arr: ArrayRef = input.to_dyn_array().unwrap().to_arrow_array();
        let arr = arr.as_any().downcast_ref::<Int32Array>().unwrap();
        let expected = Arc::new(Int32Array::from(input.clone()));
        assert_eq!(arr.values(), expected.values());
    }

    #[test]
    fn test_new_with_type() {
        let mut arr: Box<dyn DynNativeArray<i32>> = new::<i32>();
        arr.push(DynScalar::Int32(42)).unwrap();
        assert_eq!(arr.get(0), Some(DynScalar::Int32(42)));

        let mut arr: Box<dyn DynNativeArray<f64>> = new::<f64>();
        arr.push(DynScalar::Float64(3.14)).unwrap();
        assert_eq!(arr.get(0), Some(DynScalar::Float64(3.14)));

        let mut arr: Box<dyn DynNativeArray<String>> = new::<String>();
        arr.push(DynScalar::String("hello".to_string())).unwrap();
        assert_eq!(arr.get(0), Some(DynScalar::String("hello".to_string())));
    }
}