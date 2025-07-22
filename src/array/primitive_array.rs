use std::u8;

use arrow::array::{
    Array, ArrowPrimitiveType, BinaryArray, BooleanArray,
    Decimal128Array, PrimitiveArray, StringArray,
};
use arrow::buffer::{BooleanBuffer, Buffer, NullBuffer, OffsetBuffer};
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
    pub data: Vec<P::Native>,
    pub validity: Option<BitVec>,
    pub len: usize,
}

impl<P: ArrowPrimitiveType> TypedVec<P> {
    /// Create from Vec<Option<T>>.
    pub fn from_vec(vec: Vec<Option<P::Native>>) -> Self {
        let len = vec.len();
        let mut data = Vec::with_capacity(len);
        let mut validity: Option<BitVec> = None;

        for v in vec {
            match v {
                Some(val) => {
                    data.push(val);
                    if let Some(n) = validity.as_mut() {
                        n.push(true);
                    }
                }
                None => {
                    data.push(P::Native::default());
                    if validity.is_none() {
                        let mut n = BitVec::from_elem(data.len() - 1, true);
                        n.push(false);
                        validity = Some(n);
                    } else {
                        validity.as_mut().unwrap().push(false);
                    }
                }
            }
        }

        Self {
            data,
            validity,
            len,
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
        match v {
            Some(val) => {
                self.data.push(val);
                if let Some(validity) = self.validity.as_mut() {
                    validity.push(true);
                }
            }
            None => {
                self.data.push(P::Native::default());
                if self.validity.is_none() {
                    let mut n = BitVec::from_elem(self.len, true);
                    n.push(false);
                    self.validity = Some(n);
                } else {
                    self.validity.as_mut().unwrap().push(false);
                }
            }
        }
        self.len += 1;
    }

    fn get(&self, idx: usize) -> Self::ItemRef {
        if idx >= self.len {
            None
        } else {
            match &self.validity {
                Some(validity) if !validity[idx] => None,
                _ => Some(self.data[idx]),
            }
        }
    }

    fn to_arrow_array(&self) -> Self::ArrowArray {
        let values = Buffer::from_slice_ref(&self.data);
        let validity = self.validity.as_ref().map(|validity| {
            arrow::buffer::NullBuffer::from_iter(validity.iter())
        });
        PrimitiveArray::new(values.into(), validity)
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
    <<T as ArrowTyped>::ArrowType as ArrowPrimitiveType>::Native: From<T> + Default,
{
    fn from(vec: Vec<Option<T>>) -> Self {
        let len = vec.len();
        let mut data = Vec::with_capacity(len);
        let mut validity: Option<BitVec> = None;

        for opt in vec {
            match opt {
                Some(v) => {
                    data.push(<<T as ArrowTyped>::ArrowType as ArrowPrimitiveType>::Native::from(v));
                    if let Some(n) = validity.as_mut() {
                        n.push(true);
                    }
                }
                None => {
                    data.push(<<T as ArrowTyped>::ArrowType as ArrowPrimitiveType>::Native::default());
                    if validity.is_none() {
                        let mut n = BitVec::from_elem(data.len() - 1, true);
                        n.push(false);
                        validity = Some(n);
                    } else {
                        validity.as_mut().unwrap().push(false);
                    }
                }
            }
        }

        Self {
            len,
            data,
            validity,
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
    pub data: BitVec,
    pub validity: Option<BitVec>,
    pub len: usize,
}

impl BoolVec {
    /// Create from Vec<Option<bool>>
    pub fn from_vec(vec: Vec<Option<bool>>) -> Self {
        let len = vec.len();
        let mut data = BitVec::with_capacity(len);
        let mut validity: Option<BitVec> = None;

        for opt in vec {
            match opt {
                Some(b) => {
                    data.push(b);
                    if let Some(n) = validity.as_mut() {
                        n.push(true);
                    }
                }
                None => {
                    data.push(false); // dummy
                    if validity.is_none() {
                        let mut n = BitVec::from_elem(data.len() - 1, true);
                        n.push(false);
                        validity = Some(n);
                    } else {
                        validity.as_mut().unwrap().push(false);
                    }
                }
            }
        }

        Self {
            data,
            validity,
            len,
        }
    }
}

impl NativeArray for BoolVec {
    type Item = Option<bool>;
    type ItemRef = Option<bool>;
    type ArrowArray = BooleanArray;

    fn push(&mut self, item: Self::Item) {
        match item {
            Some(b) => {
                self.data.push(b);
                if let Some(n) = self.validity.as_mut() {
                    n.push(true);
                }
            }
            None => {
                self.data.push(false);
                if self.validity.is_none() {
                    let mut n = BitVec::from_elem(self.len, true);
                    n.push(false);
                    self.validity = Some(n);
                } else {
                    self.validity.as_mut().unwrap().push(false);
                }
            }
        }
        self.len += 1;
    }

    fn get(&self, idx: usize) -> Self::ItemRef {
        if idx >= self.len {
            None
        } else {
            match &self.validity {
                Some(n) if !n[idx] => None,
                _ => Some(self.data[idx]),
            }
        }
    }

    fn to_arrow_array(&self) -> Self::ArrowArray {
        let values = BooleanBuffer::from_iter(self.data.clone().into_iter());
        let validity = self.validity.as_ref().map(|validity| {
            arrow::buffer::NullBuffer::from_iter(validity.iter())
        });
        BooleanArray::new(values, validity)
    }
}

impl From<Vec<bool>> for BoolVec {
    fn from(vec: Vec<bool>) -> Self {
        let data = BitVec::from_iter(vec.into_iter());
        Self {
            len: data.len(),
            data,
            validity: None,
        }
    }
}

impl From<Vec<Option<bool>>> for BoolVec {
    fn from(vec: Vec<Option<bool>>) -> Self {
        Self::from_vec(vec)
    }
}

impl From<BitVec> for BoolVec {
    fn from(bitvec: BitVec) -> Self {
        Self {
            len: bitvec.len(),
            data: bitvec,
            validity: None,
        }
    }
}

// ========== String Vector ==========

pub struct StringVec {
    pub data: Vec<u8>,        // Flat storage of all string bytes
    pub offsets: Vec<i32>,    // Offset boundaries for each string
    pub validity: Option<BitVec>,
    pub len: usize,
}

impl StringVec {
    pub fn from_vec(vec: Vec<Option<String>>) -> Self {
        let len = vec.len();
        let mut data = Vec::new();
        let mut offsets = Vec::with_capacity(len + 1);
        offsets.push(0);
        let mut validity: Option<BitVec> = None;

        for opt in vec {
            match opt {
                Some(s) => {
                    data.extend_from_slice(s.as_bytes());
                    offsets.push(data.len() as i32);
                    if let Some(n) = validity.as_mut() {
                        n.push(true);
                    }
                }
                None => {
                    // For null values, offset stays the same (empty string)
                    offsets.push(data.len() as i32);
                    if validity.is_none() {
                        let mut n = BitVec::from_elem(offsets.len() - 2, true);
                        n.push(false);
                        validity = Some(n);
                    } else {
                        validity.as_mut().unwrap().push(false);
                    }
                }
            }
        }

        Self {
            data,
            offsets,
            validity,
            len,
        }
    }
}

impl NativeArray for StringVec {
    type Item = Option<String>;
    type ItemRef = Option<String>;
    type ArrowArray = StringArray;

    fn push(&mut self, item: Self::Item) {
        match item {
            Some(s) => {
                self.data.extend_from_slice(s.as_bytes());
                self.offsets.push(self.data.len() as i32);
                if let Some(n) = self.validity.as_mut() {
                    n.push(true);
                }
            }
            None => {
                self.offsets.push(self.data.len() as i32);
                if self.validity.is_none() {
                    let mut n = BitVec::from_elem(self.len, true);
                    n.push(false);
                    self.validity = Some(n);
                } else {
                    self.validity.as_mut().unwrap().push(false);
                }
            }
        }
        self.len += 1;
    }

    fn get(&self, idx: usize) -> Self::ItemRef {
        if idx >= self.len {
            None
        } else {
            match &self.validity {
                Some(validity) if !validity[idx] => None,
                _ => {
                    let start = self.offsets[idx] as usize;
                    let end = self.offsets[idx + 1] as usize;
                    let bytes = &self.data[start..end];
                    Some(String::from_utf8_lossy(bytes).into_owned())
                }
            }
        }
    }

    fn to_arrow_array(&self) -> Self::ArrowArray {
        let offsets_buffer = OffsetBuffer::new(self.offsets.clone().into());
        let values_buffer = Buffer::from_vec(self.data.clone());
        
        let validity = self.validity.as_ref().map(|validity| {
            NullBuffer::from_iter(validity.iter())
        });
        
        StringArray::new(offsets_buffer, values_buffer, validity)
    }
}

impl From<Vec<Option<String>>> for StringVec {
    fn from(vec: Vec<Option<String>>) -> Self {
        Self::from_vec(vec)
    }
}

impl From<Vec<String>> for StringVec {
    fn from(vec: Vec<String>) -> Self {
        let len = vec.len();
        let mut data = Vec::new();
        let mut offsets = Vec::with_capacity(len + 1);
        offsets.push(0);

        for s in vec {
            data.extend_from_slice(s.as_bytes());
            offsets.push(data.len() as i32);
        }

        Self {
            data,
            offsets,
            validity: None,
            len,
        }
    }
}

// ========== Binary Vector ==========

pub struct BinaryVec {
    pub data: Vec<u8>,        // Flat storage of all binary data
    pub offsets: Vec<i32>,    // Offset boundaries for each binary blob
    pub validity: Option<BitVec>,
    pub len: usize,
}

impl BinaryVec {
    pub fn from_vec(vec: Vec<Option<Vec<u8>>>) -> Self {
        let len = vec.len();
        let mut data = Vec::new();
        let mut offsets = Vec::with_capacity(len + 1);
        offsets.push(0);
        let mut validity: Option<BitVec> = None;

        for opt in vec {
            match opt {
                Some(bytes) => {
                    data.extend_from_slice(&bytes);
                    offsets.push(data.len() as i32);
                    if let Some(n) = validity.as_mut() {
                        n.push(true);
                    }
                }
                None => {
                    // For null values, offset stays the same (empty binary)
                    offsets.push(data.len() as i32);
                    if validity.is_none() {
                        let mut n = BitVec::from_elem(offsets.len() - 2, true);
                        n.push(false);
                        validity = Some(n);
                    } else {
                        validity.as_mut().unwrap().push(false);
                    }
                }
            }
        }

        Self {
            data,
            offsets,
            validity,
            len,
        }
    }
}

impl NativeArray for BinaryVec {
    type Item = Option<Vec<u8>>;
    type ItemRef = Option<Vec<u8>>;
    type ArrowArray = BinaryArray;

    fn push(&mut self, item: Self::Item) {
        match item {
            Some(bytes) => {
                self.data.extend_from_slice(&bytes);
                self.offsets.push(self.data.len() as i32);
                if let Some(n) = self.validity.as_mut() {
                    n.push(true);
                }
            }
            None => {
                self.offsets.push(self.data.len() as i32);
                if self.validity.is_none() {
                    let mut n = BitVec::from_elem(self.len, true);
                    n.push(false);
                    self.validity = Some(n);
                } else {
                    self.validity.as_mut().unwrap().push(false);
                }
            }
        }
        self.len += 1;
    }

    fn get(&self, idx: usize) -> Self::ItemRef {
        if idx >= self.len {
            None
        } else {
            match &self.validity {
                Some(validity) if !validity[idx] => None,
                _ => {
                    let start = self.offsets[idx] as usize;
                    let end = self.offsets[idx + 1] as usize;
                    Some(self.data[start..end].to_vec())
                }
            }
        }
    }

    fn to_arrow_array(&self) -> Self::ArrowArray {
        let offsets_buffer = OffsetBuffer::new(self.offsets.clone().into());
        let values_buffer = Buffer::from_vec(self.data.clone());
        
        let validity = self.validity.as_ref().map(|validity| {
            NullBuffer::from_iter(validity.iter())
        });
        
        BinaryArray::new(offsets_buffer, values_buffer, validity)
    }
}

impl From<Vec<Option<Vec<u8>>>> for BinaryVec {
    fn from(vec: Vec<Option<Vec<u8>>>) -> Self {
        Self::from_vec(vec)
    }
}

impl From<Vec<Vec<u8>>> for BinaryVec {
    fn from(vec: Vec<Vec<u8>>) -> Self {
        let len = vec.len();
        let mut data = Vec::new();
        let mut offsets = Vec::with_capacity(len + 1);
        offsets.push(0);

        for bytes in vec {
            data.extend_from_slice(&bytes);
            offsets.push(data.len() as i32);
        }

        Self {
            data,
            offsets,
            validity: None,
            len,
        }
    }
}

// ========== Decimal128 Vector ==========

/// A vector for storing Decimal128 values with precision and scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Decimal128Value {
    pub value: i128,
    pub precision: u8,
    pub scale: i8,
}
pub struct Decimal128Vec {
    pub data: Vec<i128>,
    pub validity: Option<BitVec>,
    pub len: usize,
    pub precision: u8,
    pub scale: i8,
}

impl Decimal128Vec {
    /// Create from Vec<Option<Decimal128Value>>
    pub fn from_vec(vec: Vec<Option<Decimal128Value>>) -> Self {
        if vec.is_empty() {
            return Self {
                data: Vec::new(),
                validity: None,
                len: 0,
                precision: 0xFF,
                scale: 0,
            }
        }

        let first = vec.iter().find_map(|v| *v).expect("At least one non-null value required");
        let precision = first.precision;
        let scale = first.scale;

        let len = vec.len();
        let mut data = Vec::with_capacity(len);
        let mut validity: Option<BitVec> = None;

        for opt in vec {
            match opt {
                Some(v) => {
                    data.push(v.value);
                    if let Some(n) = validity.as_mut() {
                        n.push(true);
                    }
                }
                None => {
                    data.push(0); // dummy slot
                    if validity.is_none() {
                        let mut n = BitVec::from_elem(data.len() - 1, true);
                        n.push(false);
                        validity = Some(n);
                    } else {
                        validity.as_mut().unwrap().push(false);
                    }
                }
            }
        }

        Self {
            data,
            validity,
            len,
            precision,
            scale,
        }
    }
}

impl NativeArray for Decimal128Vec {
    type Item = Option<Decimal128Value>;
    type ItemRef = Option<Decimal128Value>;
    type ArrowArray = Decimal128Array;

    fn push(&mut self, item: Self::Item) {
        match item {
            Some(v) => {
                if self.precision == 0xFF {
                    self.precision = v.precision;
                    self.scale = v.scale;
                }
                self.data.push(v.value);
                if let Some(validity) = self.validity.as_mut() {
                    validity.push(true);
                }
            }
            None => {
                self.data.push(0); // dummy
                if self.validity.is_none() {
                    let mut n = BitVec::from_elem(self.len, true); 
                    n.push(false);
                    self.validity = Some(n);
                } else {
                    self.validity.as_mut().unwrap().push(false);
                }
            }
        }
        self.len += 1;
    }

    fn get(&self, idx: usize) -> Self::ItemRef {
        if idx >= self.len {
            None
        } else {
            match &self.validity {
                Some(validity) if !validity[idx] => None,
                _ => Some(Decimal128Value {
                    value: self.data[idx],
                    precision: self.precision,
                    scale: self.scale,
                }),
            }
        }
    }

    fn to_arrow_array(&self) -> Self::ArrowArray {
        let buffer = Buffer::from_slice_ref(&self.data);
        let validity = self.validity.as_ref().map(|validity| {
            arrow::buffer::NullBuffer::from_iter(validity.iter())
        });

        Decimal128Array::new(buffer.into(), validity)
            .with_precision_and_scale(self.precision.into(), self.scale)
            .unwrap()
    }
}

impl From<Vec<Option<Decimal128Value>>> for Decimal128Vec {
    fn from(vec: Vec<Option<Decimal128Value>>) -> Self {
        Self::from_vec(vec)
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
            Some(Decimal128Value {
                value: 12345,
                precision: 10,
                scale: 2,
            }),
            Some(Decimal128Value {
                value: 67890,
                precision: 10,
                scale: 2,
            }),
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
