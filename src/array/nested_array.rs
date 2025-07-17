use arrow::array::{ArrayRef, FixedSizeListArray, ListArray, MapArray, StructArray};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{DataType, Field, Schema};
use std::collections::HashMap;
use std::sync::Arc;

use crate::array::dispatch::default_dyn_scalar;
use crate::array::dispatch::dynscalar_vec_to_array;
use crate::array::primitive_array::NativeArray;
use crate::datatype::DynScalar;

// ========== List Type Implementation ==========

/// A vector implementation for Arrow List arrays.
/// Stores nested vectors and converts them to Arrow ListArray format.

/// Implement Into<DynScalar> for Vec<T> (List)
impl<T> Into<DynScalar> for Vec<T>
where
    T: Into<DynScalar>,
{
    fn into(self) -> DynScalar {
        DynScalar::List(self.into_iter().map(|item| item.into()).collect())
    }
}

pub struct ListVec<T> {
    pub data: Vec<T>,
    pub field: Field,
    pub offset: Vec<i32>,
}

impl<T> ListVec<T> {
    /// Creates a new ListVec with the specified field schema.
    pub fn new(field: Field) -> Self {
        Self {
            data: Vec::new(),
            field,
            offset: vec![0],
        }
    }

    /// Creates a new ListVec from existing data and field schema.
    pub fn from_vec_with_field(data: Vec<Vec<T>>, field: Field) -> Self {
        let mut offset = vec![0];
        let mut curoff = 0;
        for list in &data {
            curoff += list.len() as i32;
            offset.push(curoff);
        }
        let flat_data = data.into_iter().flatten().collect();
        Self {
            data: flat_data,
            field,
            offset,
        }
    }
}

impl<T> NativeArray for ListVec<T>
where
    T: Clone + Into<DynScalar> + 'static,
{
    type Item = Vec<T>;
    type ItemRef<'a> = &'a Vec<T>;
    type ArrowArray = ListArray;

    fn push(&mut self, item: Self::Item) {
        self.offset
            .push(self.offset.last().unwrap() + item.len() as i32);
        self.data.extend(item);
    }

    fn get(&self, index: usize) -> Self::ItemRef<'_> {
        let slice = &self.data[self.offset[index] as usize..self.offset[index + 1] as usize];
        let vec_ref = unsafe { &*(slice as *const [T] as *const Vec<T>) };
        vec_ref
    }

    /// Converts this ListVec to an Arrow ListArray.
    fn to_arrow_array(&self) -> Self::ArrowArray {
        let values = self.data.iter().map(|x| x.clone().into()).collect();

        let values_array = dynscalar_vec_to_array(values, self.field.data_type());
        let offsets_buffer = OffsetBuffer::new(self.offset.clone().into());
        ListArray::new(
            Arc::new(self.field.clone()),
            offsets_buffer,
            values_array,
            None,
        )
    }
}

pub struct ListOptVec<T> {
    pub data: Vec<Option<Vec<T>>>,
    pub field: Field,
}

impl<T> ListOptVec<T> {
    pub fn new(field: Field) -> Self {
        Self {
            data: Vec::new(),
            field,
        }
    }
    pub fn from_vec_with_field(data: Vec<Option<Vec<T>>>, field: Field) -> Self {
        Self { data, field }
    }
}

impl<T> NativeArray for ListOptVec<T>
where
    T: Clone + Into<DynScalar> + 'static,
{
    type Item = Option<Vec<T>>;
    type ItemRef<'a> = &'a Option<Vec<T>>;
    type ArrowArray = ListArray;

    fn push(&mut self, item: Self::Item) {
        self.data.push(item);
    }

    fn get(&self, index: usize) -> Self::ItemRef<'_> {
        &self.data[index]
    }

    fn to_arrow_array(&self) -> Self::ArrowArray {
        let mut offsets = vec![0i32];
        let mut values = Vec::new();
        let mut validity = Vec::with_capacity(self.data.len());

        for list in &self.data {
            if let Some(list) = list {
                offsets.push(offsets.last().unwrap() + list.len() as i32);
                for item in list {
                    values.push(item.clone().into());
                }
                validity.push(true);
            } else {
                offsets.push(*offsets.last().unwrap());
                validity.push(false);
            }
        }
        //println!("{:?}", values);
        let values_array = dynscalar_vec_to_array(values, self.field.data_type());
        let offsets_buffer = OffsetBuffer::new(offsets.into());
        ListArray::new(
            Arc::new(self.field.clone()),
            offsets_buffer,
            values_array,
            Some(validity.into()),
        )
    }
}

// ========== Map Type Implementation ==========

/// A vector implementation for Arrow Map arrays.
/// Stores nested HashMaps and converts them to Arrow MapArray format.

/// Implement Into<DynScalar> for HashMap where both K and V implement Into<DynScalar>
impl<K, V> Into<DynScalar> for HashMap<K, V>
where
    K: Clone + Into<DynScalar>,
    V: Clone + Into<DynScalar>,
{
    fn into(self) -> DynScalar {
        let pairs: Vec<(DynScalar, DynScalar)> = self
            .into_iter()
            .map(|(k, v)| (k.into(), v.into()))
            .collect();
        DynScalar::Map(pairs)
    }
}

pub struct MapVec<K, V> {
    pub data: Vec<HashMap<K, V>>,
    pub key_field: Field,
    pub value_field: Field,
}

impl<K, V> MapVec<K, V> {
    /// Creates a new MapVec with the specified key and value field schemas.
    pub fn new(key_field: Field, value_field: Field) -> Self {
        Self {
            data: Vec::new(),
            key_field,
            value_field,
        }
    }

    /// Creates a new MapVec from existing data and field schemas.
    pub fn from_vec_with_fields(
        data: Vec<HashMap<K, V>>,
        key_field: Field,
        value_field: Field,
    ) -> Self {
        Self {
            data,
            key_field,
            value_field,
        }
    }
}

impl<K, V> NativeArray for MapVec<K, V>
where
    K: Clone + Into<DynScalar> + 'static,
    V: Clone + Into<DynScalar> + 'static,
{
    type Item = HashMap<K, V>;
    type ItemRef<'a> = &'a HashMap<K, V>;
    type ArrowArray = MapArray;

    fn push(&mut self, item: Self::Item) {
        self.data.push(item);
    }

    fn get(&self, index: usize) -> Self::ItemRef<'_> {
        &self.data[index]
    }

    /// Converts this MapVec to an Arrow MapArray.
    fn to_arrow_array(&self) -> Self::ArrowArray {
        let mut offsets = vec![0i32];
        let mut keys = Vec::new();
        let mut values = Vec::new();

        for map in &self.data {
            offsets.push(offsets.last().unwrap() + map.len() as i32);
            for (k, v) in map {
                keys.push(k.clone().into());
                values.push(v.clone().into());
            }
        }

        let keys_array = dynscalar_vec_to_array(keys, self.key_field.data_type());
        let values_array = dynscalar_vec_to_array(values, self.value_field.data_type());
        let offsets_buffer = OffsetBuffer::new(offsets.into());

        // Create struct fields exactly like Arrow MapBuilder
        let entries_field = Field::new(
            "entries",
            DataType::Struct(vec![self.key_field.clone(), self.value_field.clone()].into()),
            false,
        );

        let struct_array = StructArray::new(
            vec![self.key_field.clone(), self.value_field.clone()].into(),
            vec![keys_array, values_array],
            None,
        );

        MapArray::new(
            Arc::new(entries_field),
            offsets_buffer,
            struct_array,
            None,
            false,
        )
    }
}

pub struct MapOptVec<K, V> {
    pub data: Vec<Option<HashMap<K, V>>>,
    pub key_field: Field,
    pub value_field: Field,
}

impl<K, V> MapOptVec<K, V> {
    pub fn new(key_field: Field, value_field: Field) -> Self {
        Self {
            data: Vec::new(),
            key_field,
            value_field,
        }
    }

    pub fn from_vec_with_fields(
        data: Vec<Option<HashMap<K, V>>>,
        key_field: Field,
        value_field: Field,
    ) -> Self {
        Self {
            data,
            key_field,
            value_field,
        }
    }
}

impl<K, V> NativeArray for MapOptVec<K, V>
where
    K: Clone + Into<DynScalar> + 'static,
    V: Clone + Into<DynScalar> + 'static,
{
    type Item = Option<HashMap<K, V>>;
    type ItemRef<'a> = &'a Option<HashMap<K, V>>;
    type ArrowArray = MapArray;

    fn push(&mut self, item: Self::Item) {
        self.data.push(item)
    }

    fn get(&self, index: usize) -> Self::ItemRef<'_> {
        &self.data[index]
    }

    fn to_arrow_array(&self) -> Self::ArrowArray {
        let mut offsets = vec![0i32];
        let mut keys = Vec::new();
        let mut values = Vec::new();
        let mut validity = Vec::with_capacity(self.data.len());

        for map in &self.data {
            if let Some(map) = map {
                offsets.push(offsets.last().unwrap() + map.len() as i32);
                for (k, v) in map {
                    keys.push(k.clone().into());
                    values.push(v.clone().into());
                }
                validity.push(true);
            } else {
                offsets.push(*offsets.last().unwrap());
                validity.push(false);
            }
        }

        let keys_array = dynscalar_vec_to_array(keys, self.key_field.data_type());
        let values_array = dynscalar_vec_to_array(values, self.value_field.data_type());
        let offsets_buffer = OffsetBuffer::new(offsets.into());

        let entries_field = Field::new(
            "entries",
            DataType::Struct(vec![self.key_field.clone(), self.value_field.clone()].into()),
            false,
        );

        let struct_array = StructArray::new(
            vec![self.key_field.clone(), self.value_field.clone()].into(),
            vec![keys_array, values_array],
            None,
        );

        let nulls = if validity.iter().all(|&x| x) {
            None
        } else {
            Some(arrow::buffer::NullBuffer::from(
                arrow::buffer::BooleanBuffer::from(validity),
            ))
        };

        MapArray::new(
            Arc::new(entries_field),
            offsets_buffer,
            struct_array,
            nulls,
            false,
        )
    }
}
// ========== FixedSizeList Type Implementation ==========

/// A vector implementation for Arrow FixedSizeList arrays.
/// Stores vectors of fixed size and converts them to Arrow FixedSizeListArray format.

/// Implement Into<DynScalar> for (Vec<T>, i32) (FixedSizeList)
impl<T> Into<DynScalar> for (Vec<T>, i32)
where
    T: Into<DynScalar>,
{
    fn into(self) -> DynScalar {
        let (vec, size) = self;
        let scalars: Vec<DynScalar> = vec.into_iter().map(|item| item.into()).collect();
        DynScalar::FixedSizeList(scalars, size)
    }
}

pub struct FixedSizeListVec<T> {
    pub data: Vec<Vec<T>>,
    pub field: Field,
    pub size: i32,
}

impl<T> FixedSizeListVec<T> {
    /// Creates a new FixedSizeListVec with the specified field schema and size.
    pub fn new(field: Field, size: i32) -> Self {
        Self {
            data: Vec::new(),
            field,
            size,
        }
    }

    /// Creates a new FixedSizeListVec from existing data, field schema, and size.
    pub fn from_vec_with_field(data: Vec<Vec<T>>, field: Field, size: i32) -> Self {
        Self { data, field, size }
    }
}

impl<T> NativeArray for FixedSizeListVec<T>
where
    T: Clone + Into<DynScalar> + 'static,
{
    type Item = Vec<T>;
    type ItemRef<'a> = &'a Vec<T>;
    type ArrowArray = arrow::array::FixedSizeListArray;

    fn push(&mut self, item: Self::Item) {
        if item.len() != self.size as usize {
            panic!(
                "FixedSizeList size mismatch: expected {}, got {}",
                self.size,
                item.len()
            );
        }
        self.data.push(item);
    }

    fn get(&self, index: usize) -> Self::ItemRef<'_> {
        &self.data[index]
    }

    /// Converts this FixedSizeListVec to an Arrow FixedSizeListArray.
    fn to_arrow_array(&self) -> Self::ArrowArray {
        let mut values = Vec::new();

        for list in &self.data {
            for item in list {
                values.push(item.clone().into());
            }
        }

        let values_array = dynscalar_vec_to_array(values, self.field.data_type());

        arrow::array::FixedSizeListArray::new(
            Arc::new(self.field.clone()),
            self.size,
            values_array,
            None,
        )
    }
}

pub struct FixedSizeListOptVec<T> {
    pub data: Vec<Option<Vec<T>>>,
    pub field: Field,
    pub size: i32,
}

impl<T> FixedSizeListOptVec<T> {
    pub fn new(field: Field, size: i32) -> Self {
        Self {
            data: Vec::new(),
            field,
            size,
        }
    }

    pub fn from_vec_with_field(data: Vec<Option<Vec<T>>>, field: Field, size: i32) -> Self {
        Self { data, field, size }
    }
}

impl<T> NativeArray for FixedSizeListOptVec<T>
where
    T: Clone + Into<DynScalar> + 'static,
{
    type Item = Option<Vec<T>>;
    type ItemRef<'a> = &'a Option<Vec<T>>;
    type ArrowArray = FixedSizeListArray;

    fn push(&mut self, item: Self::Item) {
        if item.as_ref().unwrap().len() != self.size as usize {
            panic!(
                "FixedSizeList size mismatch: expected {}, got {}",
                self.size,
                item.as_ref().unwrap().len()
            );
        }
        self.data.push(item);
    }

    fn get(&self, index: usize) -> Self::ItemRef<'_> {
        &self.data[index]
    }

    fn to_arrow_array(&self) -> Self::ArrowArray {
        let mut values = Vec::new();
        let mut validity = Vec::with_capacity(self.data.len());

        for list in &self.data {
            if let Some(list) = list {
                for item in list {
                    values.push(item.clone().into());
                }
                validity.push(true);
            } else {
                validity.push(false);
            }
        }

        let values_array = dynscalar_vec_to_array(values, self.field.data_type());

        arrow::array::FixedSizeListArray::new(
            Arc::new(self.field.clone()),
            self.size,
            values_array,
            Some(validity.into()),
        )
    }
}

// ========== Struct Type Implementation ==========

/// A vector implementation for Arrow Struct arrays.
/// Stores rows as HashMaps and converts them to Arrow StructArray format.
pub struct StructVec {
    pub data: Vec<HashMap<String, DynScalar>>,
    pub schema: Schema,
}

impl StructVec {
    /// Creates a new StructVec with the specified schema.
    pub fn new(schema: Schema) -> Self {
        Self {
            data: Vec::new(),
            schema,
        }
    }

    // Creates a new StructVec from existing data and schema.
    /* pub fn from_vec_with_schema(data: Vec<HashMap<String, DynScalar>>, schema: Schema) -> Self {
        Self { data, schema }
    } */
}

impl NativeArray for StructVec {
    type Item = HashMap<String, DynScalar>;
    type ItemRef<'a> = &'a HashMap<String, DynScalar>;
    type ArrowArray = StructArray;

    fn push(&mut self, item: Self::Item) {
        self.data.push(item);
    }

    fn get(&self, index: usize) -> Self::ItemRef<'_> {
        &self.data[index]
    }

    /// Converts this StructVec to an Arrow StructArray.
    fn to_arrow_array(&self) -> Self::ArrowArray {
        let mut field_arrays: Vec<ArrayRef> = Vec::new();

        for field in self.schema.fields() {
            let field_name = field.name();
            let field_values: Vec<DynScalar> = self
                .data
                .iter()
                .map(|row| {
                    row.get(field_name)
                        .cloned()
                        .unwrap_or_else(|| default_dyn_scalar(field.data_type()))
                })
                .collect();

            let array = dynscalar_vec_to_array(field_values, field.data_type());
            field_arrays.push(array);
        }
        for (field, arr) in self.schema.fields().iter().zip(field_arrays.iter()) {
            assert_eq!(
                arr.data_type(),
                field.data_type(),
                "Field {:?} expects {:?}, got {:?}",
                field.name(),
                field.data_type(),
                arr.data_type()
            );
        }

        StructArray::new(self.schema.fields().clone(), field_arrays, None)
    }
}

pub struct StructOptVec {
    pub data: Vec<Option<HashMap<String, DynScalar>>>,
    pub schema: Schema,
}

impl StructOptVec {
    pub fn new(schema: Schema) -> Self {
        Self {
            data: Vec::new(),
            schema,
        }
    }

    pub fn from_vec_with_schema(
        data: Vec<Option<HashMap<String, DynScalar>>>,
        schema: Schema,
    ) -> Self {
        Self { data, schema }
    }
}

impl NativeArray for StructOptVec {
    type Item = Option<HashMap<String, DynScalar>>;
    type ItemRef<'a> = &'a Option<HashMap<String, DynScalar>>;
    type ArrowArray = StructArray;

    fn push(&mut self, item: Self::Item) {
        self.data.push(item);
    }

    fn get(&self, index: usize) -> Self::ItemRef<'_> {
        &self.data[index]
    }

    /// Converts this StructVec to an Arrow StructArray.
    fn to_arrow_array(&self) -> Self::ArrowArray {
        let mut field_arrays: Vec<ArrayRef> = Vec::new();
        let validity: Vec<bool> = self.data.iter().map(|row| row.is_some()).collect();

        for field in self.schema.fields() {
            let field_name = field.name();
            let field_values: Vec<DynScalar> = self
                .data
                .iter()
                .map(|opt_row| match opt_row {
                    None => DynScalar::Null,
                    Some(row) => row
                        .get(field_name)
                        .cloned()
                        .unwrap_or_else(|| default_dyn_scalar(field.data_type())),
                })
                .collect();
            println!("{:?}", field_values);
            let array = dynscalar_vec_to_array(field_values, field.data_type());
            field_arrays.push(array);
        }
        for (field, arr) in self.schema.fields().iter().zip(field_arrays.iter()) {
            assert_eq!(
                arr.data_type(),
                field.data_type(),
                "Field {:?} expects {:?}, got {:?}",
                field.name(),
                field.data_type(),
                arr.data_type()
            );
        }

        StructArray::new(
            self.schema.fields().clone(),
            field_arrays,
            Some(validity.into()),
        )
    }
}

// ========== TESTS ==========

#[cfg(test)]
mod tests {
    use super::*;
    //use crate::register_struct;
    use arrow::array::*;
    use derive_dynscalar::IntoDynScalar;
    use std::collections::HashMap;

    #[test]
    fn test_list_vec_nested_struct() {
        // Create a nested struct containing a list of integers
        let inner_field = Field::new("item", DataType::Int32, false);
        let _list_field = Field::new(
            "numbers",
            DataType::List(Arc::new(inner_field.clone())),
            false,
        );

        let mut list_vec = ListVec::new(inner_field);
        list_vec.push(vec![1, 2, 3]);
        list_vec.push(vec![4, 5]);
        list_vec.push(vec![6, 7, 8, 9]);

        let arrow_array = list_vec.to_arrow_array();

        // Test direct construction using arrow-rs
        let expected = {
            let values = Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
            let offsets = OffsetBuffer::new(vec![0, 3, 5, 9].into());
            ListArray::new(
                Arc::new(Field::new("item", DataType::Int32, false)),
                offsets,
                Arc::new(values),
                None,
            )
        };

        assert_eq!(arrow_array, expected);
        assert_eq!(arrow_array.len(), expected.len());
        assert_eq!(arrow_array.data_type(), expected.data_type());

        // Verify the values match
        for i in 0..arrow_array.len() {
            let our_list = arrow_array.value(i);
            let expected_list = expected.value(i);
            assert_eq!(our_list.as_ref(), expected_list.as_ref());
        }
    }

    #[test]
    fn test_map_vec() {
        // Test converting Vec<HashMap<i32, String>> to MapVec vs Arrow MapArray builder
        use arrow::array::{Int32Builder, MapBuilder, StringBuilder};

        let key_field = Field::new("keys", DataType::Int32, false);
        let value_field = Field::new("values", DataType::Utf8, false);

        let mut map1 = HashMap::new();
        map1.insert(1, "first".to_string());
        map1.insert(2, "second".to_string());

        let mut map2 = HashMap::new();
        map2.insert(3, "third".to_string());
        map2.insert(4, "fourth".to_string());

        let native_data = vec![map1.clone(), map2.clone()];

        let map_vec = MapVec::from_vec_with_fields(
            native_data.clone(),
            key_field.clone(),
            value_field.clone(),
        );
        let our_array = map_vec.to_arrow_array();

        let mut builder = MapBuilder::new(None, Int32Builder::new(), StringBuilder::new());

        for (k, v) in &map1 {
            builder.keys().append_value(*k);
            builder.values().append_value(v);
        }
        builder.append(true).unwrap();

        for (k, v) in &map2 {
            builder.keys().append_value(*k);
            builder.values().append_value(v);
        }
        builder.append(true).unwrap();

        let expected_array = builder.finish();

        // Test basic properties
        assert_eq!(our_array.len(), expected_array.len());
        assert_eq!(our_array.len(), 2);

        // Verify data types are compatible
        assert!(matches!(our_array.data_type(), DataType::Map(_, _)));
        assert!(matches!(expected_array.data_type(), DataType::Map(_, _)));

        // Verify individual map entries from our implementation
        assert_eq!(map_vec.get(0), &map1);
        assert_eq!(map_vec.get(1), &map2);

        // Verify the arrays have the same structure by inspecting their content
        for i in 0..our_array.len() {
            assert_eq!(our_array.is_valid(i), expected_array.is_valid(i));
        }

        // Note: Direct equality may fail due to HashMap iteration order differences
        // but the arrays are logically equivalent
        // assert_eq!(our_array, expected_array);
    }

    #[test]
    fn test_map_vec_nested() {
        // Test converting Vec<HashMap<i64, HashMap<String, i64>>>
        use arrow::array::{Int64Builder, MapBuilder, StringBuilder};

        let key_field = Field::new("keys", DataType::Int64, false);

        let nested_key_field = Field::new("keys", DataType::Utf8, false);
        let nested_value_field = Field::new("values", DataType::Int64, true); // Arrow MapBuilder uses nullable values
        let nested_entries_field = Field::new(
            "entries",
            DataType::Struct(vec![nested_key_field.clone(), nested_value_field.clone()].into()),
            false,
        );
        let value_field = Field::new(
            "values",
            DataType::Map(Arc::new(nested_entries_field), false),
            false,
        );

        let mut inner_map1 = HashMap::new();
        inner_map1.insert("score".to_string(), 95i64);
        inner_map1.insert("rank".to_string(), 1i64);

        let mut inner_map2 = HashMap::new();
        inner_map2.insert("score".to_string(), 87i64);
        inner_map2.insert("rank".to_string(), 3i64);

        let mut map1 = HashMap::new();
        map1.insert(100i64, inner_map1.clone());
        map1.insert(200i64, inner_map2.clone());

        let mut inner_map3 = HashMap::new();
        inner_map3.insert("score".to_string(), 92i64);

        let mut map2 = HashMap::new();
        map2.insert(300i64, inner_map3.clone());

        let native_data = vec![map1.clone(), map2.clone()];

        let map_vec = MapVec::from_vec_with_fields(
            native_data.clone(),
            key_field.clone(),
            value_field.clone(),
        );
        let our_array = map_vec.to_arrow_array();

        let nested_map_builder = MapBuilder::new(None, StringBuilder::new(), Int64Builder::new());
        let mut builder = MapBuilder::new(None, Int64Builder::new(), nested_map_builder);

        for map in &native_data {
            for (outer_key, inner_map) in map {
                builder.keys().append_value(*outer_key);

                let nested_builder = builder.values();
                for (inner_key, inner_value) in inner_map {
                    nested_builder.keys().append_value(inner_key);
                    nested_builder.values().append_value(*inner_value);
                }
                nested_builder.append(true).unwrap();
            }
            builder.append(true).unwrap();
        }

        let expected_array = builder.finish();

        // Test basic properties
        assert_eq!(our_array.len(), expected_array.len());
        assert_eq!(our_array.len(), 2);

        // Verify data types are compatible
        assert!(matches!(our_array.data_type(), DataType::Map(_, _)));
        assert!(matches!(expected_array.data_type(), DataType::Map(_, _)));

        // Verify individual map entries from our implementation
        assert_eq!(map_vec.get(0), &map1);
        assert_eq!(map_vec.get(1), &map2);

        // Verify the arrays have the same structure by inspecting their content
        for i in 0..our_array.len() {
            assert_eq!(our_array.is_valid(i), expected_array.is_valid(i));
        }

        // assert_eq!(our_array, expected_array);
    }

    #[test]
    fn test_map_opt_vec() {
        // Test converting Vec<Option<HashMap<i32, String>>>
        use arrow::array::{Int32Builder, MapBuilder, StringBuilder};

        let key_field = Field::new("keys", DataType::Int32, false);
        let value_field = Field::new("values", DataType::Utf8, false);

        let mut map1 = HashMap::new();
        map1.insert(1, "first".to_string());
        map1.insert(2, "second".to_string());

        let mut map2 = HashMap::new();
        map2.insert(3, "third".to_string());
        map2.insert(4, "fourth".to_string());

        let native_data = vec![
            Some(map1.clone()),
            None,
            Some(map2.clone()),
            Some(HashMap::new()),
        ];

        let map_vec = MapOptVec::from_vec_with_fields(
            native_data.clone(),
            key_field.clone(),
            value_field.clone(),
        );
        let our_array = map_vec.to_arrow_array();

        let mut builder = MapBuilder::new(None, Int32Builder::new(), StringBuilder::new());

        for opt_map in &native_data {
            match opt_map {
                Some(map) => {
                    let mut sorted_pairs: Vec<_> = map.iter().collect();
                    sorted_pairs.sort_by_key(|&(k, _)| k);
                    for (k, v) in sorted_pairs {
                        builder.keys().append_value(*k);
                        builder.values().append_value(v);
                    }
                    builder.append(true).unwrap();
                }
                None => {
                    builder.append(false).unwrap(); // null map
                }
            }
        }

        let expected_array = builder.finish();

        // Test basic properties
        assert_eq!(our_array.len(), expected_array.len());
        assert_eq!(our_array.len(), 4);

        // Verify data types are compatible
        assert!(matches!(our_array.data_type(), DataType::Map(_, _)));
        assert!(matches!(expected_array.data_type(), DataType::Map(_, _)));

        // Verify validity matches
        assert!(our_array.is_valid(0)); // Some(map1)
        assert!(!our_array.is_valid(1)); // None
        assert!(our_array.is_valid(2)); // Some(map2)
        assert!(our_array.is_valid(3)); // Some(empty map)

        assert!(expected_array.is_valid(0));
        assert!(!expected_array.is_valid(1));
        assert!(expected_array.is_valid(2));
        assert!(expected_array.is_valid(3));

        // Verify individual map entries from our implementation
        assert_eq!(map_vec.get(0), &Some(map1));
        assert_eq!(map_vec.get(1), &None);
        assert_eq!(map_vec.get(2), &Some(map2));
        assert_eq!(map_vec.get(3), &Some(HashMap::new()));

        // Verify the arrays have the same structure by inspecting their content
        for i in 0..our_array.len() {
            assert_eq!(our_array.is_valid(i), expected_array.is_valid(i));
        }

        // Note: Direct equality may fail due to HashMap iteration order differences
        // but the arrays are logically equivalent
        // assert_eq!(our_array, expected_array);
    }

    #[test]
    fn test_list_vec_from_vec_with_field() {
        use arrow::array::{Int32Array, ListArray};
        use std::sync::Arc;

        let data = vec![vec![1, 2, 3], vec![4, 5], vec![], vec![6]];
        let inner_field = Field::new("item", DataType::Int32, false);

        let list_vec = ListVec::from_vec_with_field(data.clone(), inner_field.clone());

        let arrow_array = list_vec.to_arrow_array();
        assert_eq!(arrow_array.len(), data.len());

        let values: Vec<i32> = data.iter().flatten().copied().collect();
        let expected_values = Int32Array::from(values);
        let mut offsets = vec![0_i32];
        for v in &data {
            offsets.push(offsets.last().unwrap() + v.len() as i32);
        }
        let expected = ListArray::new(
            Arc::new(inner_field),
            OffsetBuffer::new(offsets.into()),
            Arc::new(expected_values),
            None,
        );

        assert_eq!(arrow_array, expected);
    }

    #[test]
    fn test_struct_vec() {
        // Define Person struct for testing
        #[derive(IntoDynScalar)]
        struct Person {
            name: String,
            age: i32,
            email: Option<String>,
        }

        // Create schema
        let fields = vec![
            Field::new("name", DataType::Utf8, false),
            Field::new("age", DataType::Int32, false),
            Field::new("email", DataType::Utf8, true), // nullable
        ];
        let _schema = Schema::new(fields.clone());

        // Create test data
        let people = vec![
            Person {
                name: "Alice".to_string(),
                age: 30,
                email: Some("alice@example.com".to_string()),
            },
            Person {
                name: "Bob".to_string(),
                age: 25,
                email: None,
            },
            Person {
                name: "Charlie".to_string(),
                age: 35,
                email: Some("charlie@example.com".to_string()),
            },
        ];

        // Create schema
        let fields = vec![
            Field::new("name", DataType::Utf8, false),
            Field::new("age", DataType::Int32, false),
            Field::new("email", DataType::Utf8, true), 
        ];
        let _schema2 = Schema::new(fields.clone());

        // Convert using SchemaConvertible
        //let converted = convert_vector_with_schema(&people, &_schema);
        //let converted = Vec::new();

        // Convert to StructVec and then to Arrow array
        //let struct_vec = StructVec::from_vec_with_schema(converted.clone(), _schema.clone());
        let converted: Vec<DynScalar> = people.into_iter().map(|x| x.into()).collect();
        let arrow_array = dynscalar_vec_to_array(converted, &DataType::Struct(fields.clone().into()));

        // Create expected array using StructBuilder
        let expected = {
            let id_array = Arc::new(arrow::array::StringArray::from(vec![
                "Alice", "Bob", "Charlie",
            ]));
            let age_array = Arc::new(arrow::array::Int32Array::from(vec![30, 25, 35]));
            let email_array = Arc::new(arrow::array::StringArray::from(vec![
                Some("alice@example.com"),
                None,
                Some("charlie@example.com"),
            ]));

            StructArray::new(
                _schema.fields().clone(),
                vec![id_array, age_array, email_array],
                None,
            )
        };

        // Assert equality
        assert_eq!(arrow_array.to_data(), expected.to_data());
        assert_eq!(arrow_array.len(), expected.len());
        assert_eq!(arrow_array.data_type(), expected.data_type());
    }

    #[test]
    fn test_struct_vec_nested() {
        // Create a nested struct with multiple fields
        let fields = vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("score", DataType::Float64, false),
        ];
        let schema = Schema::new(fields);

        let mut struct_vec = StructVec::new(schema.clone());

        let mut row1 = HashMap::new();
        row1.insert("id".to_string(), DynScalar::Int32(1));
        row1.insert("name".to_string(), DynScalar::String("Alice".to_string()));
        row1.insert("score".to_string(), DynScalar::Float64(95.5));
        struct_vec.push(row1);

        let mut row2 = HashMap::new();
        row2.insert("id".to_string(), DynScalar::Int32(2));
        row2.insert("name".to_string(), DynScalar::String("Bob".to_string()));
        row2.insert("score".to_string(), DynScalar::Float64(87.2));
        struct_vec.push(row2);

        let arrow_array = struct_vec.to_arrow_array();

        let expected = {
            let id_array = Arc::new(Int32Array::from(vec![1, 2]));
            let name_array = Arc::new(StringArray::from(vec!["Alice", "Bob"]));
            let score_array = Arc::new(Float64Array::from(vec![95.5, 87.2]));

            StructArray::new(
                schema.fields().clone(),
                vec![id_array, name_array, score_array],
                None,
            )
        };

        assert_eq!(arrow_array.len(), expected.len());
        assert_eq!(arrow_array.data_type(), expected.data_type());

        // Verify individual column values
        for _i in 0..arrow_array.len() {
            for (col_idx, _field) in schema.fields().iter().enumerate() {
                let our_col = arrow_array.column(col_idx);
                let expected_col = expected.column(col_idx);
                assert_eq!(our_col.as_ref(), expected_col.as_ref());
            }
        }

        assert_eq!(arrow_array, expected);
    }

    #[test]
    fn test_nested_struct_with_i64_and_list() {
        // Test a proper nested structure: struct containing i64 and List fields
        let list_field = Field::new("item", DataType::Int32, false);
        let fields = vec![
            Field::new("id", DataType::Int64, false),
            Field::new("tags", DataType::List(Arc::new(list_field)), false),
        ];
        let schema = Schema::new(fields);

        let mut struct_vec = StructVec::new(schema.clone());

        let mut row1 = HashMap::new();
        row1.insert("id".to_string(), DynScalar::Int64(100));
        row1.insert(
            "tags".to_string(),
            DynScalar::List(vec![
                DynScalar::Int32(1),
                DynScalar::Int32(2),
                DynScalar::Int32(3),
            ]),
        );
        struct_vec.push(row1);

        let mut row2 = HashMap::new();
        row2.insert("id".to_string(), DynScalar::Int64(200));
        row2.insert(
            "tags".to_string(),
            DynScalar::List(vec![DynScalar::Int32(4), DynScalar::Int32(5)]),
        );
        struct_vec.push(row2);

        let arrow_array = struct_vec.to_arrow_array();

        let expected = {
            let id_array = Arc::new(Int64Array::from(vec![100, 200]));

            let list_values = Int32Array::from(vec![1, 2, 3, 4, 5]);
            let list_offsets = OffsetBuffer::new(vec![0, 3, 5].into());
            let list_field = Field::new("item", DataType::Int32, false);
            let tags_array = Arc::new(ListArray::new(
                Arc::new(list_field),
                list_offsets,
                Arc::new(list_values),
                None,
            ));

            StructArray::new(schema.fields().clone(), vec![id_array, tags_array], None)
        };

        assert_eq!(arrow_array.len(), expected.len());
        assert_eq!(arrow_array.data_type(), expected.data_type());

        // Verify the data types of columns
        let id_column = arrow_array.column(0);
        let tags_column = arrow_array.column(1);

        assert_eq!(id_column.data_type(), &DataType::Int64);
        assert_eq!(
            tags_column.data_type(),
            &DataType::List(Arc::new(Field::new("item", DataType::Int32, false)))
        );

        // Verify the actual values
        let id_array = id_column.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(id_array.value(0), 100);
        assert_eq!(id_array.value(1), 200);

        let tags_array = tags_column.as_any().downcast_ref::<ListArray>().unwrap();
        let first_list = tags_array.value(0);
        let first_list_ints = first_list.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(first_list_ints.values(), &[1, 2, 3]);

        let second_list = tags_array.value(1);
        let second_list_ints = second_list.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(second_list_ints.values(), &[4, 5]);

        assert_eq!(arrow_array, expected);
    }

    #[test]
    fn test_nested_struct_with_map() {
        // Test a struct containing a Map field
        let map_field = Field::new(
            "entries",
            DataType::Struct(
                vec![
                    Field::new("key", DataType::Utf8, false),
                    Field::new("value", DataType::Int32, false),
                ]
                .into(),
            ),
            false,
        );
        let fields = vec![
            Field::new("id", DataType::Int64, false),
            Field::new("metadata", DataType::Map(Arc::new(map_field), false), false),
        ];
        let schema = Schema::new(fields);

        let mut struct_vec = StructVec::new(schema.clone());

        let mut row1 = HashMap::new();
        row1.insert("id".to_string(), DynScalar::Int64(100));
        row1.insert(
            "metadata".to_string(),
            DynScalar::Map(vec![
                (DynScalar::String("name".to_string()), DynScalar::Int32(42)),
                (DynScalar::String("age".to_string()), DynScalar::Int32(30)),
            ]),
        );
        struct_vec.push(row1);

        let arrow_array = struct_vec.to_arrow_array();

        assert_eq!(arrow_array.len(), 1);
        assert_eq!(arrow_array.num_columns(), 2);

        // Verify the data types
        let id_column = arrow_array.column(0);
        let metadata_column = arrow_array.column(1);

        assert_eq!(id_column.data_type(), &DataType::Int64);
        assert!(matches!(metadata_column.data_type(), DataType::Map(_, _)));

        // Verify the ID value
        let id_array = id_column.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(id_array.value(0), 100);

        // Verify the map structure exists (detailed verification would be complex)
        let map_array = metadata_column.as_any().downcast_ref::<MapArray>().unwrap();
        assert_eq!(map_array.len(), 1);
    }

    #[test]
    fn test_nested_struct_empty() {
        // Test an empty StructVec
        let fields = vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ];
        let schema = Schema::new(fields);

        let struct_vec = StructVec::new(schema.clone());

        let arrow_array = struct_vec.to_arrow_array();

        assert_eq!(arrow_array.len(), 0);
        assert_eq!(arrow_array.num_columns(), 2);

        // Verify the data types
        assert_eq!(arrow_array.column(0).data_type(), &DataType::Int64);
        assert_eq!(arrow_array.column(1).data_type(), &DataType::Utf8);
    }

    #[test]
    fn test_struct_option_vec() {
        let fields = vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ];
        let schema = Schema::new(fields);

        let mut struct_vec = StructOptVec::new(schema.clone());

        let mut data = HashMap::new();
        data.insert("id".to_string(), DynScalar::Int64(1));
        data.insert("name".to_string(), DynScalar::String("udf".to_string()));

        struct_vec.push(None);
        struct_vec.push(Some(data));

        let arrow_array = struct_vec.to_arrow_array();

        let expected = StructArray::new(
            schema.fields.clone(),
            vec![
                Arc::new(Int64Array::from(vec![None, Some(1)])),
                Arc::new(StringArray::from(vec![None, Some("udf")])),
            ],
            Some(vec![false, true].into()),
        );

        assert_eq!(arrow_array, expected);
    }

    #[test]
    fn test_nested_struct_with_fixed_size_list() {
        // Test a struct containing a FixedSizeList field (using our implementation)
        let fields = vec![
            Field::new("id", DataType::Int64, false),
            Field::new(
                "coordinates",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float64, false)),
                    3, // Fixed size of 3 (x, y, z coordinates)
                ),
                false,
            ),
        ];
        let schema = Schema::new(fields);

        let mut struct_vec = StructVec::new(schema.clone());

        // Add a row with fixed-size list data
        let mut row1 = HashMap::new();
        row1.insert("id".to_string(), DynScalar::Int64(100));
        row1.insert(
            "coordinates".to_string(),
            DynScalar::FixedSizeList(
                vec![
                    DynScalar::Float64(1.0),
                    DynScalar::Float64(2.0),
                    DynScalar::Float64(3.0),
                ],
                3,
            ),
        );
        struct_vec.push(row1);

        let arrow_array = struct_vec.to_arrow_array();

        assert_eq!(arrow_array.len(), 1);
        assert_eq!(arrow_array.num_columns(), 2);

        // Verify the data types
        let id_column = arrow_array.column(0);
        let coordinates_column = arrow_array.column(1);

        assert_eq!(id_column.data_type(), &DataType::Int64);
        assert!(matches!(
            coordinates_column.data_type(),
            DataType::FixedSizeList(_, 3)
        ));

        // Verify the ID value
        let id_array = id_column.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(id_array.value(0), 100);

        // Verify the fixed size list structure exists
        let coord_array = coordinates_column
            .as_any()
            .downcast_ref::<arrow::array::FixedSizeListArray>()
            .unwrap();
        assert_eq!(coord_array.len(), 1);
        assert_eq!(coord_array.value_length(), 3);
    }

    #[test]
    fn test_list_vec_nullable_arrow() {
        use arrow::array::{Int32Builder, ListBuilder};
        use arrow::datatypes::{DataType, Field};

        let data: Vec<Option<Vec<Option<i32>>>> = vec![
            None,
            Some(vec![Some(1), None, Some(3)]), // [1, null, 3]
            Some(vec![]),                       // []
            Some(vec![Some(4), Some(5)]),       // [4,5]
        ];

        let inner_field = Field::new("item", DataType::Int32, true);

        let mut builder =
            ListBuilder::with_field(ListBuilder::new(Int32Builder::new()), inner_field.clone());
        for v in &data {
            match v {
                Some(subvec) => {
                    for i in subvec {
                        match i {
                            Some(x) => builder.values().append_value(*x),
                            None => builder.values().append_null(),
                        }
                    }
                    builder.append(true);
                }
                None => {
                    builder.append(false);
                }
            }
        }
        let expected = builder.finish();

        let input: Vec<Option<Vec<DynScalar>>> = data
            .iter()
            .map(|opt| {
                opt.as_ref().map(|v| {
                    v.iter()
                        .map(|vi| match vi {
                            Some(x) => DynScalar::Int32(*x),
                            None => DynScalar::Null,
                        })
                        .collect()
                })
            })
            .collect();

        let list_vec = ListOptVec::from_vec_with_field(input, inner_field);
        let arrow_array = list_vec.to_arrow_array();

        assert_eq!(arrow_array, expected);
    }

    use arrow::array::{Array, Int32Builder, ListBuilder, StringBuilder, StructBuilder};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn test_complex_struct() {
        // Define the nested struct for testing
        #[derive(IntoDynScalar)]
        struct NestedStruct {
            id: i32,
            description: String,
        }

        // Define the main struct for testing
        #[derive(IntoDynScalar)]
        struct ComplexStruct {
            nested_list: Vec<Vec<i32>>,
            map_field: HashMap<String, HashMap<i32, String>>,
            nested_struct: NestedStruct,
        }

        // Create schema for the nested struct
        let nested_struct_fields = vec![
            Field::new("id", DataType::Int32, true),
            Field::new("description", DataType::Utf8, true),
        ];
        let _nested_struct_schema = Schema::new(nested_struct_fields.clone());

        let inner_map_field = Field::new(
            "entries",
            DataType::Struct(
                vec![
                    Field::new("keys", DataType::Int32, true),    
                    Field::new("values", DataType::Utf8, true),   
                ]
                .into(),
            ),
            false,  
        );
        
        let map_field = Field::new(
            "entries",
            DataType::Struct(
                vec![
                    Field::new("keys", DataType::Utf8, true),     
                    Field::new(
                        "values",                                
                        DataType::Map(Arc::new(inner_map_field.clone()), false), 
                        true,  
                    ),
                ]
                .into(),
            ),
            false,  
        );
        
        // Create schema for the main struct
        let fields = vec![
            Field::new(
                "nested_list",
                DataType::List(Arc::new(Field::new(
                    "item",
                    DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
                    true,
                ))),
                true,
            ),
            Field::new(
                "map_field",
                DataType::Map(Arc::new(map_field.clone()), false), 
                true,
            ),
            Field::new(
                "nested_struct",
                DataType::Struct(nested_struct_fields.clone().into()),
                true,
            ),
        ];

        let test_data = vec![ComplexStruct {
            nested_list: vec![vec![1, 2, 3], vec![4, 5]],
            map_field: {
                let mut inner_map = HashMap::new();
                inner_map.insert(1, "One".to_string());
                inner_map.insert(2, "Two".to_string());
                let mut outer_map = HashMap::new();
                outer_map.insert("Numbers".to_string(), inner_map);
                outer_map
            },
            nested_struct: NestedStruct {
                id: 42,
                description: "Test description".to_string(),
            },
        }];

        // deprecated approach
        //let converted = test_data.into_iter().map(|x| x.into()).collect::<Vec<HashMap<_,_>>>();
        //let struct_vec = StructVec::from_vec_with_schema(converted.clone(), schema.clone());
        //let arrow_array = struct_vec.to_arrow_array();

        let converted: Vec<DynScalar> = test_data.into_iter().map(|x| x.into()).collect();
        let arrow_array = dynscalar_vec_to_array(converted.clone(), &DataType::Struct(fields.clone().into()));

        // Assert conversion and Arrow array properties
        assert_eq!(converted.len(), 1);
        assert_eq!(arrow_array.len(), 1);
        assert_eq!(
            arrow_array.data_type(),
            &DataType::Struct(fields.clone().into())
        );

        // nested_list
        let nested_list_builder = ListBuilder::new(Int32Builder::new());
        let mut outer_list_builder = ListBuilder::new(nested_list_builder);

        let nested_lists = &[vec![1, 2, 3], vec![4, 5]];
        for sublist in nested_lists {
            let values_builder = outer_list_builder.values();
            for v in sublist {
                values_builder.values().append_value(*v);
            }
            values_builder.append(true);
        }
        outer_list_builder.append(true);
        let nested_list_array = Arc::new(outer_list_builder.finish()) as ArrayRef;

        // map_field
        let inner_keys_array = Int32Array::from(vec![1, 2]);
        let inner_values_array = StringArray::from(vec!["One", "Two"]);

        let inner_struct_array = StructArray::from(vec![
            (
                Arc::new(Field::new("keys", DataType::Int32, true)), 
                Arc::new(inner_keys_array) as ArrayRef,
            ),
            (
                Arc::new(Field::new("values", DataType::Utf8, true)), 
                Arc::new(inner_values_array) as ArrayRef,
            ),
        ]);

        let inner_map_array = MapArray::new(
            Arc::new(inner_map_field.clone()),
            OffsetBuffer::from_lengths([2]), 
            inner_struct_array,
            None,
            false, 
        );

        let outer_keys_array = StringArray::from(vec!["Numbers"]);

        let outer_struct_array = StructArray::from(vec![
            (
                Arc::new(Field::new("keys", DataType::Utf8, true)), 
                Arc::new(outer_keys_array) as ArrayRef,
            ),
            (
                Arc::new(Field::new("values", DataType::Map(Arc::new(inner_map_field.clone()), false), true)),
                Arc::new(inner_map_array) as ArrayRef,
            ),
        ]);

        let outer_map_array = MapArray::new(
            Arc::new(map_field.clone()),
            OffsetBuffer::from_lengths([1]), 
            outer_struct_array,
            None,
            false, 
        );
        let map_field_array = Arc::new(outer_map_array) as ArrayRef;


        // nested_struct
        let mut id_builder = Int32Builder::new();
        let mut desc_builder = StringBuilder::new();
        id_builder.append_value(42);
        desc_builder.append_value("Test description");
        let mut nested_struct_builder = StructBuilder::new(
            nested_struct_fields.clone(),
            vec![Box::new(id_builder), Box::new(desc_builder)],
        );
        nested_struct_builder.append(true);
        let nested_struct_array = Arc::new(nested_struct_builder.finish()) as ArrayRef;

        // Integrate
        let struct_array = StructArray::from(vec![
            (
                Arc::new(Field::new(
                    "nested_list",
                    DataType::List(Arc::new(Field::new(
                        "item",
                        DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
                        true,
                    ))),
                    true,
                )),
                nested_list_array,
            ),
            (
                Arc::new(Field::new(
                    "map_field",
                    DataType::Map(Arc::new(map_field.clone()), false),
                    true,
                )),
                map_field_array,
            ),
            (
                Arc::new(Field::new(
                    "nested_struct",
                    DataType::Struct(nested_struct_fields.clone().into()),
                    true,
                )),
                nested_struct_array,
            ),
        ]);

        assert_eq!(arrow_array.len(), struct_array.len());
        assert_eq!(arrow_array.data_type(), struct_array.data_type());

        // Due to the iterate order probably be different, the direct assert will have possibility fail (in this case 50%)
        // but still need to check seperately whether it can be passed.
        // assert_eq!(arrow_array.into_data(), struct_array.into_data());
    }
}
