use arrow::array::{
    ArrayRef, DictionaryArray, ListArray, MapArray, StructArray
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::buffer::OffsetBuffer;
use std::collections::HashMap;
use std::sync::Arc;

use crate::array::primitive_array::NativeArray;
use crate::array::dispatch::dyn_scalar_vec_to_array;
use crate::datatype::DynScalar;
use crate::array::dispatch::default_dyn_scalar;

// ========== List Type Implementation ==========

/// A vector implementation for Arrow List arrays.
/// Stores nested vectors and converts them to Arrow ListArray format.
pub struct ListVec<T> {
    pub data: Vec<Vec<T>>,
    pub field: Field,
}

impl<T> ListVec<T> {
    /// Creates a new ListVec with the specified field schema.
    pub fn new(field: Field) -> Self {
        Self {
            data: Vec::new(),
            field,
        }
    }

    /// Creates a new ListVec from existing data and field schema.
    pub fn from_vec_with_field(data: Vec<Vec<T>>, field: Field) -> Self {
        Self { data, field }
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
        self.data.push(item);
    }

    fn get(&self, index: usize) -> Self::ItemRef<'_> {
        &self.data[index]
    }

    /// Converts this ListVec to an Arrow ListArray.
    fn to_arrow_array(&self) -> Self::ArrowArray {
        let mut offsets = vec![0i32];
        let mut values = Vec::new();
        
        for list in &self.data {
            offsets.push(offsets.last().unwrap() + list.len() as i32);
            for item in list {
                values.push(item.clone().into());
            }
        }
        
        let values_array = dyn_scalar_vec_to_array(values, self.field.data_type());
        let offsets_buffer = OffsetBuffer::new(offsets.into());
        ListArray::new(
            Arc::new(self.field.clone()),
            offsets_buffer,
            values_array,
            None,
        )
    }
}

// ========== Map Type Implementation ==========

/// A vector implementation for Arrow Map arrays.
/// Stores nested HashMaps and converts them to Arrow MapArray format.
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
        
        let keys_array = dyn_scalar_vec_to_array(keys, self.key_field.data_type());
        let values_array = dyn_scalar_vec_to_array(values, self.value_field.data_type());
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
        
        MapArray::new(
            Arc::new(entries_field),
            offsets_buffer,
            struct_array,
            None,
            false,
        )
    }
}

// ========== Dictionary Type Implementation ==========
// Note: Dictionary implementation is simplified for demonstration
// In a real implementation, you'd need proper handling of different key types

/// A vector implementation for Arrow Dictionary arrays.
/// Simplified implementation using i32 keys for demonstration.
pub struct DictionaryVec<V> {
    pub keys: Vec<i32>,  // Simplified to use i32 keys
    pub values: Vec<V>,
    pub value_field: Field,
}

impl<V> DictionaryVec<V> {
    /// Creates a new DictionaryVec with the specified value field schema.
    pub fn new(value_field: Field) -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            value_field,
        }
    }

    /// Creates a new DictionaryVec from existing keys, values, and field schema.
    pub fn from_vecs_with_field(keys: Vec<i32>, values: Vec<V>, value_field: Field) -> Self {
        Self {
            keys,
            values,
            value_field,
        }
    }
}

impl<V> NativeArray for DictionaryVec<V>
where
    V: Clone + Into<DynScalar> + 'static,
{
    type Item = (i32, V);
    type ItemRef<'a> = (&'a i32, &'a V);
    type ArrowArray = DictionaryArray<arrow::datatypes::Int32Type>;

    fn push(&mut self, item: Self::Item) {
        self.keys.push(item.0);
        self.values.push(item.1);
    }

    fn get(&self, index: usize) -> Self::ItemRef<'_> {
        (&self.keys[index], &self.values[index])
    }

    /// Converts this DictionaryVec to an Arrow DictionaryArray.
    fn to_arrow_array(&self) -> Self::ArrowArray {
        let values_dyn: Vec<DynScalar> = self.values.iter().cloned().map(Into::into).collect();
        let values_array = dyn_scalar_vec_to_array(values_dyn, self.value_field.data_type());
        
        // Create keys array
        let keys_array = arrow::array::Int32Array::from(self.keys.clone());
        
        DictionaryArray::try_new(
            keys_array,
            values_array,
        ).unwrap()
    }
}

// ========== FixedSizeList Type Implementation ==========

/// A vector implementation for Arrow FixedSizeList arrays.
/// Stores vectors of fixed size and converts them to Arrow FixedSizeListArray format.
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
            panic!("FixedSizeList size mismatch: expected {}, got {}", self.size, item.len());
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
        
        let values_array = dyn_scalar_vec_to_array(values, self.field.data_type());
        
        arrow::array::FixedSizeListArray::new(
            Arc::new(self.field.clone()),
            self.size,
            values_array,
            None,
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

    /// Creates a new StructVec from existing data and schema.
    pub fn from_vec_with_schema(data: Vec<HashMap<String, DynScalar>>, schema: Schema) -> Self {
        Self { data, schema }
    }
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
            let field_values: Vec<DynScalar> = self.data.iter()
                .map(|row| row.get(field_name).cloned().unwrap_or_else(|| default_dyn_scalar(field.data_type())))
                .collect();
            
            let array = dyn_scalar_vec_to_array(field_values, field.data_type());
            field_arrays.push(array);
        }
        for (field, arr) in self.schema.fields().iter().zip(field_arrays.iter()) {
            assert_eq!(arr.data_type(), field.data_type(), "Field {:?} expects {:?}, got {:?}", field.name(), field.data_type(), arr.data_type());
        }
        
        StructArray::new(
            self.schema.fields().clone(),
            field_arrays,
            None,
        )
    }
}


// ========== TESTS ==========

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::*;
    use std::collections::HashMap;

    #[test]
    fn test_list_vec_nested_struct() {
        // Create a nested struct containing a list of integers
        let inner_field = Field::new("item", DataType::Int32, false);
        let _list_field = Field::new("numbers", DataType::List(Arc::new(inner_field.clone())), false);
        
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
    fn test_list_vec_from_vec_with_field() {
        use arrow::datatypes::{DataType, Field};
        use std::sync::Arc;
        use arrow::array::{Int32Array, ListArray};

        // 1. 构造vec数据和Field
        let data = vec![
            vec![1, 2, 3],
            vec![4, 5],
            vec![],
            vec![6],
        ];
        let inner_field = Field::new("item", DataType::Int32, false);

        // 2. 调用 from_vec_with_field
        let list_vec = ListVec::from_vec_with_field(data.clone(), inner_field.clone());

        // 3. 调用 to_arrow_array
        let arrow_array = list_vec.to_arrow_array();
        assert_eq!(arrow_array.len(), data.len());

        // 4. 用Arrow ListArray手工构造期望
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

        // 5. 断言
        assert_eq!(arrow_array, expected);
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
        
        // Add some test data
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
        
        // Test direct construction using arrow-rs
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
        
        // Add first row with i64 and list of integers
        let mut row1 = HashMap::new();
        row1.insert("id".to_string(), DynScalar::Int64(100));
        row1.insert("tags".to_string(), DynScalar::List(vec![
            DynScalar::Int32(1),
            DynScalar::Int32(2),
            DynScalar::Int32(3),
        ]));
        struct_vec.push(row1);
        
        // Add second row
        let mut row2 = HashMap::new();
        row2.insert("id".to_string(), DynScalar::Int64(200));
        row2.insert("tags".to_string(), DynScalar::List(vec![
            DynScalar::Int32(4),
            DynScalar::Int32(5),
        ]));
        struct_vec.push(row2);
        
        let arrow_array = struct_vec.to_arrow_array();
        
        // Test direct construction using arrow-rs for comparison
        let expected = {
            let id_array = Arc::new(Int64Array::from(vec![100, 200]));
            
            // Create the list array manually
            let list_values = Int32Array::from(vec![1, 2, 3, 4, 5]);
            let list_offsets = OffsetBuffer::new(vec![0, 3, 5].into());
            let list_field = Field::new("item", DataType::Int32, false);
            let tags_array = Arc::new(ListArray::new(
                Arc::new(list_field),
                list_offsets,
                Arc::new(list_values),
                None,
            ));
            
            StructArray::new(
                schema.fields().clone(),
                vec![id_array, tags_array],
                None,
            )
        };
        
        assert_eq!(arrow_array.len(), expected.len());
        assert_eq!(arrow_array.data_type(), expected.data_type());
        
        // Verify the data types of columns
        let id_column = arrow_array.column(0);
        let tags_column = arrow_array.column(1);
        
        assert_eq!(id_column.data_type(), &DataType::Int64);
        assert_eq!(tags_column.data_type(), &DataType::List(Arc::new(Field::new("item", DataType::Int32, false))));
        
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
    }

    #[test]
    fn test_nested_struct_with_map() {
        // Test a struct containing a Map field
        let map_field = Field::new(
            "entries",
            DataType::Struct(vec![
                Field::new("key", DataType::Utf8, false),
                Field::new("value", DataType::Int32, false),
            ].into()),
            false,
        );
        let fields = vec![
            Field::new("id", DataType::Int64, false),
            Field::new("metadata", DataType::Map(Arc::new(map_field), false), false),
        ];
        let schema = Schema::new(fields);
        
        let mut struct_vec = StructVec::new(schema.clone());
        
        // Add a row with map data
        let mut row1 = HashMap::new();
        row1.insert("id".to_string(), DynScalar::Int64(100));
        row1.insert("metadata".to_string(), DynScalar::Map(vec![
            (DynScalar::String("name".to_string()), DynScalar::Int32(42)),
            (DynScalar::String("age".to_string()), DynScalar::Int32(30)),
        ]));
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
    fn test_nested_struct_with_dictionary() {
        // Test a struct containing a Dictionary field
        let fields = vec![
            Field::new("id", DataType::Int64, false),
            Field::new("category", DataType::Dictionary(
                Box::new(DataType::Int32),
                Box::new(DataType::Utf8),
            ), false),
        ];
        let schema = Schema::new(fields);
        
        let mut struct_vec = StructVec::new(schema.clone());
        
        // Add a row with dictionary data
        // In a real scenario, you'd have one dictionary shared across multiple rows
        // but each row would have its own key index
        let mut row1 = HashMap::new();
        row1.insert("id".to_string(), DynScalar::Int64(100));
        row1.insert("category".to_string(), DynScalar::Dictionary(
            Box::new(DynScalar::Int32(0)), // Key index
            Box::new(DynScalar::String("red".to_string())), // The actual value
        ));
        struct_vec.push(row1.clone());
        struct_vec.push(row1.clone());
        
        let arrow_array = struct_vec.to_arrow_array();
        
        assert_eq!(arrow_array.len(), 2);
        assert_eq!(arrow_array.num_columns(), 2);
        
        // Verify the data types
        let id_column = arrow_array.column(0);
        let category_column = arrow_array.column(1);
        
        assert_eq!(id_column.data_type(), &DataType::Int64);
        assert!(matches!(category_column.data_type(), DataType::Dictionary(_, _)));
        
        // Verify the ID value
        let id_array = id_column.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(id_array.value(0), 100);
        
        // Verify the dictionary structure exists
        let dict_array = category_column.as_any().downcast_ref::<DictionaryArray<arrow::datatypes::Int32Type>>().unwrap();
        assert_eq!(dict_array.len(), 2); // 1 row with 1 key
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
    fn test_nested_struct_with_fixed_size_list() {
        // Test a struct containing a FixedSizeList field (using our implementation)
        let fields = vec![
            Field::new("id", DataType::Int64, false),
            Field::new("coordinates", DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float64, false)),
                3, // Fixed size of 3 (x, y, z coordinates)
            ), false),
        ];
        let schema = Schema::new(fields);
        
        let mut struct_vec = StructVec::new(schema.clone());
        
        // Add a row with fixed-size list data
        let mut row1 = HashMap::new();
        row1.insert("id".to_string(), DynScalar::Int64(100));
        row1.insert("coordinates".to_string(), DynScalar::FixedSizeList(
            vec![
                DynScalar::Float64(1.0),
                DynScalar::Float64(2.0),
                DynScalar::Float64(3.0),
            ],
            3,
        ));
        struct_vec.push(row1);
        
        let arrow_array = struct_vec.to_arrow_array();
        
        assert_eq!(arrow_array.len(), 1);
        assert_eq!(arrow_array.num_columns(), 2);
        
        // Verify the data types
        let id_column = arrow_array.column(0);
        let coordinates_column = arrow_array.column(1);
        
        assert_eq!(id_column.data_type(), &DataType::Int64);
        assert!(matches!(coordinates_column.data_type(), DataType::FixedSizeList(_, 3)));
        
        // Verify the ID value
        let id_array = id_column.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(id_array.value(0), 100);
        
        // Verify the fixed size list structure exists
        let coord_array = coordinates_column.as_any().downcast_ref::<arrow::array::FixedSizeListArray>().unwrap();
        assert_eq!(coord_array.len(), 1);
        assert_eq!(coord_array.value_length(), 3);
    }

    #[test]
    fn test_list_vec_nullable_arrow_expected() {
        use arrow::array::{Int32Builder, ListBuilder};
        use arrow::datatypes::{DataType, Field};

        let data: Vec<Option<Vec<Option<i32>>>> = vec![
            Some(vec![Some(1), None, Some(3)]),  // [1, null, 3]
            Some(vec![]),                        // []
            Some(vec![Some(4), Some(5)]),        // [4,5]
        ];

        let inner_field = Field::new("item", DataType::Int32, true);

        let mut builder = ListBuilder::with_field(ListBuilder::new(Int32Builder::new()), inner_field.clone());
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

        let input: Vec<Vec<DynScalar>> = data.iter()
            .filter_map(|opt| opt.as_ref().map(|v| v.iter().map(|vi| match vi {
                Some(x) => DynScalar::Int32(*x),
                None => DynScalar::Null
            }).collect()))
            .collect();

        let list_vec = ListVec::from_vec_with_field(input, inner_field);
        let arrow_array = list_vec.to_arrow_array();

        assert_eq!(arrow_array, expected); 

    }

}